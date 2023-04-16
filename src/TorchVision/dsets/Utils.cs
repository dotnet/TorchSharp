// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.NetworkInformation;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web;
using Tensorboard;
using static Tensorboard.CostGraphDef.Types;

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/b78d98bb152ffb9c0c0f5365f59f475c70b1784e/torchvision/datasets/utils.py
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class datasets
        {
            public static partial class utils
            {
                private static async Task _save_response_content_async(
                    byte[] head,
                    Stream content,
                    string destination,
                    long? length = null,
                    CancellationToken cancellationToken = default)
                {
                    using (var fh = File.OpenWrite(destination)) {
                        await fh.WriteAsync(head, 0, head.Length, cancellationToken);
                        await content.CopyToAsync(fh);
                        // pbar.update(len(chunk))
                    }
                }

                internal static string calculate_md5(string fpath)
                {
                    MD5 md5 = MD5.Create();
                    byte[] hash;
                    using (var stream = File.OpenRead(fpath)) {
                        hash = md5.ComputeHash(stream);
                    }
                    return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                }

                internal static bool check_md5(string fpath, string md5)
                {
                    return md5 == calculate_md5(fpath);
                }

                internal static bool check_integrity(string fpath, string? md5 = null)
                {
                    if (!File.Exists(fpath)) {
                        return false;
                    }
                    if (md5 is null) {
                        return true;
                    }
                    return check_md5(fpath, md5);
                }

                private static async Task<(string, byte[], Stream)> _extract_gdrive_api_response_async(
                    HttpResponseMessage response,
                    int chunk_size = 32 * 1024,
                    CancellationToken cancellationToken = default)
                {
                    var stream = await response.Content.ReadAsStreamAsync();
                    var buf = new byte[chunk_size];
                    int pos = 0;
                    while (pos < chunk_size) {
                        int readLen = await stream.ReadAsync(buf, pos, chunk_size - pos, cancellationToken);
                        if (readLen == 0) break;
                        pos += readLen;
                    }
                    buf = buf.AsSpan(0, pos).ToArray();

                    string api_response = string.Empty;
                    string first_chunk = Encoding.ASCII.GetString(buf);
                    string pattern = "<title>Google Drive - (?<api_response>.+?)</title>";
                    var match = Regex.Match(first_chunk, pattern);
                    if (match.Success) {
                        api_response = match.Groups["api_response"].Value;
                    }
                    return (api_response, buf, stream);
                }

                /// <summary>
                /// Download a Google Drive file from  and place it in root.
                /// </summary>
                /// <param name="file_id">id of file to be downloaded</param>
                /// <param name="root">Directory to place downloaded file in</param>
                /// <param name="filename">Name to save the file under. If null, use the id of the file.</param>
                /// <param name="md5">MD5 checksum of the download. If null, do not check</param>
                public static void download_file_from_google_drive(
                    string file_id, string root, string? filename = null, string? md5 = null)
                {
                    try {
                        Task.Run(async () => {
                            await download_file_from_google_drive_async(file_id, root, filename, md5);
                        }).Wait();
                    } catch (AggregateException ex) {
                        throw ex.InnerException ?? ex;
                    }
                }

                internal static async Task download_file_from_google_drive_async(
                    string file_id, string root, string? filename = null, string? md5 = null,
                    CancellationToken cancellationToken = default)
                {
                    // Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
                    if (filename is null) {
                        filename = file_id;
                    }
                    var fpath = Path.Combine(root, filename);

                    Directory.CreateDirectory(root);

                    if (check_integrity(fpath, md5)) {
                        if (md5 is null) {
                            Console.Write($"Using downloaded file: {fpath}");
                        } else {
                            Console.Write($"Using downloaded and verified file: {fpath}");
                        }
                        return;
                    }

                    CookieContainer cookies = new CookieContainer();
                    HttpClientHandler handler = new HttpClientHandler();
                    handler.CookieContainer = cookies;

                    using (var httpClient = new HttpClient(handler)) {
                        string url = "https://drive.google.com/uc";

                        var queryString = HttpUtility.ParseQueryString(string.Empty);
                        queryString.Add("id", file_id);
                        queryString.Add("export", "download");
                        string query_url = url + "?" + queryString.ToString();
                        HttpResponseMessage? response = null;

                        try {
                            response = await httpClient.GetAsync(query_url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                            response.EnsureSuccessStatusCode();
                            var responseCookies = cookies.GetCookies(new Uri(url));
                            string token = string.Empty;
                            foreach (var obj in responseCookies) {
                                Cookie? kv = obj as Cookie;
                                if (kv != null && kv.Name.StartsWith("download_warning")) {
                                    token = kv.Value;
                                    Console.WriteLine("{0}={1}", kv.Name, kv.Value);
                                }
                            }
                            string api_response = string.Empty;
                            byte[]? head = null;
                            Stream? stream = null;
                            if (string.IsNullOrEmpty(token)) {
                                (api_response, head, stream) = await _extract_gdrive_api_response_async(response, cancellationToken: cancellationToken);
                                token = api_response == "Virus scan warning" ? "t" : string.Empty;
                            }

                            if (!string.IsNullOrEmpty(token)) {
                                queryString.Add("confirm", token);
                                query_url = url + "?" + queryString.ToString();
                                response.Dispose();
                                response = null;
                                response = await httpClient.GetAsync(query_url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                                (api_response, head, stream) = await _extract_gdrive_api_response_async(response);
                            }

                            if (head == null || stream == null) {
                                // unreachable
                                throw new InvalidDataException();
                            }

                            if (api_response == "Quota exceeded") {
                                throw new InvalidDataException(
                                    $"The daily quota of the file {filename} is exceeded and it " +
                                    "can't be downloaded. This is a limitation of Google Drive " +
                                    "and can only be overcome by trying again later.");
                            }

                            await _save_response_content_async(head, stream, fpath, cancellationToken: cancellationToken);
                        } finally {
                            if (response != null) {
                                response.Dispose();
                            }
                        }

                        // In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
                        if (new FileInfo(fpath).Length < 10 * 1024) {
                            string text = _ReadAllTextAsync(fpath);
                            // Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
                            if (Regex.Match(text, @"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)").Success) {
                                Console.WriteLine(
                                    $"We detected some HTML elements in the downloaded file. " +
                                    $"This most likely means that the download triggered an unhandled API response by GDrive. " +
                                    $"Please report this to torchvision at https://github.com/pytorch/vision/issues including " +
                                    $"the response:\n\n{text}"
                                );
                            }
                        }

                        if (md5 is not null && !check_md5(fpath, md5)) {
                            throw new InvalidDataException(
                                $"The MD5 checksum of the download file {fpath} does not match the one on record." +
                                $"Please delete the file and try again. " +
                                $"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues.");
                        }
                    }
                }

                private static string _ReadAllTextAsync(string path, CancellationToken cancellationToken = default)
                {
                    var memory = new MemoryStream();
                    using (var stream = File.OpenRead(path)) {
                        stream.CopyTo(memory);
                    }
                    return Encoding.ASCII.GetString(memory.GetBuffer());
                }

                internal static void _extract_tar(string from_path, string to_path)
                {
                    using (var fileStream = File.OpenRead(from_path)) {
                        using (var inputStream = new GZipInputStream(fileStream)) {
                            using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(inputStream, Encoding.UTF8)) {
                                tarArchive.ExtractContents(to_path);
                            }
                        }
                    }
                }

                internal static void _extract_zip(string from_path, string to_path)
                {
                    using (var fileStream = File.OpenRead(from_path)) {
                        using (ZipArchive zipArchive = new ZipArchive(fileStream)) {
                            zipArchive.ExtractToDirectory(to_path);
                        }
                    }
                }

                internal static string extract_archive(string from_path, string? to_path = null, bool remove_finished = false)
                {
                    if (to_path is null) {
                        to_path = Path.GetDirectoryName(from_path);
                        if (to_path is null) {
                            throw new InvalidDataException();
                        }
                    }

                    if (Path.GetExtension(from_path) != "zip") {
                        _extract_zip(from_path, to_path);
                    }
                    if (from_path.EndsWith(".tar.gz")) {
                        _extract_tar(from_path, to_path);
                    } else {
                        throw new ArgumentException();
                    }

                    if (remove_finished) {
                        File.Delete(from_path);
                    }

                    return to_path;
                }

                private static string iterable_to_str(IList<string> iterable)
                {
                    return "'" + string.Join("', '", iterable) + "'";
                }

                internal static string verify_str_arg(
                    string value,
                    string? arg = null,
                    IList<string>? valid_values = null,
                    string? custom_msg = null)
                {
                    if (valid_values is null) {
                        return value;
                    }

                    if (!valid_values.Contains(value)) {
                        string msg;
                        if (custom_msg is not null) {
                            msg = custom_msg;
                        } else {
                            msg = string.Format(
                                "Unknown value '{0}' for argument {1}. Valid values are {2}.",
                                value, arg, iterable_to_str(valid_values));
                        }
                        throw new ArgumentException(msg);
                    }

                    return value;
                }
            }
        }
    }
}
