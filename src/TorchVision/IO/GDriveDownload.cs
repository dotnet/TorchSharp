// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web;

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/b78d98bb152ffb9c0c0f5365f59f475c70b1784e/torchvision/datasets/utils.py
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class io
        {
            public class GDriveDownload
            {
                private static async Task SaveResponseContentAsync(
                    byte[] head,
                    Stream content,
                    string destination,
                    long? length = null,
                    CancellationToken cancellationToken = default)
                {
                    using (var progress_bar = torch.hub.CreateProgressBar(false))
                    using (var fh = File.OpenWrite(destination)) {
                        progress_bar.Maximum = length;
                        await fh.WriteAsync(head, 0, head.Length, cancellationToken);
                        progress_bar.Value = head.Length;
                        byte[] buffer = new byte[64 * 1024];
                        while (true) {
                            int ret = await content.ReadAsync(buffer, 0, buffer.Length, cancellationToken);
                            if (ret == 0) break;
                            await fh.WriteAsync(buffer, 0, ret, cancellationToken);
                            progress_bar.Value += ret;
                        }
                    }
                }

                private static string CalculateMD5(string fpath)
                {
                    MD5 md5 = MD5.Create();
                    byte[] hash;
                    using (var stream = File.OpenRead(fpath)) {
                        hash = md5.ComputeHash(stream);
                    }
                    return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                }

                private static bool CheckMD5(string fpath, string md5)
                {
                    return md5 == CalculateMD5(fpath);
                }

                /// <summary>
                /// Check if the file exists and verify its content with MD5.
                /// </summary>
                /// <param name="fpath">A path to the file</param>
                /// <param name="md5">MD5 checksum</param>
                /// <returns>True if the file exists and MD5 of its content matches</returns>
                public static bool CheckIntegrity(string fpath, string? md5 = null)
                {
                    if (!File.Exists(fpath)) {
                        return false;
                    }
                    if (md5 is null) {
                        return true;
                    }
                    return CheckMD5(fpath, md5);
                }

                private static async Task<(string, byte[], Stream)> ExtractGdriveApiResponseAsync(
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
                /// Download a Google Drive file and place it in root.
                /// </summary>
                /// <param name="file_id">id of file to be downloaded</param>
                /// <param name="root">Directory to place downloaded file in</param>
                /// <param name="filename">Name to save the file under. If null, use the id of the file.</param>
                /// <param name="md5">MD5 checksum of the download. If null, do not check</param>
                public static void DownloadFileFromGoogleDrive(
                    string file_id, string root, string? filename = null, string? md5 = null)
                {
                    try {
                        Task.Run(async () => {
                            await DownloadFileFromGoogleDriveAsync(file_id, root, filename, md5);
                        }).Wait();
                    } catch (AggregateException ex) {
                        throw ex.InnerException ?? ex;
                    }
                }

                /// <summary>
                /// Download a Google Drive file and place it in root.
                /// </summary>
                /// <param name="file_id">id of file to be downloaded</param>
                /// <param name="root">Directory to place downloaded file in</param>
                /// <param name="filename">Name to save the file under. If null, use the id of the file.</param>
                /// <param name="md5">MD5 checksum of the download. If null, do not check</param>
                /// <param name="cancellationToken">A cancellation token</param>
                public static async Task DownloadFileFromGoogleDriveAsync(
                    string file_id, string root, string? filename = null, string? md5 = null,
                    CancellationToken cancellationToken = default)
                {
                    // Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
                    if (filename is null) {
                        filename = file_id;
                    }
                    var fpath = Path.Combine(root, filename);

                    Directory.CreateDirectory(root);

                    if (CheckIntegrity(fpath, md5)) {
                        if (md5 is null) {
                            Console.WriteLine($"Using downloaded file: {fpath}");
                        } else {
                            Console.WriteLine($"Using downloaded and verified file: {fpath}");
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
                                }
                            }
                            string api_response = string.Empty;
                            byte[]? head = null;
                            Stream? stream = null;
                            if (string.IsNullOrEmpty(token)) {
                                (api_response, head, stream) = await ExtractGdriveApiResponseAsync(response, cancellationToken: cancellationToken);
                                token = api_response == "Virus scan warning" ? "t" : string.Empty;
                            }

                            if (!string.IsNullOrEmpty(token)) {
                                queryString.Add("confirm", token);
                                query_url = url + "?" + queryString.ToString();
                                response.Dispose();
                                response = null;
                                response = await httpClient.GetAsync(query_url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                                (api_response, head, stream) = await ExtractGdriveApiResponseAsync(response);
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

                            await SaveResponseContentAsync(head, stream, fpath, cancellationToken: cancellationToken);
                        } finally {
                            if (response != null) {
                                response.Dispose();
                            }
                        }

                        // In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
                        if (new FileInfo(fpath).Length < 10 * 1024) {
                            string text = ReadAllTextAsync(fpath);
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

                        if (md5 is not null && !CheckMD5(fpath, md5)) {
                            throw new InvalidDataException(
                                $"The MD5 checksum of the download file {fpath} does not match the one on record." +
                                $"Please delete the file and try again. " +
                                $"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues.");
                        }
                    }
                }

                private static string ReadAllTextAsync(string path, CancellationToken cancellationToken = default)
                {
                    var memory = new MemoryStream();
                    using (var stream = File.OpenRead(path)) {
                        stream.CopyTo(memory);
                    }
                    return Encoding.ASCII.GetString(memory.GetBuffer());
                }
            }
        }
    }
}