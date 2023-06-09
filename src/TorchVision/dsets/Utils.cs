// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;

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
        public static partial class datasets
        {
            public static partial class utils
            {
                /// <summary>
                /// Download a Google Drive file and place it in root.
                /// </summary>
                /// <param name="file_id">id of file to be downloaded</param>
                /// <param name="root">Directory to place downloaded file in</param>
                /// <param name="filename">Name to save the file under. If null, use the id of the file.</param>
                /// <param name="md5">MD5 checksum of the download. If null, do not check</param>
                public static void download_file_from_google_drive(
                    string file_id, string root, string? filename = null, string? md5 = null)
                {
                    io.GDriveDownload.DownloadFileFromGoogleDrive(file_id, root, filename, md5);
                }

                private static void ExtractTar(string from_path, string to_path)
                {
                    using (var fileStream = File.OpenRead(from_path)) {
                        using (var inputStream = new GZipInputStream(fileStream)) {
                            using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(inputStream, Encoding.UTF8)) {
                                tarArchive.ExtractContents(to_path);
                            }
                        }
                    }
                }

                private static void ExtractZip(string from_path, string to_path)
                {
                    ZipFile.ExtractToDirectory(from_path, to_path);
                }

                /// <summary>
                /// Extract an archive.
                ///
                /// The archive type and a possible compression is automatically detected from the file name.If the file is compressed
                /// but not an archive the call is dispatched to :func:`decompress`.
                /// </summary>
                /// <param name="from_path">Path to the file to be extracted.</param>
                /// <param name="to_path"> Path to the directory the file will be extracted to.If omitted, the directory of the file is used.</param>
                /// <param name="remove_finished">If ``True``, remove the file after the extraction.</param>
                /// <returns>Path to the directory the file was extracted to.</returns>
                /// <exception cref="InvalidDataException"></exception>
                /// <exception cref="ArgumentException"></exception>
                public static string extract_archive(string from_path, string? to_path = null, bool remove_finished = false)
                {
                    if (to_path is null) {
                        to_path = Path.GetDirectoryName(from_path);
                        if (to_path is null) {
                            throw new InvalidDataException();
                        }
                    }

                    if (Path.GetExtension(from_path) == ".zip") {
                        ExtractZip(from_path, to_path);
                    } else if (from_path.EndsWith(".tar.gz")) {
                        ExtractTar(from_path, to_path);
                    } else {
                        throw new ArgumentException();
                    }

                    if (remove_finished) {
                        File.Delete(from_path);
                    }

                    return to_path;
                }

                private static string IterableToStr(IList<string> iterable)
                {
                    return "'" + string.Join("', '", iterable) + "'";
                }

                internal static string VerifyStrArg(
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
                                value, arg, IterableToStr(valid_values));
                        }
                        throw new ArgumentException(msg);
                    }

                    return value;
                }
            }
        }
    }
}