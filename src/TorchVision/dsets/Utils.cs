// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using System.Web;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class datasets
        {
            public static partial class utils
            {
                internal static string calculate_md5(string fpath)
                {
                    MD5 md5 = MD5.Create();
                    byte[] hash;
                    using (var stream = File.OpenRead(fpath))
                    {
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

                internal static void download_file_from_google_drive(
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
                    throw new NotImplementedException();
                }

                internal static string extract_archive(string from_path, string? to_path = null, bool remove_finished = false)
                {
                    throw new NotImplementedException();
                }
            }
        }
    }
}
