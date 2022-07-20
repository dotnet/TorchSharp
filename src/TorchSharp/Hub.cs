// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class hub
        {
            /// <summary>
            /// Download the url to a file 
            /// </summary>
            /// <param name="url">The URL to download</param>
            /// <param name="dst">The file path to download the URL into</param>
            /// <param name="hash_prefix">If non null, the SHA256 hash of downloaded content must match this prefix</param>
            /// <exception cref="InvalidDataException">SHA256 hash doesn't match</exception>
            public static void download_url_to_file(string url, string dst, string hash_prefix = null)
            {
                try {
                    Task.Run(async () => {
                        await download_url_to_file_async(url, dst, hash_prefix);
                    }).Wait();
                } catch (AggregateException ex) {
                    throw ex.InnerException;
                }
            }

            /// <summary>
            /// Download the url to a file 
            /// </summary>
            /// <param name="url">The URL to download</param>
            /// <param name="dst">The file path to download the URL into</param>
            /// <param name="cancellationToken">A cancellation token</param>
            /// <param name="hash_prefix">If non null, the SHA256 hash of downloaded content must match this prefix</param>
            /// <exception cref="InvalidDataException">SHA256 hash doesn't match</exception>
            public static async Task download_url_to_file_async(string url, string dst, string hash_prefix = null, CancellationToken cancellationToken = default)
            {
                try {
                    using (var httpClient = new HttpClient()) {
                        using (var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken)) {
                            response.EnsureSuccessStatusCode();
                            using (var writer = File.OpenWrite(dst)) {
                                await response.Content.CopyToAsync(writer);
                            }
                        }
                    }
                } catch (Exception) {
                    if (File.Exists(dst)) {
                        File.Delete(dst);
                    }
                    throw;
                }

                string fileHash = GetFileChecksum(dst);
                if (hash_prefix != null && !fileHash.StartsWith(hash_prefix)) {
                    throw new InvalidDataException($"invalid hash value (expected \"{hash_prefix}\", got \"{fileHash}\"");
                }
            }

            private static string GetFileChecksum(string path)
            {
                using (SHA256 sha256 = SHA256.Create()) {
                    using (var stream = File.OpenRead(path)) {
                        var hashValue = sha256.ComputeHash(stream);
                        var sb = new StringBuilder();
                        foreach (var value in hashValue) {
                            sb.Append($"{value:x2}");
                        }
                        return sb.ToString();
                    }
                }
            }
        }
    }
}