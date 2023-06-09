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
            private static Func<bool, IProgressBar> createProgressBarFunc = null;

            /// <summary>
            /// Set a function to create a progress bar.
            /// </summary>
            /// <param name="func">A progress bar function or null to set default the console progress bar</param>
            public static void set_create_progress_bar_func(Func<bool, IProgressBar> func)
            {
                createProgressBarFunc = func;
            }

            /// <summary>
            /// Create a progress bar.
            /// </summary>
            /// <param name="hidden">Make a hidden progress bar.</param>
            /// <returns>A progress bar</returns>
            public static IProgressBar CreateProgressBar(bool hidden)
            {
                IProgressBar progress_bar;
                if (createProgressBarFunc == null) {
                    progress_bar = new ConsoleProgressBar(hidden);
                } else {
                    progress_bar = createProgressBarFunc(hidden);
                }
                return progress_bar;
            }

            /// <summary>
            /// Download the url to a file 
            /// </summary>
            /// <param name="url">The URL to download</param>
            /// <param name="dst">The file path to download the URL into</param>
            /// <param name="hash_prefix">If non null, the SHA256 hash of downloaded content must match this prefix</param>
            /// <param name="progress">whether or not to display a progress bar to stderr</param>
            /// <exception cref="InvalidDataException">SHA256 hash doesn't match</exception>
            public static void download_url_to_file(string url, string dst, string hash_prefix = null, bool progress = true)
            {
                try {
                    Task.Run(async () => {
                        await download_url_to_file_async(url, dst, hash_prefix, progress);
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
            /// <param name="progress">whether or not to display a progress bar to stderr</param>
            /// <exception cref="InvalidDataException">SHA256 hash doesn't match</exception>
            public static async Task download_url_to_file_async(string url, string dst, string hash_prefix = null, bool progress = true, CancellationToken cancellationToken = default)
            {
                try {
                    using (var progress_bar = CreateProgressBar(!progress))
                    using (var httpClient = new HttpClient())
                    using (var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken)) {
                        response.EnsureSuccessStatusCode();
                        progress_bar.Maximum = response.Content.Headers.ContentLength;
                        using (var writer = File.OpenWrite(dst)) {
                            byte[] buffer = new byte[64 * 1024];
                            if (progress) {
                                var reader = await response.Content.ReadAsStreamAsync();
                                while (true) {
                                    int ret = await reader.ReadAsync(buffer, 0, buffer.Length, cancellationToken);
                                    if (ret == 0) break;
                                    await writer.WriteAsync(buffer, 0, ret, cancellationToken);
                                    progress_bar.Value += ret;
                                }
                            } else {
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