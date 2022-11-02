// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.IO;
using System.Text;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            public static partial class utils
            {
                /// <summary>
                /// Extract an archive.
                /// </summary>
                /// <param name="from_path">The path of the archive</param>
                /// <param name="to_path">The path to extract the archive</param>
                internal static void extract_archive(string from_path, string to_path)
                {
                    using (var fileStream = File.OpenRead(from_path)) {
                        using (var inputStream = new GZipInputStream(fileStream)) {
                            using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(inputStream, Encoding.UTF8)) {
                                tarArchive.ExtractContents(to_path);
                            }
                        }
                    }
                }
            }
        }
    }
}