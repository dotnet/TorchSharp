// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;

// This code was inspired by code found in the SciSharpStack-Examples repository located at:
//
//      https://github.com/SciSharp/SciSharp-Stack-Examples
//
// Original License: https://github.com/SciSharp/SciSharp-Stack-Examples/blob/master/LICENSE
//
// Original copyright information was not found at the above location.

namespace TorchSharp.Utils
{
    public static class Decompress
    {
        public static void DecompressGZipFile(string gzipFileName, string targetDir)
        {
            byte[] buf = new byte[4096];
            string fnOut = Path.Combine(targetDir, Path.GetFileNameWithoutExtension(gzipFileName));

            using (var fs = File.OpenRead(gzipFileName)) {
                // Check for GZIP magic bytes (0x1F, 0x8B) to detect whether the
                // file is actually gzip-compressed. Some HTTP servers or CDNs
                // transparently decompress .gz files during transfer, leaving
                // an uncompressed file with a .gz extension.
                var b1 = fs.ReadByte();
                var b2 = fs.ReadByte();
                fs.Position = 0;

                if (b1 == 0x1F && b2 == 0x8B) {
                    using (var gzipStream = new GZipInputStream(fs))
                    using (var fsOut = File.Create(fnOut)) {
                        StreamUtils.Copy(gzipStream, fsOut, buf);
                    }
                } else {
                    using (var fsOut = File.Create(fnOut)) {
                        StreamUtils.Copy(fs, fsOut, buf);
                    }
                }
            }
        }
        public static void ExtractTGZ(string gzArchiveName, string destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() => {
                using (var inStream = File.OpenRead(gzArchiveName)) {
                    using (var gzipStream = new GZipInputStream(inStream)) {
#pragma warning disable CS0618 // Type or member is obsolete
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
#pragma warning restore CS0618 // Type or member is obsolete
                            tarArchive.ExtractContents(destFolder);
                    }
                }
            });

            while (!task.IsCompleted) {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extraction completed.");
        }

    }
}
