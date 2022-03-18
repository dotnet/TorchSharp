using System;
using System.IO;
using Xunit;

namespace FileRestitcherTests
{
    public class Tests
    {
        //[Fact]
        public void TestRestitchingSmallFiles()
        {
            long size = 1024; // * 1024 * 1024;   // enable this to test massive files, takes a long time
            var oneChunk = new byte[size];
            var rnd = new System.Random();
            for (int i = 0; i < 10; i++)
                oneChunk[i] = (byte)rnd.Next(0,255);
            var tmpDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"/tmp";
            System.Console.WriteLine("tmpDir = {0}", tmpDir);
            if (Directory.Exists(tmpDir))
                Directory.Delete(tmpDir, true);
            Directory.CreateDirectory(tmpDir);
            Directory.CreateDirectory(tmpDir + @"/some-package-primary/runtimes");
            Directory.CreateDirectory(tmpDir + @"/some-package-fragment1/fragments");
            Directory.CreateDirectory(tmpDir + @"/some-package-fragment2/fragments");
            Directory.CreateDirectory(tmpDir + @"/some-package-fragment3/fragments");
            File.WriteAllBytes(tmpDir + @"/some-package-primary/runtimes/a.so", oneChunk);
            File.WriteAllBytes(tmpDir + @"/some-package-fragment1/fragments/a.so.fragment1", oneChunk);
            File.WriteAllBytes(tmpDir + @"/some-package-fragment2/fragments/a.so.fragment2", oneChunk);
            File.WriteAllBytes(tmpDir + @"/some-package-fragment3/fragments/a.so.fragment3", oneChunk);
            System.Threading.Thread.Sleep(1000);
            using (var sha256Hash = System.Security.Cryptography.SHA256.Create()) {
                var expectedFile = tmpDir + @"/a.so";
                Console.WriteLine("Writing restored primary file at {0}", expectedFile);
                var os = File.OpenWrite(expectedFile);
                os.Write(oneChunk, 0, oneChunk.Length);
                os.Write(oneChunk, 0, oneChunk.Length);
                os.Write(oneChunk, 0, oneChunk.Length);
                os.Write(oneChunk, 0, oneChunk.Length);
                os.Close();
                var os2 = File.OpenRead(expectedFile);
                byte[] bytes = sha256Hash.ComputeHash(os2);
                var builder = new System.Text.StringBuilder();
                for (int i = 0; i < bytes.Length; i++) {
                    builder.Append(bytes[i].ToString("x2"));
                }
                var sha = builder.ToString();
                File.WriteAllText(tmpDir + @"/some-package-primary/runtimes/a.so.sha", sha);
                os2.Close();
            }

            Assert.Equal(new FileInfo(tmpDir + @"/some-package-primary/runtimes/a.so").Length, 1* size);

            ConsoleApp2.Program.Restitch(tmpDir + @"/some-package-primary");
            Assert.True(File.Exists(tmpDir + @"/some-package-primary/runtimes/a.so"));
            Assert.Equal(new FileInfo(tmpDir + @"/some-package-primary/runtimes/a.so").Length, 4* size);
            Assert.False(File.Exists(tmpDir + @"/some-package-fragment1/fragments/a.so.fragment1"));
            Assert.False(File.Exists(tmpDir + @"/some-package-fragment2/fragments/a.so.fragment2"));
            Assert.False(File.Exists(tmpDir + @"/some-package-fragment3/fragments/a.so.fragment3"));
        }
    }

}
