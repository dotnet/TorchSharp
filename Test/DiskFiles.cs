using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Torch.IO;

namespace Test
{
    [TestClass]
    public class DiskFiles
    {
        [TestMethod]
        public void CreateWritableDiskFile()
        {
            var file = new Torch.IO.DiskFile("test1.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);
            Assert.IsFalse(file.IsBinary);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void CreateReadWritableDiskFile()
        {
            var file = new Torch.IO.DiskFile("test2.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanRead);
            Assert.IsTrue(file.CanWrite);
            Assert.IsTrue(file.IsBinary);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteByteToDiskFile()
        {
            var file = new Torch.IO.DiskFile("test3.dat", "wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadByteViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test4.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(13);
            file.WriteByte(17);

            file.Seek(0);
            var rd = file.ReadByte();
            Assert.AreEqual(13,rd);
            rd = file.ReadByte();
            Assert.AreEqual(17, rd);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteShortToDiskFile()
        {
            var file = new Torch.IO.DiskFile("test5.dat", "wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteShort(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadShortViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test6.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteShort(13);
            file.WriteShort(17);
            file.Seek(0);
            var rd = file.ReadShort();
            Assert.AreEqual(13, rd);
            rd = file.ReadShort();
            Assert.AreEqual(17, rd);
        }

        [TestMethod]
        public void WriteIntToDiskFile()
        {
            var file = new Torch.IO.DiskFile("test7.dat", "wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteInt(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadIntViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test8.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteInt(13);
            file.WriteInt(17);
            file.Seek(0);
            var rd = file.ReadInt();
            Assert.AreEqual(13, rd);
            rd = file.ReadInt();
            Assert.AreEqual(17, rd);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteLongToDiskFile()
        {
            var file = new Torch.IO.DiskFile("test9.dat", "wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteLong(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadLongViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("testA.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteLong(13);
            file.WriteLong(17);
            file.Seek(0);
            var rd = file.ReadLong();
            Assert.AreEqual(13, rd);
            rd = file.ReadLong();
            Assert.AreEqual(17, rd);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteFloatToDiskFile()
        {
            var file = new Torch.IO.DiskFile("testB.dat", "wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteFloat(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadFloatViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("testC.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteFloat(13);
            file.WriteFloat(17);
            file.Seek(0);
            var rd = file.ReadFloat();
            Assert.AreEqual(13, rd);
            rd = file.ReadFloat();
            Assert.AreEqual(17, rd);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteDoubleToDiskFile()
        {
            var file = new Torch.IO.DiskFile("testD.dat", "wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteDouble(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadDoubleViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("testE.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteDouble(13);
            file.WriteDouble(17);
            file.Seek(0);
            var rd = file.ReadDouble();
            Assert.AreEqual(13, rd);
            rd = file.ReadDouble();
            Assert.AreEqual(17, rd);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageBytesViaDiskFile()
        {
            const int size = 10;

            var file = new Torch.IO.DiskFile("test15.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new TorchSharp.ByteTensor.ByteStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (byte)i;
            }

            file.WriteBytes(storage0);
            Assert.AreEqual(size*sizeof(byte), file.Position);
            file.Seek(0);

            var storage1 = new TorchSharp.ByteTensor.ByteStorage(size);
            var rd = file.ReadBytes(storage1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageShortsViaDiskFile()
        {
            const int size = 10;

            var file = new Torch.IO.DiskFile("test16.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new TorchSharp.ShortTensor.ShortStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (short)i;
            }

            file.WriteShorts(storage0);
            Assert.AreEqual(size*sizeof(short), file.Position);
            file.Seek(0);

            var storage1 = new TorchSharp.ShortTensor.ShortStorage(size);
            var rd = file.ReadShorts(storage1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(short), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageIntsViaDiskFile()
        {
            const int size = 10;

            var file = new Torch.IO.DiskFile("test17.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new TorchSharp.IntTensor.IntStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = i;
            }

            file.WriteInts(storage0);
            Assert.AreEqual(size*sizeof(int), file.Position);
            file.Seek(0);

            var storage1 = new TorchSharp.IntTensor.IntStorage(size);
            var rd = file.ReadInts(storage1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(int), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageLongsViaDiskFile()
        {
            const int size = 10;

            var file = new Torch.IO.DiskFile("test18.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new TorchSharp.LongTensor.LongStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = i;
            }

            file.WriteLongs(storage0);
            Assert.AreEqual(size*sizeof(long), file.Position);
            file.Seek(0);

            var storage1 = new TorchSharp.LongTensor.LongStorage(size);
            var rd = file.ReadLongs(storage1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(long), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageFloatsViaDiskFile()
        {
            const int size = 10;

            var file = new Torch.IO.DiskFile("test19.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new TorchSharp.FloatTensor.FloatStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (float)i;
            }

            file.WriteFloats(storage0);
            Assert.AreEqual(size*sizeof(float), file.Position);
            file.Seek(0);

            var storage1 = new TorchSharp.FloatTensor.FloatStorage(size);
            var rd = file.ReadFloats(storage1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(float), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageDoublesViaDiskFile()
        {
            const int size = 10;

            var file = new Torch.IO.DiskFile("test1A.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new TorchSharp.DoubleTensor.DoubleStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (double)i;
            }

            file.WriteDoubles(storage0);
            Assert.AreEqual(size*sizeof(double), file.Position);
            file.Seek(0);

            var storage1 = new TorchSharp.DoubleTensor.DoubleStorage(size);
            var rd = file.ReadDoubles(storage1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(double), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadBytesViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("testF.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new byte[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteBytes(data0);
            //Assert.AreEqual(data0.Length*sizeof(byte), file.Position);
            file.Seek(0);

            var data1 = new byte[data0.Length];
            var rd = file.ReadBytes(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            //Assert.AreEqual(data0.Length * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadShortsViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test10.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new short[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteShorts(data0);
            Assert.AreEqual(data0.Length * sizeof(short), file.Position);
            file.Seek(0);

            var data1 = new short[data0.Length];
            var rd = file.ReadShorts(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            Assert.AreEqual(data0.Length * sizeof(short), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadIntsViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test11.dat", "rwb");
            file.UseNativeEndianEncoding();
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            Assert.IsTrue(file.IsBinary);

            var data0 = new int[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = i+32000000;
            file.WriteInts(data0);
            file.Flush();
            Assert.AreEqual(data0.Length * sizeof(int), file.Position);
            file.Seek(0);

            var data1 = new int[data0.Length];
            var rd = file.ReadInts(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            Assert.AreEqual(data0.Length * sizeof(int), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadFloatsViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test13.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new float[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (float)i;
            file.WriteFloats(data0);
            Assert.AreEqual(data0.Length * sizeof(float), file.Position);
            file.Seek(0);

            var data1 = new float[data0.Length];
            var rd = file.ReadFloats(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            Assert.AreEqual(data0.Length * sizeof(float), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadLongsViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test12.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new long[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = i;
            file.WriteLongs(data0);
            Assert.AreEqual(data0.Length * sizeof(long), file.Position);
            file.Seek(0);

            var data1 = new long[data0.Length];
            var rd = file.ReadLongs(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            Assert.AreEqual(data0.Length * sizeof(long), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadDoublesViaDiskFile()
        {
            var file = new Torch.IO.DiskFile("test14.dat", "rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new double[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (double)i;
            file.WriteDoubles(data0);
            Assert.AreEqual(data0.Length * sizeof(double), file.Position);
            file.Seek(0);

            var data1 = new double[data0.Length];
            var rd = file.ReadDoubles(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            Assert.AreEqual(data0.Length * sizeof(double), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
    }
}
