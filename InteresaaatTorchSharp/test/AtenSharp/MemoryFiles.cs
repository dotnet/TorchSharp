using Microsoft.VisualStudio.TestTools.UnitTesting;
using Torch.IO;

namespace AtenSharp.Test
{
    [TestClass]
    public class MemoryFiles
    {
        [TestMethod]
        public void CreateStorage()
        {
            var storage = new MemoryFile.CharStorage(256);
            Assert.IsNotNull(storage);
        }

        [TestMethod]
        public void CreateWritableMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);
            Assert.IsFalse(file.IsBinary);
            Assert.IsTrue(file.IsAscii);

            file.Close();
            Assert.IsFalse(file.IsOpen);

            file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);
            Assert.IsTrue(file.IsBinary);
            Assert.IsFalse(file.IsAscii);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void CreateReadableMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("r");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanRead);
            Assert.IsFalse(file.CanWrite);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }


        [TestMethod]
        public void CreateReadWritableMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanRead);
            Assert.IsTrue(file.CanWrite);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void SeekInMemoryFile()
        {
            var storage = new MemoryFile.CharStorage(256);
            Assert.IsNotNull(storage);
            storage.Fill(0);
            var file = new Torch.IO.MemoryFile(storage, "w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);

            file.Seek(150);
            Assert.AreEqual(150, file.Position);

            file.Seek(15);
            Assert.AreEqual(15, file.Position);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void SeekToEndInMemoryFile()
        {
            var storage = new MemoryFile.CharStorage(256);
            Assert.IsNotNull(storage);
            storage.Fill(0);
            var file = new Torch.IO.MemoryFile(storage, "w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);

            file.SeekEnd();
            Assert.AreEqual(255, file.Position);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteByteToMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(17);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadByteViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(13);
            file.WriteByte(17);
            file.Seek(0);
            var rd = file.ReadByte();
            Assert.AreEqual(13, rd);
            rd = file.ReadByte();
            Assert.AreEqual(17, rd);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteShortToMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadShortViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteShort(13);
            file.WriteShort(17);
            file.Seek(0);
            var rd = file.ReadShort();
            Assert.AreEqual(13, rd);
            rd = file.ReadShort();
            Assert.AreEqual(17, rd);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteIntToMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteInt(17);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadIntViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteLongToMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteLong(17);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadLongViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteFloatToMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteFloat(17);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadFloatViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteDoubleToMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("wb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteDouble(17);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadDoubleViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteAndReadByteTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var tensor0 = new AtenSharp.ByteTensor(size);
            for (var i = 0; i < size; ++i) 
            {
                tensor0[i] = (byte)i;
            }

            file.WriteTensor(tensor0);
            Assert.AreEqual(size*sizeof(byte), file.Position);
            file.Seek(0);
        
            var tensor1 = new AtenSharp.ByteTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadShortTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var tensor0 = new AtenSharp.ShortTensor(size);
            for (var i = 0; i < size; ++i) 
            {
                tensor0[i] = (short)i;
            }

            file.WriteTensor(tensor0);
            Assert.AreEqual(size*sizeof(short), file.Position);
            file.Seek(0);
        
            var tensor1 = new AtenSharp.ShortTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(short), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadIntTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var tensor0 = new AtenSharp.IntTensor(size);
            for (var i = 0; i < size; ++i) 
            {
                tensor0[i] = (int)i;
            }

            file.WriteTensor(tensor0);
            Assert.AreEqual(size*sizeof(int), file.Position);
            file.Seek(0);
        
            var tensor1 = new AtenSharp.IntTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(int), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadLongTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var tensor0 = new AtenSharp.LongTensor(size);
            for (var i = 0; i < size; ++i) 
            {
                tensor0[i] = (long)i;
            }

            file.WriteTensor(tensor0);
            Assert.AreEqual(size*sizeof(long), file.Position);
            file.Seek(0);
        
            var tensor1 = new AtenSharp.LongTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(long), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadFloatTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var tensor0 = new AtenSharp.FloatTensor(size);
            for (var i = 0; i < size; ++i) 
            {
                tensor0[i] = (float)i;
            }

            file.WriteTensor(tensor0);
            Assert.AreEqual(size*sizeof(float), file.Position);
            file.Seek(0);
        
            var tensor1 = new AtenSharp.FloatTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(float), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadDoubleTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var tensor0 = new AtenSharp.DoubleTensor(size);
            for (var i = 0; i < size; ++i) 
            {
                tensor0[i] = (double)i;
            }

            file.WriteTensor(tensor0);
            Assert.AreEqual(size*sizeof(double), file.Position);
            file.Seek(0);
        
            var tensor1 = new AtenSharp.DoubleTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.AreEqual(rd, size);
            Assert.AreEqual(size * sizeof(double), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }
        
        [TestMethod]
        public void WriteAndReadStorageBytesViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new AtenSharp.ByteTensor.ByteStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (byte)i;
            }

            file.WriteBytes(storage0);
            Assert.AreEqual(size*sizeof(byte), file.Position);
            file.Seek(0);

            var storage1 = new AtenSharp.ByteTensor.ByteStorage(size);
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
        public void WriteAndReadStorageShortsViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new AtenSharp.ShortTensor.ShortStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (short)i;
            }

            file.WriteShorts(storage0);
            Assert.AreEqual(size*sizeof(short), file.Position);
            file.Seek(0);

            var storage1 = new AtenSharp.ShortTensor.ShortStorage(size);
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
        public void WriteAndReadStorageIntsViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new AtenSharp.IntTensor.IntStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = i;
            }

            file.WriteInts(storage0);
            Assert.AreEqual(size*sizeof(int), file.Position);
            file.Seek(0);

            var storage1 = new AtenSharp.IntTensor.IntStorage(size);
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
        public void WriteAndReadStorageLongsViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new AtenSharp.LongTensor.LongStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = i;
            }

            file.WriteLongs(storage0);
            Assert.AreEqual(size*sizeof(long), file.Position);
            file.Seek(0);

            var storage1 = new AtenSharp.LongTensor.LongStorage(size);
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
        public void WriteAndReadStorageFloatsViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new AtenSharp.FloatTensor.FloatStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (float)i;
            }

            file.WriteFloats(storage0);
            Assert.AreEqual(size*sizeof(float), file.Position);
            file.Seek(0);

            var storage1 = new AtenSharp.FloatTensor.FloatStorage(size);
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
        public void WriteAndReadStorageDoublesViaMemoryFile()
        {
            const int size = 10;

            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var storage0 = new AtenSharp.DoubleTensor.DoubleStorage(size);
            for (var i = 0; i < size; ++i) 
            {
                storage0[i] = (double)i;
            }

            file.WriteDoubles(storage0);
            Assert.AreEqual(size*sizeof(double), file.Position);
            file.Seek(0);

            var storage1 = new AtenSharp.DoubleTensor.DoubleStorage(size);
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
        public void WriteAndReadBytesViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new byte[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteBytes(data0);
            Assert.AreEqual(data0.Length*sizeof(byte), file.Position);
            file.Seek(0);

            var data1 = new byte[data0.Length];
            var rd = file.ReadBytes(data1, data0.Length);

            Assert.AreEqual(rd, data0.Length);
            Assert.AreEqual(data0.Length * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.AreEqual(data0[i], data1[i]);
            }
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadShortsViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteAndReadIntsViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);

            var data0 = new int[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteInts(data0);
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
        public void WriteAndReadFloatsViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteAndReadLongsViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
        public void WriteAndReadDoublesViaMemoryFile()
        {
            var file = new Torch.IO.MemoryFile("rwb");
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
