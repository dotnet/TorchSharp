using Torch.IO;
using Xunit;

namespace AtenSharp.Test
{
    public class DiskFiles
    {
        [Fact]
        public void CreateWritableDiskFile()
        {
            var file = new DiskFile("test1.dat", "w");

            Assert.False(file.CanRead);
            Assert.True(file.CanWrite);
            Assert.False(file.IsBinary);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void CreateReadWritableDiskFile()
        {
            var file = new DiskFile("test2.dat", "rwb");

            Assert.True(file.CanRead);
            Assert.True(file.CanWrite);
            Assert.True(file.IsBinary);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void CreateQuietDiskFile()
        {
            var file = new DiskFile("test1q.dat", "w");

            Assert.False(file.IsQuiet);
            file.IsQuiet = true;
            Assert.True(file.IsQuiet);
            file.IsQuiet = false;
            Assert.False(file.IsQuiet);

            file.Close();
            Assert.False(file.IsOpen);
        }


        [Fact]
        public void CreateAutoSpacingDiskFile()
        {
            var file = new DiskFile("test1as.dat", "w");

            Assert.True(file.IsAutoSpacing);
            file.IsAutoSpacing = false;
            Assert.False(file.IsAutoSpacing);
            file.IsAutoSpacing = true;
            Assert.True(file.IsAutoSpacing);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteByteToDiskFile()
        {
            var file = new DiskFile("test3.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteByte(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadByteViaDiskFile()
        {
            var file = new DiskFile("test4.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteByte(13);
            file.WriteByte(17);

            file.Seek(0);
            var rd = file.ReadByte();
            Assert.Equal(13,rd);
            rd = file.ReadByte();
            Assert.Equal(17, rd);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteCharToDiskFile()
        {
            var file = new DiskFile("test3c.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteChar(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadCharViaDiskFile()
        {
            var file = new DiskFile("test4c.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteChar(13);
            file.WriteChar(17);

            file.Seek(0);
            var rd = file.ReadChar();
            Assert.Equal(13, rd);
            rd = file.ReadChar();
            Assert.Equal(17, rd);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteShortToDiskFile()
        {
            var file = new DiskFile("test5.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteShort(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadShortViaDiskFile()
        {
            var file = new DiskFile("test6.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteShort(13);
            file.WriteShort(17);
            file.Seek(0);
            var rd = file.ReadShort();
            Assert.Equal(13, rd);
            rd = file.ReadShort();
            Assert.Equal(17, rd);
        }

        [Fact]
        public void WriteIntToDiskFile()
        {
            var file = new DiskFile("test7.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteInt(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadIntViaDiskFile()
        {
            var file = new DiskFile("test8.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteInt(13);
            file.WriteInt(17);
            file.Seek(0);
            var rd = file.ReadInt();
            Assert.Equal(13, rd);
            rd = file.ReadInt();
            Assert.Equal(17, rd);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteLongToDiskFile()
        {
            var file = new DiskFile("test9.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteLong(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadLongViaDiskFile()
        {
            var file = new DiskFile("testA.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteLong(13);
            file.WriteLong(17);
            file.Seek(0);
            var rd = file.ReadLong();
            Assert.Equal(13, rd);
            rd = file.ReadLong();
            Assert.Equal(17, rd);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteFloatToDiskFile()
        {
            var file = new DiskFile("testB.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteFloat(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadFloatViaDiskFile()
        {
            var file = new DiskFile("testC.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteFloat(13);
            file.WriteFloat(17);
            file.Seek(0);
            var rd = file.ReadFloat();
            Assert.Equal(13, rd);
            rd = file.ReadFloat();
            Assert.Equal(17, rd);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteDoubleToDiskFile()
        {
            var file = new DiskFile("testD.dat", "wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteDouble(17);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadDoubleViaDiskFile()
        {
            var file = new DiskFile("testE.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteDouble(13);
            file.WriteDouble(17);
            file.Seek(0);
            var rd = file.ReadDouble();
            Assert.Equal(13, rd);
            rd = file.ReadDouble();
            Assert.Equal(17, rd);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageBytesViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test15.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new ByteTensor.ByteStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = (byte)i;
            }

            file.WriteBytes(storage0);
            Assert.Equal(size*sizeof(byte), file.Position);
            file.Seek(0);

            var storage1 = new ByteTensor.ByteStorage(size);
            var rd = file.ReadBytes(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageCharsViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test15c.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new MemoryFile.CharStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = (byte)i;
            }

            file.WriteChars(storage0);
            Assert.Equal(size * sizeof(byte), file.Position);
            file.Seek(0);

            var storage1 = new MemoryFile.CharStorage(size);
            var rd = file.ReadChars(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageShortsViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test16.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new ShortTensor.ShortStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = (short)i;
            }

            file.WriteShorts(storage0);
            Assert.Equal(size*sizeof(short), file.Position);
            file.Seek(0);

            var storage1 = new ShortTensor.ShortStorage(size);
            var rd = file.ReadShorts(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(short), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageIntsViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test17.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new IntTensor.IntStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = i;
            }

            file.WriteInts(storage0);
            Assert.Equal(size*sizeof(int), file.Position);
            file.Seek(0);

            var storage1 = new IntTensor.IntStorage(size);
            var rd = file.ReadInts(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(int), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageLongsViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test18.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new LongTensor.LongStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = i;
            }

            file.WriteLongs(storage0);
            Assert.Equal(size*sizeof(long), file.Position);
            file.Seek(0);

            var storage1 = new LongTensor.LongStorage(size);
            var rd = file.ReadLongs(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(long), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageFloatsViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test19.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new FloatTensor.FloatStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = (float)i;
            }

            file.WriteFloats(storage0);
            Assert.Equal(size*sizeof(float), file.Position);
            file.Seek(0);

            var storage1 = new FloatTensor.FloatStorage(size);
            var rd = file.ReadFloats(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(float), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadStorageDoublesViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test1A.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var storage0 = new DoubleTensor.DoubleStorage(size);
            for (var i = 0; i < size; ++i)
            {
                storage0[i] = (double)i;
            }

            file.WriteDoubles(storage0);
            Assert.Equal(size*sizeof(double), file.Position);
            file.Seek(0);

            var storage1 = new DoubleTensor.DoubleStorage(size);
            var rd = file.ReadDoubles(storage1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(double), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(storage0[i], storage1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadBytesViaDiskFile()
        {
            var file = new DiskFile("testF.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new byte[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteBytes(data0);
            //Assert.Equal(data0.Length*sizeof(byte), file.Position);
            file.Seek(0);

            var data1 = new byte[data0.Length];
            var rd = file.ReadBytes(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            //Assert.Equal(data0.Length * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadShortsViaDiskFile()
        {
            var file = new DiskFile("test10.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new short[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteShorts(data0);
            Assert.Equal(data0.Length * sizeof(short), file.Position);
            file.Seek(0);

            var data1 = new short[data0.Length];
            var rd = file.ReadShorts(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            Assert.Equal(data0.Length * sizeof(short), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadIntsViaDiskFile()
        {
            var file = new DiskFile("test11.dat", "rwb");
            file.UseNativeEndianEncoding();
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            Assert.True(file.IsBinary);

            var data0 = new int[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = i+32000000;
            file.WriteInts(data0);
            file.Flush();
            Assert.Equal(data0.Length * sizeof(int), file.Position);
            file.Seek(0);

            var data1 = new int[data0.Length];
            var rd = file.ReadInts(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            Assert.Equal(data0.Length * sizeof(int), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadFloatsViaDiskFile()
        {
            var file = new DiskFile("test13.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new float[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (float)i;
            file.WriteFloats(data0);
            Assert.Equal(data0.Length * sizeof(float), file.Position);
            file.Seek(0);

            var data1 = new float[data0.Length];
            var rd = file.ReadFloats(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            Assert.Equal(data0.Length * sizeof(float), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadLongsViaDiskFile()
        {
            var file = new DiskFile("test12.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new long[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = i;
            file.WriteLongs(data0);
            Assert.Equal(data0.Length * sizeof(long), file.Position);
            file.Seek(0);

            var data1 = new long[data0.Length];
            var rd = file.ReadLongs(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            Assert.Equal(data0.Length * sizeof(long), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadDoublesViaDiskFile()
        {
            var file = new DiskFile("test14.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new double[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (double)i;
            file.WriteDoubles(data0);
            Assert.Equal(data0.Length * sizeof(double), file.Position);
            file.Seek(0);

            var data1 = new double[data0.Length];
            var rd = file.ReadDoubles(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            Assert.Equal(data0.Length * sizeof(double), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadByteTensorViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test1B.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var tensor0 = new ByteTensor(size);
            for (var i = 0; i < size; ++i)
            {
                tensor0[i] = (byte)i;
            }

            file.WriteTensor(tensor0);
            Assert.Equal(size*sizeof(byte), file.Position);
            file.Seek(0);

            var tensor1 = new ByteTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadShortTensorViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test1C.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var tensor0 = new ShortTensor(size);
            for (var i = 0; i < size; ++i)
            {
                tensor0[i] = (short)i;
            }

            file.WriteTensor(tensor0);
            Assert.Equal(size*sizeof(short), file.Position);
            file.Seek(0);

            var tensor1 = new ShortTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(short), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadIntTensorViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test1D.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var tensor0 = new IntTensor(size);
            for (var i = 0; i < size; ++i)
            {
                tensor0[i] = (int)i;
            }

            file.WriteTensor(tensor0);
            Assert.Equal(size*sizeof(int), file.Position);
            file.Seek(0);

            var tensor1 = new IntTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(int), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadLongTensorViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test1E.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var tensor0 = new LongTensor(size);
            for (var i = 0; i < size; ++i)
            {
                tensor0[i] = (long)i;
            }

            file.WriteTensor(tensor0);
            Assert.Equal(size*sizeof(long), file.Position);
            file.Seek(0);

            var tensor1 = new LongTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(long), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadFloatTensorViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test1F.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var tensor0 = new FloatTensor(size);
            for (var i = 0; i < size; ++i)
            {
                tensor0[i] = (float)i;
            }

            file.WriteTensor(tensor0);
            Assert.Equal(size*sizeof(float), file.Position);
            file.Seek(0);

            var tensor1 = new FloatTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(float), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadDoubleTensorViaDiskFile()
        {
            const int size = 10;

            var file = new DiskFile("test20.dat", "rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var tensor0 = new DoubleTensor(size);
            for (var i = 0; i < size; ++i)
            {
                tensor0[i] = (double)i;
            }

            file.WriteTensor(tensor0);
            Assert.Equal(size*sizeof(double), file.Position);
            file.Seek(0);

            var tensor1 = new DoubleTensor(size);
            var rd = file.ReadTensor(tensor1);

            Assert.Equal(rd, size);
            Assert.Equal(size * sizeof(double), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(tensor1[i], tensor1[i]);
            }

            file.Close();
            Assert.False(file.IsOpen);
        }
    }
}
