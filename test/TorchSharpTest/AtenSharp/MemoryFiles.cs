using Torch.IO;
using Xunit;

namespace AtenSharp.Test
{
    public class MemoryFiles
    {
        [Fact]
        public void CreateStorage()
        {
            var storage = new MemoryFile.CharStorage(256);
            Assert.NotNull(storage);
        }

        [Fact]
        public void CreateWritableMemoryFile()
        {
            var file = new MemoryFile("w");
            Assert.NotNull(file);
            Assert.False(file.CanRead);
            Assert.True(file.CanWrite);
            Assert.False(file.IsBinary);
            Assert.True(file.IsAscii);

            file.Close();
            Assert.False(file.IsOpen);

            file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.False(file.CanRead);
            Assert.True(file.CanWrite);
            Assert.True(file.IsBinary);
            Assert.False(file.IsAscii);

            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void CreateReadableMemoryFile()
        {
            var file = new MemoryFile("r");
            Assert.NotNull(file);
            Assert.True(file.CanRead);
            Assert.False(file.CanWrite);
            file.Close();
            Assert.False(file.IsOpen);
        }


        [Fact]
        public void CreateReadWritableMemoryFile()
        {
            var file = new MemoryFile("rw");
            Assert.NotNull(file);
            Assert.True(file.CanRead);
            Assert.True(file.CanWrite);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void SeekInMemoryFile()
        {
            var storage = new MemoryFile.CharStorage(256);
            Assert.NotNull(storage);
            storage.Fill(0);
            var file = new MemoryFile(storage, "w");
            Assert.NotNull(file);
            Assert.False(file.CanRead);
            Assert.True(file.CanWrite);

            file.Seek(150);
            Assert.Equal(150, file.Position);

            file.Seek(15);
            Assert.Equal(15, file.Position);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void SeekToEndInMemoryFile()
        {
            var storage = new MemoryFile.CharStorage(256);
            Assert.NotNull(storage);
            storage.Fill(0);
            var file = new MemoryFile(storage, "w");
            Assert.NotNull(file);
            Assert.False(file.CanRead);
            Assert.True(file.CanWrite);

            file.SeekEnd();
            Assert.Equal(255, file.Position);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteByteToMemoryFile()
        {
            var file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteByte(17);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadByteViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteByte(13);
            file.WriteByte(17);
            file.Seek(0);
            var rd = file.ReadByte();
            Assert.Equal(13, rd);
            rd = file.ReadByte();
            Assert.Equal(17, rd);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteShortToMemoryFile()
        {
            var file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadShortViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteShort(13);
            file.WriteShort(17);
            file.Seek(0);
            var rd = file.ReadShort();
            Assert.Equal(13, rd);
            rd = file.ReadShort();
            Assert.Equal(17, rd);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteIntToMemoryFile()
        {
            var file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteInt(17);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadIntViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteLongToMemoryFile()
        {
            var file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteLong(17);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadLongViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteFloatToMemoryFile()
        {
            var file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteFloat(17);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadFloatViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteDoubleToMemoryFile()
        {
            var file = new MemoryFile("wb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);
            file.WriteDouble(17);
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadDoubleViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteAndReadByteTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadShortTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadIntTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadLongTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadFloatTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadDoubleTensorViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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

        [Fact]
        public void WriteAndReadStorageBytesViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadStorageShortsViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadStorageIntsViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadStorageLongsViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadStorageFloatsViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadStorageDoublesViaMemoryFile()
        {
            const int size = 10;

            var file = new MemoryFile("rwb");
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
        public void WriteAndReadBytesViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new byte[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteBytes(data0);
            Assert.Equal(data0.Length*sizeof(byte), file.Position);
            file.Seek(0);

            var data1 = new byte[data0.Length];
            var rd = file.ReadBytes(data1, data0.Length);

            Assert.Equal(rd, data0.Length);
            Assert.Equal(data0.Length * sizeof(byte), file.Position);

            for (var i = 0; i < rd; ++i)
            {
                Assert.Equal(data0[i], data1[i]);
            }
            file.Close();
            Assert.False(file.IsOpen);
        }

        [Fact]
        public void WriteAndReadShortsViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteAndReadIntsViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
            Assert.NotNull(file);
            Assert.True(file.CanWrite);

            var data0 = new int[4];
            for (var i = 0; i < data0.Length; ++i) data0[i] = (byte)i;
            file.WriteInts(data0);
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
        public void WriteAndReadFloatsViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteAndReadLongsViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
        public void WriteAndReadDoublesViaMemoryFile()
        {
            var file = new MemoryFile("rwb");
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
    }
}
