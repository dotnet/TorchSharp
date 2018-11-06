using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Torch.IO;

namespace Test
{
    [TestClass]
    public class MemoryFiles
    {
        [TestMethod]
        public void CreateStorage()
        {
            var storage = new THMemoryFile.CharStorage(256);
            Assert.IsNotNull(storage);
        }

        [TestMethod]
        public void CreateWritableMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void CreateReadableMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("r");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanRead);
            Assert.IsFalse(file.CanWrite);
        }

        
        [TestMethod]
        public void CreateReadWritableMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanRead);
            Assert.IsTrue(file.CanWrite);
        }

        [TestMethod]
        public void SeekInMemoryFile()
        {
            var storage = new THMemoryFile.CharStorage(256);
            Assert.IsNotNull(storage);
            var file = new Torch.IO.THMemoryFile(storage, "w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);

            file.Seek(150);
            Assert.AreEqual(150, file.Position);

            file.Seek(15);
            Assert.AreEqual(15, file.Position);
        }

        [TestMethod]
        public void SeekToEndInMemoryFile()
        {
            var storage = new THMemoryFile.CharStorage(256);
            Assert.IsNotNull(storage);
            var file = new Torch.IO.THMemoryFile(storage, "w");
            Assert.IsNotNull(file);
            Assert.IsFalse(file.CanRead);
            Assert.IsTrue(file.CanWrite);

            file.SeekEnd();
            Assert.AreEqual(255, file.Position);
        }

        [TestMethod]
        public void WriteByteToMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(17);
        }

        [TestMethod]
        public void WriteAndReadByteViaMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(13);
            file.WriteByte(17);
            file.Seek(0);
            var rd = file.ReadByte();
            Assert.AreEqual(13,rd);
            rd = file.ReadByte();
            Assert.AreEqual(17, rd);
        }

        [TestMethod]
        public void WriteShortToMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
        }

        [TestMethod]
        public void WriteAndReadShortViaMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
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
        public void WriteIntToMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteInt(17);
        }

        [TestMethod]
        public void WriteAndReadIntViaMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteInt(13);
            file.WriteInt(17);
            file.Seek(0);
            var rd = file.ReadInt();
            Assert.AreEqual(13, rd);
            rd = file.ReadInt();
            Assert.AreEqual(17, rd);
        }

        [TestMethod]
        public void WriteLongToMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteLong(17);
        }

        [TestMethod]
        public void WriteAndReadLongViaMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteLong(13);
            file.WriteLong(17);
            file.Seek(0);
            var rd = file.ReadLong();
            Assert.AreEqual(13, rd);
            rd = file.ReadLong();
            Assert.AreEqual(17, rd);
        }

        [TestMethod]
        public void WriteFloatToMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteFloat(17);
        }

        [TestMethod]
        public void WriteAndReadFloatViaMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteFloat(13);
            file.WriteFloat(17);
            file.Seek(0);
            var rd = file.ReadFloat();
            Assert.AreEqual(13, rd);
            rd = file.ReadFloat();
            Assert.AreEqual(17, rd);
        }

        [TestMethod]
        public void WriteDoubleToMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteDouble(17);
        }

        [TestMethod]
        public void WriteAndReadDoubleViaMemoryFile()
        {
            var file = new Torch.IO.THMemoryFile("rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteDouble(13);
            file.WriteDouble(17);
            file.Seek(0);
            var rd = file.ReadDouble();
            Assert.AreEqual(13, rd);
            rd = file.ReadDouble();
            Assert.AreEqual(17, rd);
        }
    }
}
