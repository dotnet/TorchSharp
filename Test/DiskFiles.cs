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

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void CreateReadWritableDiskFile()
        {
            var file = new Torch.IO.DiskFile("test2.dat", "rw");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanRead);
            Assert.IsTrue(file.CanWrite);
            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteByteToDiskFile()
        {
            var file = new Torch.IO.DiskFile("test3.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteByte(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadByteViaMemoryFile()
        {
            var file = new Torch.IO.DiskFile("test4.dat", "rw");
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
        public void WriteShortToMemoryFile()
        {
            var file = new Torch.IO.DiskFile("test5.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteShort(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadShortViaMemoryFile()
        {
            var file = new Torch.IO.DiskFile("test6.dat", "rw");
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
            var file = new Torch.IO.DiskFile("test7.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteInt(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadIntViaMemoryFile()
        {
            var file = new Torch.IO.DiskFile("test8.dat", "rw");
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
            var file = new Torch.IO.DiskFile("test9.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteLong(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadLongViaMemoryFile()
        {
            var file = new Torch.IO.DiskFile("testA.dat", "rw");
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
            var file = new Torch.IO.DiskFile("testB.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteFloat(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadFloatViaMemoryFile()
        {
            var file = new Torch.IO.DiskFile("testC.dat", "rw");
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
            var file = new Torch.IO.DiskFile("testD.dat", "w");
            Assert.IsNotNull(file);
            Assert.IsTrue(file.CanWrite);
            file.WriteDouble(17);

            file.Close();
            Assert.IsFalse(file.IsOpen);
        }

        [TestMethod]
        public void WriteAndReadDoubleViaMemoryFile()
        {
            var file = new Torch.IO.DiskFile("testE.dat", "rw");
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
    }
}
