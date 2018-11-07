using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;
using TorchSharp;

namespace Torch.IO {

	public abstract partial class THFile : IDisposable
	{
        [DllImport("caffe2")] extern static byte THFile_readByteScalar(HType self);
		/// <summary>
		///   Read one byte from the file.
		/// </summary>
		/// <returns>A byte read from the current file position.</returns>
        public byte ReadByte() { return THFile_readByteScalar(this.handle); }

        [DllImport("caffe2")] extern static void THFile_writeByteScalar(HType self, byte scalar);
		/// <summary>
		///   Write one byte to the file.
		/// </summary>
		/// <param name="value">A byte to write at the current file position.</param>
        public void WriteByte(byte value) { THFile_writeByteScalar(this.handle, value); }

        [DllImport("caffe2")] extern static long THFile_readByte(HType self, ByteTensor.ByteStorage.HType storage);
		/// <summary>
		///   Read bytes from the file into the given storage.
		/// </summary>
		/// <param name="storage">A storage object to read data into.</param>
		/// <returns>The number of bytes read.</returns>
        public long ReadByte(ByteTensor.ByteStorage storage) { return THFile_readByte(this.handle, storage.handle); }

        [DllImport("caffe2")] extern static long THFile_writeByte(HType self, ByteTensor.ByteStorage.HType storage);
		/// <summary>
		///   Write bytes to the file from the given storage.
		/// </summary>
		/// <param name="storage">A storage object fetch data from.</param>
		/// <returns>The number of bytes written.</returns>
        public long WriteByte(ByteTensor.ByteStorage storage) { return THFile_writeByte(this.handle, storage.handle); }


        [DllImport("caffe2")] extern static short THFile_readShortScalar(HType self);
		/// <summary>
		///   Read one short from the file.
		/// </summary>
		/// <returns>A short read from the current file position.</returns>
        public short ReadShort() { return THFile_readShortScalar(this.handle); }

        [DllImport("caffe2")] extern static void THFile_writeShortScalar(HType self, short scalar);
		/// <summary>
		///   Write one short to the file.
		/// </summary>
		/// <param name="value">A short to write at the current file position.</param>
        public void WriteShort(short value) { THFile_writeShortScalar(this.handle, value); }

        [DllImport("caffe2")] extern static long THFile_readShort(HType self, ShortTensor.ShortStorage.HType storage);
		/// <summary>
		///   Read shorts from the file into the given storage.
		/// </summary>
		/// <param name="storage">A storage object to read data into.</param>
		/// <returns>The number of bytes read.</returns>
        public long ReadShort(ShortTensor.ShortStorage storage) { return THFile_readShort(this.handle, storage.handle); }

        [DllImport("caffe2")] extern static long THFile_writeShort(HType self, ShortTensor.ShortStorage.HType storage);
		/// <summary>
		///   Write shorts to the file from the given storage.
		/// </summary>
		/// <param name="storage">A storage object fetch data from.</param>
		/// <returns>The number of bytes written.</returns>
        public long WriteShort(ShortTensor.ShortStorage storage) { return THFile_writeShort(this.handle, storage.handle); }


        [DllImport("caffe2")] extern static int THFile_readIntScalar(HType self);
		/// <summary>
		///   Read one int from the file.
		/// </summary>
		/// <returns>A int read from the current file position.</returns>
        public int ReadInt() { return THFile_readIntScalar(this.handle); }

        [DllImport("caffe2")] extern static void THFile_writeIntScalar(HType self, int scalar);
		/// <summary>
		///   Write one int to the file.
		/// </summary>
		/// <param name="value">A int to write at the current file position.</param>
        public void WriteInt(int value) { THFile_writeIntScalar(this.handle, value); }

        [DllImport("caffe2")] extern static long THFile_readInt(HType self, IntTensor.IntStorage.HType storage);
		/// <summary>
		///   Read ints from the file into the given storage.
		/// </summary>
		/// <param name="storage">A storage object to read data into.</param>
		/// <returns>The number of bytes read.</returns>
        public long ReadInt(IntTensor.IntStorage storage) { return THFile_readInt(this.handle, storage.handle); }

        [DllImport("caffe2")] extern static long THFile_writeInt(HType self, IntTensor.IntStorage.HType storage);
		/// <summary>
		///   Write ints to the file from the given storage.
		/// </summary>
		/// <param name="storage">A storage object fetch data from.</param>
		/// <returns>The number of bytes written.</returns>
        public long WriteInt(IntTensor.IntStorage storage) { return THFile_writeInt(this.handle, storage.handle); }


        [DllImport("caffe2")] extern static long THFile_readLongScalar(HType self);
		/// <summary>
		///   Read one long from the file.
		/// </summary>
		/// <returns>A long read from the current file position.</returns>
        public long ReadLong() { return THFile_readLongScalar(this.handle); }

        [DllImport("caffe2")] extern static void THFile_writeLongScalar(HType self, long scalar);
		/// <summary>
		///   Write one long to the file.
		/// </summary>
		/// <param name="value">A long to write at the current file position.</param>
        public void WriteLong(long value) { THFile_writeLongScalar(this.handle, value); }

        [DllImport("caffe2")] extern static long THFile_readLong(HType self, LongTensor.LongStorage.HType storage);
		/// <summary>
		///   Read longs from the file into the given storage.
		/// </summary>
		/// <param name="storage">A storage object to read data into.</param>
		/// <returns>The number of bytes read.</returns>
        public long ReadLong(LongTensor.LongStorage storage) { return THFile_readLong(this.handle, storage.handle); }

        [DllImport("caffe2")] extern static long THFile_writeLong(HType self, LongTensor.LongStorage.HType storage);
		/// <summary>
		///   Write longs to the file from the given storage.
		/// </summary>
		/// <param name="storage">A storage object fetch data from.</param>
		/// <returns>The number of bytes written.</returns>
        public long WriteLong(LongTensor.LongStorage storage) { return THFile_writeLong(this.handle, storage.handle); }


        [DllImport("caffe2")] extern static double THFile_readDoubleScalar(HType self);
		/// <summary>
		///   Read one double from the file.
		/// </summary>
		/// <returns>A double read from the current file position.</returns>
        public double ReadDouble() { return THFile_readDoubleScalar(this.handle); }

        [DllImport("caffe2")] extern static void THFile_writeDoubleScalar(HType self, double scalar);
		/// <summary>
		///   Write one double to the file.
		/// </summary>
		/// <param name="value">A double to write at the current file position.</param>
        public void WriteDouble(double value) { THFile_writeDoubleScalar(this.handle, value); }

        [DllImport("caffe2")] extern static long THFile_readDouble(HType self, DoubleTensor.DoubleStorage.HType storage);
		/// <summary>
		///   Read doubles from the file into the given storage.
		/// </summary>
		/// <param name="storage">A storage object to read data into.</param>
		/// <returns>The number of bytes read.</returns>
        public long ReadDouble(DoubleTensor.DoubleStorage storage) { return THFile_readDouble(this.handle, storage.handle); }

        [DllImport("caffe2")] extern static long THFile_writeDouble(HType self, DoubleTensor.DoubleStorage.HType storage);
		/// <summary>
		///   Write doubles to the file from the given storage.
		/// </summary>
		/// <param name="storage">A storage object fetch data from.</param>
		/// <returns>The number of bytes written.</returns>
        public long WriteDouble(DoubleTensor.DoubleStorage storage) { return THFile_writeDouble(this.handle, storage.handle); }


        [DllImport("caffe2")] extern static float THFile_readFloatScalar(HType self);
		/// <summary>
		///   Read one float from the file.
		/// </summary>
		/// <returns>A float read from the current file position.</returns>
        public float ReadFloat() { return THFile_readFloatScalar(this.handle); }

        [DllImport("caffe2")] extern static void THFile_writeFloatScalar(HType self, float scalar);
		/// <summary>
		///   Write one float to the file.
		/// </summary>
		/// <param name="value">A float to write at the current file position.</param>
        public void WriteFloat(float value) { THFile_writeFloatScalar(this.handle, value); }

        [DllImport("caffe2")] extern static long THFile_readFloat(HType self, FloatTensor.FloatStorage.HType storage);
		/// <summary>
		///   Read floats from the file into the given storage.
		/// </summary>
		/// <param name="storage">A storage object to read data into.</param>
		/// <returns>The number of bytes read.</returns>
        public long ReadFloat(FloatTensor.FloatStorage storage) { return THFile_readFloat(this.handle, storage.handle); }

        [DllImport("caffe2")] extern static long THFile_writeFloat(HType self, FloatTensor.FloatStorage.HType storage);
		/// <summary>
		///   Write floats to the file from the given storage.
		/// </summary>
		/// <param name="storage">A storage object fetch data from.</param>
		/// <returns>The number of bytes written.</returns>
        public long WriteFloat(FloatTensor.FloatStorage storage) { return THFile_writeFloat(this.handle, storage.handle); }


	}
}