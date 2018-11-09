using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;
using TorchSharp;

namespace Torch.IO {

    public abstract partial class File : IDisposable
    {
        [DllImport("caffe2")] 
        extern static byte THFile_readByteScalar(HType self);

        /// <summary>
        ///   Read one byte from the file.
        /// </summary>
        /// <returns>A byte read from the current file position.</returns>
        public byte ReadByte() { return THFile_readByteScalar(this.handle); }

        [DllImport("caffe2")] 
        extern static void THFile_writeByteScalar(HType self, byte scalar);

        /// <summary>
        ///   Write one byte to the file.
        /// </summary>
        /// <param name="value">A byte to write at the current file position.</param>
        public void WriteByte(byte value) { THFile_writeByteScalar(this.handle, value); }

        [DllImport("caffe2")] 
        extern static long THFile_readByte(HType self, ByteTensor.ByteStorage.HType storage);

        /// <summary>
        ///   Read bytes from the file into the given storage.
        /// </summary>
        /// <param name="storage">A storage object to read data into.</param>
        /// <returns>The number of bytes read.</returns>
        public long ReadBytes(ByteTensor.ByteStorage storage) { return THFile_readByte(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_writeByte(HType self, ByteTensor.ByteStorage.HType storage);

        /// <summary>
        ///   Write bytes to the file from the given storage.
        /// </summary>
        /// <param name="storage">A storage object fetch data from.</param>
        /// <returns>The number of bytes written.</returns>
        public long WriteBytes(ByteTensor.ByteStorage storage) { return THFile_writeByte(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_readByteRaw(HType self, IntPtr data, long n);
        
        /// <summary>
        ///   Read bytes from the file into the given byte array.
        /// </summary>
        /// <param name="data">An array to place the data in after reading it from the file.</param>
        /// <param name="n">The maximum number of bytes to read.</param>
        /// <returns>The number of bytes read.</returns>
        public long ReadBytes(byte[] data, int n)
        {
            if (n > data.Length)
                throw new ArgumentOutOfRangeException("n cannot be greater than data.Length");
            unsafe
            {
                fixed (byte *dest = data)
                {
                    var readItems = THFile_readByteRaw(this.handle, (IntPtr)dest, n);
                    return readItems;
                }
            }
        }

		/// <summary>
		///   Read bytes from the file into the given byte tensor.
		/// </summary>
		/// <param name="tensor">A tensor to place the data in after reading it from the file.</param>
		/// <returns>The number of bytes read.</returns>
		public long ReadTensor(TorchSharp.ByteTensor tensor)
		{
			return THFile_readByteRaw(this.handle, tensor.Data, tensor.NumElements);			
		}

		[DllImport("caffe2")] 
		extern static long THFile_writeByteRaw(HType self, IntPtr data, long n);

        /// <summary>
        ///   Write bytes to the file from the given byte array.
        /// </summary>
        /// <param name="data">An array containing data to be written to the file.</param>
        /// <param name="n">The number of bytes to write.</param>
        /// <returns>The number of bytes written.</returns>
        public long WriteBytes(byte[] data, int n = -1)
        {
            n = (n == -1) ? data.Length : Math.Min(n, data.Length);
            unsafe
            {
                fixed (byte *dest = data)
                {
                    var writtenItems = THFile_writeByteRaw(this.handle, (IntPtr)dest, n);
                    return writtenItems;
                }
            }
        }
		/// <summary>
		///   Write bytes to the file from the given byte tensor.
		/// </summary>
		/// <param name="tensor">A tensor containing data to be written to the file.</param>
		/// <returns>The number of bytes written.</returns>
		public long WriteTensor(TorchSharp.ByteTensor tensor)
		{
			return THFile_writeByteRaw(this.handle, tensor.Data, tensor.NumElements);			
		}
        [DllImport("caffe2")] 
        extern static short THFile_readShortScalar(HType self);

        /// <summary>
        ///   Read one short from the file.
        /// </summary>
        /// <returns>A short read from the current file position.</returns>
        public short ReadShort() { return THFile_readShortScalar(this.handle); }

        [DllImport("caffe2")] 
        extern static void THFile_writeShortScalar(HType self, short scalar);

        /// <summary>
        ///   Write one short to the file.
        /// </summary>
        /// <param name="value">A short to write at the current file position.</param>
        public void WriteShort(short value) { THFile_writeShortScalar(this.handle, value); }

        [DllImport("caffe2")] 
        extern static long THFile_readShort(HType self, ShortTensor.ShortStorage.HType storage);

        /// <summary>
        ///   Read shorts from the file into the given storage.
        /// </summary>
        /// <param name="storage">A storage object to read data into.</param>
        /// <returns>The number of shorts read.</returns>
        public long ReadShorts(ShortTensor.ShortStorage storage) { return THFile_readShort(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_writeShort(HType self, ShortTensor.ShortStorage.HType storage);

        /// <summary>
        ///   Write shorts to the file from the given storage.
        /// </summary>
        /// <param name="storage">A storage object fetch data from.</param>
        /// <returns>The number of shorts written.</returns>
        public long WriteShorts(ShortTensor.ShortStorage storage) { return THFile_writeShort(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_readShortRaw(HType self, IntPtr data, long n);
        
        /// <summary>
        ///   Read shorts from the file into the given short array.
        /// </summary>
        /// <param name="data">An array to place the data in after reading it from the file.</param>
        /// <param name="n">The maximum number of shorts to read.</param>
        /// <returns>The number of shorts read.</returns>
        public long ReadShorts(short[] data, int n)
        {
            if (n > data.Length)
                throw new ArgumentOutOfRangeException("n cannot be greater than data.Length");
            unsafe
            {
                fixed (short *dest = data)
                {
                    var readItems = THFile_readShortRaw(this.handle, (IntPtr)dest, n);
                    return readItems;
                }
            }
        }

		/// <summary>
		///   Read shorts from the file into the given short tensor.
		/// </summary>
		/// <param name="tensor">A tensor to place the data in after reading it from the file.</param>
		/// <returns>The number of shorts read.</returns>
		public long ReadTensor(TorchSharp.ShortTensor tensor)
		{
			return THFile_readShortRaw(this.handle, tensor.Data, tensor.NumElements);			
		}

		[DllImport("caffe2")] 
		extern static long THFile_writeShortRaw(HType self, IntPtr data, long n);

        /// <summary>
        ///   Write shorts to the file from the given short array.
        /// </summary>
        /// <param name="data">An array containing data to be written to the file.</param>
        /// <param name="n">The number of shorts to write.</param>
        /// <returns>The number of shorts written.</returns>
        public long WriteShorts(short[] data, int n = -1)
        {
            n = (n == -1) ? data.Length : Math.Min(n, data.Length);
            unsafe
            {
                fixed (short *dest = data)
                {
                    var writtenItems = THFile_writeShortRaw(this.handle, (IntPtr)dest, n);
                    return writtenItems;
                }
            }
        }
		/// <summary>
		///   Write shorts to the file from the given short tensor.
		/// </summary>
		/// <param name="tensor">A tensor containing data to be written to the file.</param>
		/// <returns>The number of shorts written.</returns>
		public long WriteTensor(TorchSharp.ShortTensor tensor)
		{
			return THFile_writeShortRaw(this.handle, tensor.Data, tensor.NumElements);			
		}
        [DllImport("caffe2")] 
        extern static int THFile_readIntScalar(HType self);

        /// <summary>
        ///   Read one int from the file.
        /// </summary>
        /// <returns>A int read from the current file position.</returns>
        public int ReadInt() { return THFile_readIntScalar(this.handle); }

        [DllImport("caffe2")] 
        extern static void THFile_writeIntScalar(HType self, int scalar);

        /// <summary>
        ///   Write one int to the file.
        /// </summary>
        /// <param name="value">A int to write at the current file position.</param>
        public void WriteInt(int value) { THFile_writeIntScalar(this.handle, value); }

        [DllImport("caffe2")] 
        extern static long THFile_readInt(HType self, IntTensor.IntStorage.HType storage);

        /// <summary>
        ///   Read ints from the file into the given storage.
        /// </summary>
        /// <param name="storage">A storage object to read data into.</param>
        /// <returns>The number of ints read.</returns>
        public long ReadInts(IntTensor.IntStorage storage) { return THFile_readInt(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_writeInt(HType self, IntTensor.IntStorage.HType storage);

        /// <summary>
        ///   Write ints to the file from the given storage.
        /// </summary>
        /// <param name="storage">A storage object fetch data from.</param>
        /// <returns>The number of ints written.</returns>
        public long WriteInts(IntTensor.IntStorage storage) { return THFile_writeInt(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_readIntRaw(HType self, IntPtr data, long n);
        
        /// <summary>
        ///   Read ints from the file into the given int array.
        /// </summary>
        /// <param name="data">An array to place the data in after reading it from the file.</param>
        /// <param name="n">The maximum number of ints to read.</param>
        /// <returns>The number of ints read.</returns>
        public long ReadInts(int[] data, int n)
        {
            if (n > data.Length)
                throw new ArgumentOutOfRangeException("n cannot be greater than data.Length");
            unsafe
            {
                fixed (int *dest = data)
                {
                    var readItems = THFile_readIntRaw(this.handle, (IntPtr)dest, n);
                    return readItems;
                }
            }
        }

		/// <summary>
		///   Read ints from the file into the given int tensor.
		/// </summary>
		/// <param name="tensor">A tensor to place the data in after reading it from the file.</param>
		/// <returns>The number of ints read.</returns>
		public long ReadTensor(TorchSharp.IntTensor tensor)
		{
			return THFile_readIntRaw(this.handle, tensor.Data, tensor.NumElements);			
		}

		[DllImport("caffe2")] 
		extern static long THFile_writeIntRaw(HType self, IntPtr data, long n);

        /// <summary>
        ///   Write ints to the file from the given int array.
        /// </summary>
        /// <param name="data">An array containing data to be written to the file.</param>
        /// <param name="n">The number of ints to write.</param>
        /// <returns>The number of ints written.</returns>
        public long WriteInts(int[] data, int n = -1)
        {
            n = (n == -1) ? data.Length : Math.Min(n, data.Length);
            unsafe
            {
                fixed (int *dest = data)
                {
                    var writtenItems = THFile_writeIntRaw(this.handle, (IntPtr)dest, n);
                    return writtenItems;
                }
            }
        }
		/// <summary>
		///   Write ints to the file from the given int tensor.
		/// </summary>
		/// <param name="tensor">A tensor containing data to be written to the file.</param>
		/// <returns>The number of ints written.</returns>
		public long WriteTensor(TorchSharp.IntTensor tensor)
		{
			return THFile_writeIntRaw(this.handle, tensor.Data, tensor.NumElements);			
		}
        [DllImport("caffe2")] 
        extern static long THFile_readLongScalar(HType self);

        /// <summary>
        ///   Read one long from the file.
        /// </summary>
        /// <returns>A long read from the current file position.</returns>
        public long ReadLong() { return THFile_readLongScalar(this.handle); }

        [DllImport("caffe2")] 
        extern static void THFile_writeLongScalar(HType self, long scalar);

        /// <summary>
        ///   Write one long to the file.
        /// </summary>
        /// <param name="value">A long to write at the current file position.</param>
        public void WriteLong(long value) { THFile_writeLongScalar(this.handle, value); }

        [DllImport("caffe2")] 
        extern static long THFile_readLong(HType self, LongTensor.LongStorage.HType storage);

        /// <summary>
        ///   Read longs from the file into the given storage.
        /// </summary>
        /// <param name="storage">A storage object to read data into.</param>
        /// <returns>The number of longs read.</returns>
        public long ReadLongs(LongTensor.LongStorage storage) { return THFile_readLong(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_writeLong(HType self, LongTensor.LongStorage.HType storage);

        /// <summary>
        ///   Write longs to the file from the given storage.
        /// </summary>
        /// <param name="storage">A storage object fetch data from.</param>
        /// <returns>The number of longs written.</returns>
        public long WriteLongs(LongTensor.LongStorage storage) { return THFile_writeLong(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_readLongRaw(HType self, IntPtr data, long n);
        
        /// <summary>
        ///   Read longs from the file into the given long array.
        /// </summary>
        /// <param name="data">An array to place the data in after reading it from the file.</param>
        /// <param name="n">The maximum number of longs to read.</param>
        /// <returns>The number of longs read.</returns>
        public long ReadLongs(long[] data, int n)
        {
            if (n > data.Length)
                throw new ArgumentOutOfRangeException("n cannot be greater than data.Length");
            unsafe
            {
                fixed (long *dest = data)
                {
                    var readItems = THFile_readLongRaw(this.handle, (IntPtr)dest, n);
                    return readItems;
                }
            }
        }

		/// <summary>
		///   Read longs from the file into the given long tensor.
		/// </summary>
		/// <param name="tensor">A tensor to place the data in after reading it from the file.</param>
		/// <returns>The number of longs read.</returns>
		public long ReadTensor(TorchSharp.LongTensor tensor)
		{
			return THFile_readLongRaw(this.handle, tensor.Data, tensor.NumElements);			
		}

		[DllImport("caffe2")] 
		extern static long THFile_writeLongRaw(HType self, IntPtr data, long n);

        /// <summary>
        ///   Write longs to the file from the given long array.
        /// </summary>
        /// <param name="data">An array containing data to be written to the file.</param>
        /// <param name="n">The number of longs to write.</param>
        /// <returns>The number of longs written.</returns>
        public long WriteLongs(long[] data, int n = -1)
        {
            n = (n == -1) ? data.Length : Math.Min(n, data.Length);
            unsafe
            {
                fixed (long *dest = data)
                {
                    var writtenItems = THFile_writeLongRaw(this.handle, (IntPtr)dest, n);
                    return writtenItems;
                }
            }
        }
		/// <summary>
		///   Write longs to the file from the given long tensor.
		/// </summary>
		/// <param name="tensor">A tensor containing data to be written to the file.</param>
		/// <returns>The number of longs written.</returns>
		public long WriteTensor(TorchSharp.LongTensor tensor)
		{
			return THFile_writeLongRaw(this.handle, tensor.Data, tensor.NumElements);			
		}
        [DllImport("caffe2")] 
        extern static float THFile_readFloatScalar(HType self);

        /// <summary>
        ///   Read one float from the file.
        /// </summary>
        /// <returns>A float read from the current file position.</returns>
        public float ReadFloat() { return THFile_readFloatScalar(this.handle); }

        [DllImport("caffe2")] 
        extern static void THFile_writeFloatScalar(HType self, float scalar);

        /// <summary>
        ///   Write one float to the file.
        /// </summary>
        /// <param name="value">A float to write at the current file position.</param>
        public void WriteFloat(float value) { THFile_writeFloatScalar(this.handle, value); }

        [DllImport("caffe2")] 
        extern static long THFile_readFloat(HType self, FloatTensor.FloatStorage.HType storage);

        /// <summary>
        ///   Read floats from the file into the given storage.
        /// </summary>
        /// <param name="storage">A storage object to read data into.</param>
        /// <returns>The number of floats read.</returns>
        public long ReadFloats(FloatTensor.FloatStorage storage) { return THFile_readFloat(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_writeFloat(HType self, FloatTensor.FloatStorage.HType storage);

        /// <summary>
        ///   Write floats to the file from the given storage.
        /// </summary>
        /// <param name="storage">A storage object fetch data from.</param>
        /// <returns>The number of floats written.</returns>
        public long WriteFloats(FloatTensor.FloatStorage storage) { return THFile_writeFloat(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_readFloatRaw(HType self, IntPtr data, long n);
        
        /// <summary>
        ///   Read floats from the file into the given float array.
        /// </summary>
        /// <param name="data">An array to place the data in after reading it from the file.</param>
        /// <param name="n">The maximum number of floats to read.</param>
        /// <returns>The number of floats read.</returns>
        public long ReadFloats(float[] data, int n)
        {
            if (n > data.Length)
                throw new ArgumentOutOfRangeException("n cannot be greater than data.Length");
            unsafe
            {
                fixed (float *dest = data)
                {
                    var readItems = THFile_readFloatRaw(this.handle, (IntPtr)dest, n);
                    return readItems;
                }
            }
        }

		/// <summary>
		///   Read floats from the file into the given float tensor.
		/// </summary>
		/// <param name="tensor">A tensor to place the data in after reading it from the file.</param>
		/// <returns>The number of floats read.</returns>
		public long ReadTensor(TorchSharp.FloatTensor tensor)
		{
			return THFile_readFloatRaw(this.handle, tensor.Data, tensor.NumElements);			
		}

		[DllImport("caffe2")] 
		extern static long THFile_writeFloatRaw(HType self, IntPtr data, long n);

        /// <summary>
        ///   Write floats to the file from the given float array.
        /// </summary>
        /// <param name="data">An array containing data to be written to the file.</param>
        /// <param name="n">The number of floats to write.</param>
        /// <returns>The number of floats written.</returns>
        public long WriteFloats(float[] data, int n = -1)
        {
            n = (n == -1) ? data.Length : Math.Min(n, data.Length);
            unsafe
            {
                fixed (float *dest = data)
                {
                    var writtenItems = THFile_writeFloatRaw(this.handle, (IntPtr)dest, n);
                    return writtenItems;
                }
            }
        }
		/// <summary>
		///   Write floats to the file from the given float tensor.
		/// </summary>
		/// <param name="tensor">A tensor containing data to be written to the file.</param>
		/// <returns>The number of floats written.</returns>
		public long WriteTensor(TorchSharp.FloatTensor tensor)
		{
			return THFile_writeFloatRaw(this.handle, tensor.Data, tensor.NumElements);			
		}
        [DllImport("caffe2")] 
        extern static double THFile_readDoubleScalar(HType self);

        /// <summary>
        ///   Read one double from the file.
        /// </summary>
        /// <returns>A double read from the current file position.</returns>
        public double ReadDouble() { return THFile_readDoubleScalar(this.handle); }

        [DllImport("caffe2")] 
        extern static void THFile_writeDoubleScalar(HType self, double scalar);

        /// <summary>
        ///   Write one double to the file.
        /// </summary>
        /// <param name="value">A double to write at the current file position.</param>
        public void WriteDouble(double value) { THFile_writeDoubleScalar(this.handle, value); }

        [DllImport("caffe2")] 
        extern static long THFile_readDouble(HType self, DoubleTensor.DoubleStorage.HType storage);

        /// <summary>
        ///   Read doubles from the file into the given storage.
        /// </summary>
        /// <param name="storage">A storage object to read data into.</param>
        /// <returns>The number of doubles read.</returns>
        public long ReadDoubles(DoubleTensor.DoubleStorage storage) { return THFile_readDouble(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_writeDouble(HType self, DoubleTensor.DoubleStorage.HType storage);

        /// <summary>
        ///   Write doubles to the file from the given storage.
        /// </summary>
        /// <param name="storage">A storage object fetch data from.</param>
        /// <returns>The number of doubles written.</returns>
        public long WriteDoubles(DoubleTensor.DoubleStorage storage) { return THFile_writeDouble(this.handle, storage.handle); }

        [DllImport("caffe2")] 
        extern static long THFile_readDoubleRaw(HType self, IntPtr data, long n);
        
        /// <summary>
        ///   Read doubles from the file into the given double array.
        /// </summary>
        /// <param name="data">An array to place the data in after reading it from the file.</param>
        /// <param name="n">The maximum number of doubles to read.</param>
        /// <returns>The number of doubles read.</returns>
        public long ReadDoubles(double[] data, int n)
        {
            if (n > data.Length)
                throw new ArgumentOutOfRangeException("n cannot be greater than data.Length");
            unsafe
            {
                fixed (double *dest = data)
                {
                    var readItems = THFile_readDoubleRaw(this.handle, (IntPtr)dest, n);
                    return readItems;
                }
            }
        }

		/// <summary>
		///   Read doubles from the file into the given double tensor.
		/// </summary>
		/// <param name="tensor">A tensor to place the data in after reading it from the file.</param>
		/// <returns>The number of doubles read.</returns>
		public long ReadTensor(TorchSharp.DoubleTensor tensor)
		{
			return THFile_readDoubleRaw(this.handle, tensor.Data, tensor.NumElements);			
		}

		[DllImport("caffe2")] 
		extern static long THFile_writeDoubleRaw(HType self, IntPtr data, long n);

        /// <summary>
        ///   Write doubles to the file from the given double array.
        /// </summary>
        /// <param name="data">An array containing data to be written to the file.</param>
        /// <param name="n">The number of doubles to write.</param>
        /// <returns>The number of doubles written.</returns>
        public long WriteDoubles(double[] data, int n = -1)
        {
            n = (n == -1) ? data.Length : Math.Min(n, data.Length);
            unsafe
            {
                fixed (double *dest = data)
                {
                    var writtenItems = THFile_writeDoubleRaw(this.handle, (IntPtr)dest, n);
                    return writtenItems;
                }
            }
        }
		/// <summary>
		///   Write doubles to the file from the given double tensor.
		/// </summary>
		/// <param name="tensor">A tensor containing data to be written to the file.</param>
		/// <returns>The number of doubles written.</returns>
		public long WriteTensor(TorchSharp.DoubleTensor tensor)
		{
			return THFile_writeDoubleRaw(this.handle, tensor.Data, tensor.NumElements);			
		}
    }
}