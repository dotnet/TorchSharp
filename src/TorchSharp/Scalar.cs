using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public sealed class Scalar : IDisposable
    {
        internal IntPtr Handle { get; private set; }

        internal Scalar(IntPtr handle)
        {
            Handle = handle;
        }

        public static implicit operator Scalar(byte value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(short value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(int value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(long value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(float value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(double value)
        {
            return value.ToScalar();
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        [DllImport("LibTorchSharp")]
        extern static void THSThorch_dispose_scalar(IntPtr handle);

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        internal void Dispose(bool disposing)
        {
            if (disposing)
            {
                THSThorch_dispose_scalar(Handle);
                Handle = IntPtr.Zero;
            }
        }
    }

    public static class ScalarExtensionMethods
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_btos(byte hanvaluedle);

        public static Scalar ToScalar(this byte value)
        {
            return new Scalar(THSTorch_btos(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_stos(short hanvaluedle);

        public static Scalar ToScalar(this short value)
        {
            return new Scalar(THSTorch_stos(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_itos(int hanvaluedle);

        public static Scalar ToScalar(this int value)
        {
            return new Scalar(THSTorch_itos(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_ltos(long hanvaluedle);

        public static Scalar ToScalar(this long value)
        {
            return new Scalar(THSTorch_ltos(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_ftos(float hanvaluedle);

        public static Scalar ToScalar(this float value)
        {
            return new Scalar(THSTorch_ftos(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_dtos(double hanvaluedle);

        public static Scalar ToScalar(this double value)
        {
            return new Scalar(THSTorch_dtos(value));
        }
    }
}
