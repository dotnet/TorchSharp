// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
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

        public static implicit operator Scalar(sbyte value)
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

        public static implicit operator Scalar(Half value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(System.Numerics.Complex value)
        {
            return value.ToScalar();
        }

        public static implicit operator Scalar(bool value)
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
        extern static IntPtr THSTorch_uint8_to_scalar(byte value);

        public static Scalar ToScalar(this byte value)
        {
            return new Scalar(THSTorch_uint8_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int8_to_scalar(sbyte value);

        public static Scalar ToScalar(this sbyte value)
        {
            return new Scalar(THSTorch_int8_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_short_to_scalar(short value);

        public static Scalar ToScalar(this short value)
        {
            return new Scalar(THSTorch_short_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int32_to_scalar(int value);

        public static Scalar ToScalar(this int value)
        {
            return new Scalar(THSTorch_int32_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_long_to_scalar(long value);

        public static Scalar ToScalar(this long value)
        {
            return new Scalar(THSTorch_long_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float32_to_scalar(float value);

        public static Scalar ToScalar(this float value)
        {
            return new Scalar(THSTorch_float32_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float64_to_scalar(double value);

        public static Scalar ToScalar(this double value)
        {
            return new Scalar(THSTorch_float64_to_scalar(value));
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_bool_to_scalar(bool value);

        public static Scalar ToScalar(this bool value)
        {
            return new Scalar(THSTorch_bool_to_scalar(value));
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_half_to_scalar(Half value);

        public static Scalar ToScalar(this Half value)
        {
            return new Scalar(THSTorch_half_to_scalar(value));
        }
        //[DllImport("LibTorchSharp")]
        //extern static IntPtr THSTorch_complex32_to_scalar(System.Numerics.Complex value);

        //public static Scalar ToScalar(this System.Numerics.Complex value)
        //{
        //    return new Scalar(THSTorch_complex32_to_scalar(value));
        //}

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_complex64_to_scalar(System.Numerics.Complex value);

        public static Scalar ToScalar(this System.Numerics.Complex value)
        {
            return new Scalar(THSTorch_complex64_to_scalar(value));
        }
    }
}
