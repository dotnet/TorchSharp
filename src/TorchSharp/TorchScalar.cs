// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public sealed class TorchScalar : IDisposable
    {
        internal IntPtr Handle { get; private set; }

        internal TorchScalar(IntPtr handle)
        {
            Handle = handle;
        }

        public static implicit operator TorchScalar(byte value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(sbyte value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(short value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(int value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(long value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(float value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(double value)
        {
            return value.ToScalar();
        }

        public static implicit operator TorchScalar(bool value)
        {
            return value.ToScalar();
        }

        /// <summary>
        ///   Finalize the tensor. Releases the tensor and its associated data.
        /// </summary>
        ~TorchScalar() => Dispose(false);
 
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        [DllImport("LibTorchSharp")]
        extern static void THSTorch_dispose_scalar(IntPtr handle);

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero) {
                THSTorch_dispose_scalar(Handle);
                Handle = IntPtr.Zero;
            }
        }
    }

    public static class ScalarExtensionMethods
    {

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_uint8_to_scalar(byte value);

        public static TorchScalar ToScalar(this byte value)
        {
            return new TorchScalar(THSTorch_uint8_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int8_to_scalar(sbyte value);

        public static TorchScalar ToScalar(this sbyte value)
        {
            return new TorchScalar(THSTorch_int8_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int16_to_scalar(short value);

        public static TorchScalar ToScalar(this short value)
        {
            return new TorchScalar(THSTorch_int16_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int32_to_scalar(int value);

        public static TorchScalar ToScalar(this int value)
        {
            return new TorchScalar(THSTorch_int32_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int64_to_scalar(long value);

        public static TorchScalar ToScalar(this long value)
        {
            return new TorchScalar(THSTorch_int64_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float32_to_scalar(float value);

        public static TorchScalar ToScalar(this float value)
        {
            return new TorchScalar(THSTorch_float32_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float64_to_scalar(double value);

        public static TorchScalar ToScalar(this double value)
        {
            return new TorchScalar(THSTorch_float64_to_scalar(value));
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_bool_to_scalar(bool value);

        public static TorchScalar ToScalar(this bool value)
        {
            return new TorchScalar(THSTorch_bool_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float16_to_scalar(float value);

        public static TorchScalar ToFloat16Scalar(this float value)
        {
            return new TorchScalar(THSTorch_float16_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_bfloat16_to_scalar(float value);

        public static TorchScalar ToBFloat16Scalar(this float value)
        {
            return new TorchScalar(THSTorch_bfloat16_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static float THSTorch_scalar_to_float32(IntPtr handle);

        public static float ToSingle(this TorchScalar value)
        {
            return THSTorch_scalar_to_float32(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static double THSTorch_scalar_to_float64(IntPtr handle);

        public static double ToDouble(this TorchScalar value)
        {
            return THSTorch_scalar_to_float64(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static sbyte THSTorch_scalar_to_int8(IntPtr handle);

        public static sbyte ToSByte(this TorchScalar value)
        {
            return THSTorch_scalar_to_int8(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static byte THSTorch_scalar_to_uint8(IntPtr handle);

        public static byte ToByte(this TorchScalar value)
        {
            return THSTorch_scalar_to_uint8(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static short THSTorch_scalar_to_int16(IntPtr handle);

        public static short ToInt16(this TorchScalar value)
        {
            return THSTorch_scalar_to_int16(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static int THSTorch_scalar_to_int32(IntPtr handle);

        public static int ToInt32(this TorchScalar value)
        {
            return THSTorch_scalar_to_int32(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static long THSTorch_scalar_to_int64(IntPtr handle);

        public static long ToInt64(this TorchScalar value)
        {
            return THSTorch_scalar_to_int64(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static bool THSTorch_scalar_to_bool(IntPtr handle);

        public static bool ToBoolean(this TorchScalar value)
        {
            return THSTorch_scalar_to_bool(value.Handle);
        }

    }
}
