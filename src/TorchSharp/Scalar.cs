// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    /// <summary>
    /// Represents a dynamically typed scalar value to the LibTorch runtime.
    /// </summary>
    public sealed class Scalar : IDisposable
    {
        internal IntPtr Handle {
            get {
                if (handle == IntPtr.Zero)
                    throw new InvalidOperationException("Scalar invalid -- empty handle.");
                return handle;
            }
            private set { handle = value; }
        }
        private IntPtr handle;

        internal Scalar(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(byte value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(sbyte value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(short value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(int value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(long value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(float value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(double value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(bool value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar((float,float) value)
        {
            return value.ToScalar();
        }

        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(System.Numerics.Complex value)
        {
            return value.ToScalar();
        }

        [DllImport("LibTorchSharp")]
        extern static byte THSTorch_scalar_type(IntPtr value);

        /// <summary>
        /// Gets the actual type of the Scalar value
        /// </summary>
        public torch.ScalarType Type {
            get {
                return (torch.ScalarType)THSTorch_scalar_type(Handle);
            }
        }

        /// <summary>
        ///   Finalize the tensor. Releases the tensor and its associated data.
        /// </summary>
        ~Scalar() => Dispose(false);
 
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

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this byte value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_uint8_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int8_to_scalar(sbyte value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this sbyte value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int8_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int16_to_scalar(short value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this short value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int16_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int32_to_scalar(int value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this int value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int32_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_int64_to_scalar(long value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this long value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int64_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float32_to_scalar(float value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this float value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_float32_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float64_to_scalar(double value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this double value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_float64_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_complex32_to_scalar(float real, float imaginary);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this (float, float) value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_complex32_to_scalar(value.Item1, value.Item2));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_complex64_to_scalar(double real, double imaginary);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this System.Numerics.Complex value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_complex64_to_scalar(value.Real, value.Imaginary));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_bool_to_scalar(bool value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this bool value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_bool_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_float16_to_scalar(float value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToFloat16Scalar(this float value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_float16_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_bfloat16_to_scalar(float value);

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToBFloat16Scalar(this float value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_bfloat16_to_scalar(value));
        }

        [DllImport("LibTorchSharp")]
        extern static float THSTorch_scalar_to_float32(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static float ToSingle(this Scalar value)
        {
            return THSTorch_scalar_to_float32(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static double THSTorch_scalar_to_float64(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static double ToDouble(this Scalar value)
        {
            return THSTorch_scalar_to_float64(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static sbyte THSTorch_scalar_to_int8(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static sbyte ToSByte(this Scalar value)
        {
            return THSTorch_scalar_to_int8(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static byte THSTorch_scalar_to_uint8(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static byte ToByte(this Scalar value)
        {
            return THSTorch_scalar_to_uint8(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static short THSTorch_scalar_to_int16(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static short ToInt16(this Scalar value)
        {
            return THSTorch_scalar_to_int16(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static int THSTorch_scalar_to_int32(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static int ToInt32(this Scalar value)
        {
            return THSTorch_scalar_to_int32(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static long THSTorch_scalar_to_int64(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static long ToInt64(this Scalar value)
        {
            return THSTorch_scalar_to_int64(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static bool THSTorch_scalar_to_bool(IntPtr handle);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static bool ToBoolean(this Scalar value)
        {
            return THSTorch_scalar_to_bool(value.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void THSTorch_scalar_to_complex32(IntPtr handle, AllocatePinnedArray allocator);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static (float Real, float Imaginary) ToComplexFloat32(this Scalar value)
        {
            float[] floatArray;

            using (var pa = new PinnedArray<float>()) {
                THSTorch_scalar_to_complex32(value.Handle, pa.CreateArray);
                torch.CheckForErrors();
                floatArray = pa.Array;
            }

            return (floatArray[0], floatArray[1]);
        }

        [DllImport("LibTorchSharp")]
        extern static void THSTorch_scalar_to_complex64(IntPtr handle, AllocatePinnedArray allocator);

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static System.Numerics.Complex ToComplexFloat64(this Scalar value)
        {
            double[] floatArray;

            using (var pa = new PinnedArray<double>()) {
                THSTorch_scalar_to_complex64(value.Handle, pa.CreateArray);
                torch.CheckForErrors();
                floatArray = pa.Array;
            }

            return new System.Numerics.Complex(floatArray[0], floatArray[1]);
        }
    }
}
