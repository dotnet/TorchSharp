// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
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

#if NET6_0_OR_GREATER
        /// <summary>
        /// Implicitly convert a .NET scalar value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(Half value)
        {
            return value.ToScalar();
        }
#endif

        /// <summary>
        /// Implicitly convert a BFloat16 value to Scalar
        /// </summary>
        /// <param name="value">The scalar value.</param>
        public static implicit operator Scalar(BFloat16 value)
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
        public static implicit operator Scalar((float, float) value)
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

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        void Dispose(bool disposing)
        {
            if (handle != IntPtr.Zero) {
                THSTorch_dispose_scalar(handle);
                handle = IntPtr.Zero;
            }
        }
    }

    public static class ScalarExtensionMethods
    {
        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this byte value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_uint8_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this sbyte value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int8_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this short value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int16_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this int value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int32_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this long value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_int64_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this float value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_float32_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this double value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_float64_to_scalar(value));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this (float, float) value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_complex32_to_scalar(value.Item1, value.Item2));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this System.Numerics.Complex value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_complex64_to_scalar(value.Real, value.Imaginary));
        }

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this bool value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_bool_to_scalar(value));
        }

#if NET6_0_OR_GREATER
        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this Half value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_float16_to_scalar((float)value));
        }
#endif

        /// <summary>
        /// Explcitly construct a Scalar from a .NET scalar.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToBFloat16Scalar(this float value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_bfloat16_to_scalar(value));
        }

        /// <summary>
        /// Explicitly construct a Scalar from a BFloat16 value.
        /// </summary>
        /// <param name="value">The input scalar value</param>
        public static Scalar ToScalar(this BFloat16 value)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new Scalar(THSTorch_bfloat16_to_scalar(value.ToSingle()));
        }

#if NET6_0_OR_GREATER
        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static Half ToHalf(this Scalar value)
        {
            Half res;
            THSTorch_scalar_to_float16(value.Handle, out res);
            return res;
        }
#endif

        /// <summary>
        /// Explicitly convert a Scalar value to a BFloat16.
        /// </summary>
        /// <param name="value">The input value.</param>
        public static BFloat16 ToBFloat16(this Scalar value)
        {
            THSTorch_scalar_to_bfloat16(value.Handle, out ushort res);
            return BFloat16.FromRawValue(res);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static float ToSingle(this Scalar value)
        {
            return THSTorch_scalar_to_float32(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static double ToDouble(this Scalar value)
        {
            return THSTorch_scalar_to_float64(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static sbyte ToSByte(this Scalar value)
        {
            return THSTorch_scalar_to_int8(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static byte ToByte(this Scalar value)
        {
            return THSTorch_scalar_to_uint8(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static short ToInt16(this Scalar value)
        {
            return THSTorch_scalar_to_int16(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static int ToInt32(this Scalar value)
        {
            return THSTorch_scalar_to_int32(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static long ToInt64(this Scalar value)
        {
            return THSTorch_scalar_to_int64(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static bool ToBoolean(this Scalar value)
        {
            return THSTorch_scalar_to_bool(value.Handle);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static (float Real, float Imaginary) ToComplexFloat32(this Scalar value)
        {
            THSTorch_scalar_to_complex32(value.Handle, out float real, out float imaginary);
            torch.CheckForErrors();

            return (real, imaginary);
        }

        /// <summary>
        /// Explicitly convert a Scalar value to a .NET scalar
        /// </summary>
        /// <param name="value">The input value.</param>
        public static System.Numerics.Complex ToComplexFloat64(this Scalar value)
        {
            THSTorch_scalar_to_complex64(value.Handle, out double real, out double imaginary);
            torch.CheckForErrors();

            return new System.Numerics.Complex(real, imaginary);
        }
    }
}
