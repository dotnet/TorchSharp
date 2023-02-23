// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTorch_scalar_to_bool(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorch_dispose_scalar(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern byte THSTorch_scalar_type(IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorch_can_cast(int type1, int type2);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorch_promote_types(int type1, int type2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_uint8_to_scalar(byte value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_int8_to_scalar(sbyte value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_int16_to_scalar(short value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_int32_to_scalar(int value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_int64_to_scalar(long value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_float32_to_scalar(float value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_float64_to_scalar(double value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_complex32_to_scalar(float real, float imaginary);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_complex64_to_scalar(double real, double imaginary);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_bool_to_scalar([MarshalAs(UnmanagedType.U1)] bool value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_float16_to_scalar(float value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_bfloat16_to_scalar(float value);

        [DllImport("LibTorchSharp")]
        internal static extern float THSTorch_scalar_to_float32(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern double THSTorch_scalar_to_float64(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSTorch_scalar_to_int8(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern byte THSTorch_scalar_to_uint8(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern short THSTorch_scalar_to_int16(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorch_scalar_to_int32(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTorch_scalar_to_int64(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorch_scalar_to_complex32(IntPtr handle, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorch_scalar_to_complex64(IntPtr handle, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_get_and_reset_last_err();

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTorch_lstsq(IntPtr handle, IntPtr b, out IntPtr qr);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorch_get_num_threads();

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorch_set_num_threads(int threads);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorch_get_num_interop_threads();

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorch_set_num_interop_threads(int threads);

        [DllImport("LibTorchSharp")]
        internal static extern UInt64 THSBackend_get_default_directml_device();
    }
}
