// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_manual_seed(long seed);

        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_manual_seed_all(long seed);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSBackend_cublas_get_allow_tf32();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cublas_set_allow_tf32([MarshalAs(UnmanagedType.U1)] bool flag);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSBackend_cudnn_get_allow_tf32();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cudnn_set_allow_tf32([MarshalAs(UnmanagedType.U1)] bool flag);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSBackend_cuda_get_allow_fp16_reduced_precision_reduction();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_allow_fp16_reduced_precision_reduction([MarshalAs(UnmanagedType.U1)] bool flag);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSBackend_cuda_get_enable_flash_sdp();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_enable_flash_sdp([MarshalAs(UnmanagedType.U1)] bool flag);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSBackend_cuda_get_enable_math_sdp();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_enable_math_sdp([MarshalAs(UnmanagedType.U1)] bool flag);

        [DllImport("LibTorchSharp")]
        internal static extern int THSCuda_get_major_compute_capability(int device=0);
        [DllImport("LibTorchSharp")]
        internal static extern int THSCuda_get_minor_compute_capability(int device = 0);
        [DllImport("LibTorchSharp")]
        internal static extern int THSCuda_get_device_count(ref int count);
        [DllImport("LibTorchSharp")]
        internal static extern int THSCuda_get_free_total(int device, ref int id, ref ulong free, ref ulong total);
        [DllImport("LibTorchSharp")]
        internal static extern ulong THSCuda_get_total_memory(int device);
        [DllImport("LibTorchSharp")]
        internal static extern ulong THSCuda_get_global_total_memory(int device);
    }
}
