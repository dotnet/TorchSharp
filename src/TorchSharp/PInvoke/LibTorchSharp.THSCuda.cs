// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_manual_seed(long seed);

        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_manual_seed_all(long seed);

        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_synchronize(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern bool THSBackend_cuda_get_allow_tf32();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_allow_tf32(bool flag);

        [DllImport("LibTorchSharp")]
        internal static extern bool THSBackend_cudnn_get_allow_tf32();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cudnn_set_allow_tf32(bool flag);

        [DllImport("LibTorchSharp")]
        internal static extern bool THSBackend_cuda_get_allow_fp16_reduced_precision_reduction();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_allow_fp16_reduced_precision_reduction(bool flag);

        [DllImport("LibTorchSharp")]
        internal static extern bool THSBackend_cuda_get_enable_flash_sdp();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_enable_flash_sdp(bool flag);

        [DllImport("LibTorchSharp")]
        internal static extern bool THSBackend_cuda_get_enable_math_sdp();
        [DllImport("LibTorchSharp")]
        internal static extern void THSBackend_cuda_set_enable_math_sdp(bool flag);
    }
}
