// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTorchCuda_is_available();

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTorchCuda_cudnn_is_available();

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorchCuda_device_count();

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorchCuda_synchronize(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorchCuda_empty_cache();

        [DllImport("LibTorchSharp")]
        internal static extern ulong THSTorchCuda_memory_allocated(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern ulong THSTorchCuda_max_memory_allocated(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorchCuda_reset_peak_memory_stats(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern ulong THSTorchCuda_memory_reserved(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern ulong THSTorchCuda_max_memory_reserved(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorchCuda_mem_get_info(long device_index, out ulong free, out ulong total);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTorchCuda_set_device(long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTorchCuda_current_device();
    }
}
