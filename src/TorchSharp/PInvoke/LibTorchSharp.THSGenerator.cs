// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern long THSGenerator_initial_seed(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSGenerator_get_rng_state(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSGenerator_set_rng_state(IntPtr handle, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern void THSGenerator_gen_manual_seed(IntPtr handle, long seed);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSGenerator_new(ulong seed, long device_type, long device_index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSGenerator_default_generator();

        [DllImport("LibTorchSharp")]
        internal static extern void THSGenerator_dispose(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSGenerator_manual_seed(long seed);
    }
}
