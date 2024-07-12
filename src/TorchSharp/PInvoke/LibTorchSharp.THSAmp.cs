// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_amp_foreach_non_finite_check_and_unscale_(IntPtr tensors, long tLength, IntPtr found_inf, IntPtr inv_scale);

    }
}