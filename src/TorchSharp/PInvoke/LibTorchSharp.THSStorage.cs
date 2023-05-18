// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern ulong THSStorage_nbytes(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern void THSStorage_set_nbytes(IntPtr tensor, ulong nbytes);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSStorage_data_ptr(IntPtr tensor);
    }
}
