// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAutograd_isGradEnabled();

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_setGrad(bool enabled);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_grad(
            IntPtr outputs, long oLength,
            IntPtr inputs, long iLength,
            IntPtr grad_outs, long gLength,
            [MarshalAs(UnmanagedType.U1)] bool retain_graph,
            [MarshalAs(UnmanagedType.U1)] bool create_graph,
            [MarshalAs(UnmanagedType.U1)] bool allow_unused,
            AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_backward(
            IntPtr tensors, long tLength,
            IntPtr grad_tensors, long gtLength,
            [MarshalAs(UnmanagedType.U1)] bool retain_graph,
            [MarshalAs(UnmanagedType.U1)] bool create_graph,
            IntPtr inputs, long iLength);

    }
}
