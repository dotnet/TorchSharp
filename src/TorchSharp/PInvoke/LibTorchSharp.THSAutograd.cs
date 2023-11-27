// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_isGradEnabled();

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_setGrad([MarshalAs(UnmanagedType.U1)] bool enabled);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_isInferenceModeEnabled();

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAutograd_getInferenceModeGuard([MarshalAs(UnmanagedType.U1)] bool mode);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_deleteInferenceModeGuard(IntPtr guard);
        
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_isAnomalyEnabled();

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_shouldCheckNaN();

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_setAnomaly([MarshalAs(UnmanagedType.U1)] bool enabled, [MarshalAs(UnmanagedType.U1)] bool check_nan);

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
