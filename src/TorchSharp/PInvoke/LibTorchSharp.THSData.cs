// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
#pragma warning disable CA2101
        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSData_loaderMNIST(
            [MarshalAs(UnmanagedType.LPStr)] string filename,
            long batchSize,
            [MarshalAs(UnmanagedType.U1)] bool isTrain);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSData_loaderCIFAR10(
            [MarshalAs(UnmanagedType.LPStr)] string path,
            long batchSize,
            [MarshalAs(UnmanagedType.U1)] bool isTrain);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSData_current(IntPtr iterator, IntPtr data, IntPtr target);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSData_moveNext(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        internal static extern long THSData_size(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        internal static extern void THSData_reset(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        internal static extern void THSData_dispose(IntPtr iterator);
    }
#pragma warning restore CA2101
}
