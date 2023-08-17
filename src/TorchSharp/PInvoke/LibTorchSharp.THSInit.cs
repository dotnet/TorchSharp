// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern double THSInit_calculate_gain(long nonlinearity, double param);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_constant_(IntPtr tensor, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_dirac_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_eye_(IntPtr matrix);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_normal_(IntPtr tensor, double mean, double std);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_trunc_normal_(IntPtr tensor, double mean, double std, double a, double b);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_ones_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_orthogonal_(IntPtr tensor, double gain);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_sparse_(IntPtr tensor, double sparsity, double std);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_uniform_(IntPtr tensor, double low, double high);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_kaiming_normal_(IntPtr tensor, double a, long mode, long nonlinearity);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_kaiming_uniform_(IntPtr tensor, double a, long mode, long nonlinearity);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_xavier_normal_(IntPtr tensor, double gain);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_xavier_uniform_(IntPtr tensor, double gain);

        [DllImport("LibTorchSharp")]
        internal static extern void THSInit_zeros_(IntPtr tensor);
    }
}
