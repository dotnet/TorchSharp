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
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAmp_amp_update_scale_(IntPtr self, IntPtr growth_tracker, IntPtr found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAmp_amp_update_scale_out(IntPtr outt,IntPtr self, IntPtr growth_tracker,  IntPtr found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAmp_amp_update_scale_outf(IntPtr self,IntPtr growth_tracker,  IntPtr found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval, IntPtr outt);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAMP_amp_update_scale(IntPtr self,IntPtr growth_tracker,  IntPtr found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval, out IntPtr sec);
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_torch_function_mode_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_cache_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_cpu_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_gpu_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_xpu_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_hpu_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_ipu_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern bool THSAmp_is_autocast_xla_enabled();
        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSAmp_get_autocast_cpu_dtype();
        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSAmp_get_autocast_gpu_dtype();
        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSAmp_get_autocast_xpu_dtype();
        [DllImport("LibTorchSharp")]
        internal static extern int THSAmp_autocast_increment_nesting();
        [DllImport("LibTorchSharp")]
        internal static extern int THSAmp_autocast_decrement_nesting();
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_set_autocast_enabled(bool enabled);
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_set_autocast_cache_enabled(bool enabled);
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_set_autocast_cpu_dtype(sbyte dtype);
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_set_autocast_gpu_dtype(sbyte dtype);
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_set_autocast_xpu_dtype(sbyte dtype);
        [DllImport("LibTorchSharp")]
        internal static extern void THSAmp_clear_autocast_cache();


    }
}