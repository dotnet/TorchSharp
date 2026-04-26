using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern void THSHistogram_histogram(
            IntPtr input,
            long bins,
            IntPtr ranges,
            long length,
            IntPtr weight,
            [MarshalAs(UnmanagedType.U1)]bool density,
            out IntPtr hist,
            out IntPtr hist_bins
        );
        internal static extern void THSHistogram_tensor(
            IntPtr input,
            IntPtr bins,
            IntPtr weight,
            [MarshalAs(UnmanagedType.U1)] bool density,
            out IntPtr hist,
            out IntPtr hist_bins);
        internal static extern void THSHistogramdd(
            IntPtr input,
            IntPtr bins,
            long length,
            IntPtr ranges,
            long length_ranges,
            IntPtr weight,
            [MarshalAs(UnmanagedType.U1)] bool density,
            out IntPtr hist,
            AllocatePinnedArray allocator);
        internal static extern void THSHistogramdd_intbins(
            IntPtr input,
            long bins,
            IntPtr ranges,
            long length_ranges,
            IntPtr weight,
            [MarshalAs(UnmanagedType.U1)] bool density,
            out IntPtr hist,
            AllocatePinnedArray allocator);
        internal static extern void THSHistogramdd_tensors(
            IntPtr input,
            IntPtr tensors,
            long length,
            IntPtr ranges,
            long length_ranges,
            IntPtr weight,
            [MarshalAs(UnmanagedType.U1)] bool density,
            IntPtr hist,
            AllocatePinnedArray allocator);
    }
}
