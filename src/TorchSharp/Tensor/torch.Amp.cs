using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static void _amp_foreach_non_finite_check_and_unscale_(IList<Tensor> tensors, Tensor found_inf, Tensor inv_scale)
        {
            using var ts = new PinnedArray<IntPtr>();
            IntPtr tens = ts.CreateArray(tensors.Select(x => x.Handle).ToArray());
            THSAmp_amp_foreach_non_finite_check_and_unscale_(tens, ts.Array.Length, found_inf.Handle, inv_scale.Handle);
        }
    }
}
