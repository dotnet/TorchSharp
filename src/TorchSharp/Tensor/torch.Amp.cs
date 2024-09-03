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

        public static torch.Tensor amp_update_scale_(Tensor self, Tensor growth_tracker, Tensor found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval)
        {
            var res = THSAmp_amp_update_scale_(self.Handle, growth_tracker.Handle, found_inf.Handle, scale_growth_factor, scale_backoff_factor, growth_interval);
            if(res == IntPtr.Zero)
                torch.CheckForErrors();
            return new Tensor(res);
        }
        public static torch.Tensor amp_update_scale_out(Tensor outt, Tensor self, Tensor growth_tracker, Tensor found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval)
        {
            var res = THSAmp_amp_update_scale_out(outt.Handle, self.Handle, growth_tracker.Handle, found_inf.Handle, scale_growth_factor, scale_backoff_factor, growth_interval);
            if(res == IntPtr.Zero)
                torch.CheckForErrors();
            return new Tensor(res);
        }
        public static torch.Tensor amp_update_scale_outf(Tensor self, Tensor growth_tracker, Tensor found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval, Tensor outt)
        {
            var res = THSAmp_amp_update_scale_outf(self.Handle, growth_tracker.Handle, found_inf.Handle, scale_growth_factor, scale_backoff_factor, growth_interval, outt.Handle);
            if(res == IntPtr.Zero)
                torch.CheckForErrors();
            return new Tensor(res);
        }
        public static (torch.Tensor, torch.Tensor) amp_update_scale(Tensor self, Tensor growth_tracker, Tensor found_inf, double scale_growth_factor, double scale_backoff_factor, long growth_interval)
        {
            var res = THSAMP_amp_update_scale(self.Handle, growth_tracker.Handle, found_inf.Handle, scale_growth_factor, scale_backoff_factor, growth_interval, out var res1);
            if(res == IntPtr.Zero || res1 == IntPtr.Zero)
                torch.CheckForErrors();
            return (new Tensor(res), new Tensor(res1));
        }
    }
}
