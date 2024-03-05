using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.Amp
{
    public class GradScaler
    {
        private bool Enabled;

        private torch.Tensor _scale, _growth_tracker;

        private float InitScale, GrowthFactor, BackoffFactor, GrowthInterval, InitGrowthTracker;

        //https://github.com/pytorch/pytorch/blob/main/torch/amp/grad_scaler.py
        public GradScaler(torch.Device dev, float init_scale = 2.0e16f, float growth_factor = 2.0f,
            float backoff_factor = 0.5f, int growth_interval = 2000, bool enabled = true)
        {
            Debug.Assert(dev == torch.CPU || dev == torch.CUDA);
            this.Enabled = enabled;
            this.InitScale = init_scale;
            this.GrowthFactor = growth_factor;
            this.BackoffFactor = backoff_factor;
            this.GrowthInterval = growth_interval;
            this.InitGrowthTracker = 0.0f;
            throw new NotImplementedException();
        }

        private void LazyInitScaleGrowthTracker(torch.Device dev)
        {
            this._scale = torch.full(0, this.InitScale, torch.ScalarType.Float32, device: dev);
            this._growth_tracker = torch.full(0, this.InitGrowthTracker, torch.ScalarType.Float32, device: dev);
        }

        //private check_scale_growth_tracker
        public torch.Tensor scale(torch.Tensor output)
        {
            if (!Enabled)
                return output;
            if (_scale.numel() == 0)
                this.LazyInitScaleGrowthTracker(output.device);
            return output * this._scale.to(output.device, output.dtype, true);
        }

        public torch.Tensor unscale_grads(torch.optim.Optimizer optimizer, torch.Tensor inv_scale, torch.Tensor found_inf, bool allow_fp16)
        {
            return false;
        }

        public void unscale(torch.optim.Optimizer optimizer)
        {
            if (!Enabled)
                return;

            
        }
        /*public IList<torch.Tensor> scale(IList<torch.Tensor> outputs)
        {


        }*/
    }
}