using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;

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
            Enabled = enabled;
            InitScale = init_scale;
            GrowthFactor = growth_factor;
            BackoffFactor = backoff_factor;
            GrowthInterval = growth_interval;
            InitGrowthTracker = 0.0f;
            throw new NotImplementedException();
        }

        private void LazyInitScaleGrowthTracker(torch.Device dev)
        {
            _scale = torch.full(0, InitScale, torch.ScalarType.Float32, device: dev);
            _growth_tracker = torch.full(0, InitGrowthTracker, torch.ScalarType.Int32, device: dev);
        }

        //private check_scale_growth_tracker
        public torch.Tensor scale(torch.Tensor output)
        {
            if (!Enabled)
                return output;
            if (_scale.is_null())
                LazyInitScaleGrowthTracker(output.device);
            return output * _scale.to(output.device, output.dtype, true);
        }

        public IList<torch.Tensor> scale(IList<torch.Tensor> outputs)
        {
            apply_scale(outputs);
            return outputs;
        }
        private class MultiDeviceReplicator
        {
            private torch.Tensor master;

            internal Dictionary<torch.Device, torch.Tensor> per_device_tensors = new Dictionary<torch.Device, torch.Tensor>();
            public MultiDeviceReplicator(torch.Tensor master_tensor)
            {
                master = master_tensor;
            }

            public torch.Tensor Get(torch.Device device)
            {
                torch.Tensor retval=null;
                if (!per_device_tensors.ContainsKey(device)) {
                    retval = master.to(device, true, non_blocking: true);
                    per_device_tensors.Add(device, retval);
                }
                return retval;
            }
        }
        
        private torch.Tensor apply_scale(torch.Tensor scale)
        {
            IList<MultiDeviceReplicator> stash = new List<MultiDeviceReplicator>();
            if (stash.Count == 0) {
                if (_scale.is_null()) {
                    LazyInitScaleGrowthTracker(scale.device);
                }
                stash.Add(new MultiDeviceReplicator(_scale));
            }
            return scale * stash[0].Get(scale.device);
        }

        private void apply_scale(IList<torch.Tensor> scales)
        {
            for (int i = 0; i < scales.Count; i++)
                scales[i] = apply_scale(scales[i]);
        }
        public Dictionary<torch.Device, torch.Tensor> unscale_grads(torch.optim.Optimizer optimizer, torch.Tensor inv_scale, torch.Tensor found_inf, bool allow_fp16)
        {
            var per_device_inv_scale = new MultiDeviceReplicator(inv_scale);
            var per_device_found_inf= new MultiDeviceReplicator(found_inf);
            Dictionary<torch.Device, Dictionary<torch.ScalarType, IList<torch.Tensor>>> per_device_and_dtype_grads = new Dictionary<torch.Device, Dictionary<torch.ScalarType, IList<torch.Tensor>>>();

            using (torch.no_grad()) {
                if (optimizer is AdamW adamW){ //Some optimizer have parameter tensor for unscale_grads i need that.
                    using (var enumer = adamW.parameters().GetEnumerator()) {
                        while (enumer.MoveNext()) {
                            var param = enumer.Current;
                            if (param.is_null()) 
                                continue;
                            if (!allow_fp16 && param.dtype == torch.ScalarType.Float16)
                                throw new Exception("Attempting to unscale FP16 Gradients");
                            torch.Tensor to_unscale;
                            if (param.grad.is_sparse) {
                                if (param.grad.dtype == torch.ScalarType.Float16) {
                                    
                                    param.grad = param.grad.coalesce();
                                }

                                to_unscale = param.grad.SparseValues;
                            } else {
                                to_unscale = param.grad;
                            }

                            if (!per_device_and_dtype_grads.ContainsKey(to_unscale.device)) {
                                per_device_and_dtype_grads.Add(to_unscale.device, new Dictionary<torch.ScalarType, IList<torch.Tensor>>());
                                per_device_and_dtype_grads[to_unscale.device].Add(to_unscale.dtype, new List<torch.Tensor>());
                                per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].Add(to_unscale);
                            } else {
                                if (!per_device_and_dtype_grads[to_unscale.device].ContainsKey(to_unscale.dtype)) {
                                    per_device_and_dtype_grads[to_unscale.device].Add(to_unscale.dtype, new List<torch.Tensor>());
                                } else {
                                    per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].Add(to_unscale);
                                }
                            }

                        }
                    }

                    foreach (var d in per_device_and_dtype_grads)
                        foreach (var g in d.Value)
                            torch._amp_foreach_non_finite_check_and_unscale_(g.Value, per_device_found_inf.Get(d.Key), per_device_inv_scale.Get(d.Key));
                }
            }

            return per_device_found_inf.per_device_tensors;
        }

        public void unscale(torch.optim.Optimizer optimizer)
        {
            if (!Enabled)
                return;
        }
    }
}