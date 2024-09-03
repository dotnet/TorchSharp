using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using TorchSharp.Utils;

namespace TorchSharp.Amp
{
    public class GradScaler
    {
        private bool Enabled;
        public torch.Device device;
        private torch.Tensor _scale, _growth_tracker;
        private float InitScale, GrowthFactor, BackoffFactor, GrowthInterval, InitGrowthTracker;
        private UnorderedMap<int, UnorderedMap<string, object>> _per_optimizer_states = new UnorderedMap<int, UnorderedMap<string, object>>();

        public enum OptState
        {
            Ready,
            Unscaled,
            Stepped
        }

        private UnorderedMap<string, object> _refresh_per_optimizer_state()
        {
            return new UnorderedMap<string, object>() {
                { "state", OptState.Ready }, { "found_inf_per_device", null}
            };
        }
        //https://github.com/pytorch/pytorch/blob/main/torch/amp/grad_scaler.py
        public GradScaler(torch.Device dev, float init_scale = 2.0e16f, float growth_factor = 2.0f,
            float backoff_factor = 0.5f, int growth_interval = 2000, bool enabled = true)
        {
            Debug.Assert(dev == torch.CPU || dev == torch.CUDA);
            device = dev;
            Enabled = enabled;
            InitScale = init_scale;
            GrowthFactor = growth_factor;
            BackoffFactor = backoff_factor;
            GrowthInterval = growth_interval;
            InitGrowthTracker = 0.0f;

            throw new NotImplementedException("This need to finish");
        }

        private Tuple<torch.Tensor, torch.Tensor> check_scale_growth_tracker(string name)
        {
            var fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration.";
            Debug.Assert(_scale.is_null(), $"Attempted {name} but {nameof(_scale)} is None {fix}");
            Debug.Assert(_growth_tracker.is_null(), $"Attempted {name} but {nameof(_growth_tracker)} is None {fix}");
            return new Tuple<torch.Tensor, torch.Tensor>(_scale, _growth_tracker);
        }


        private void LazyInitScaleGrowthTracker(torch.Device dev)
        {
            _scale = torch.full(0, InitScale, torch.ScalarType.Float32, device: dev);
            _growth_tracker = torch.full(0, InitGrowthTracker, torch.ScalarType.Int32, device: dev);
        }
        //private Dictionary<string, object>

        //private check_scale_growth_tracker
        public torch.Tensor scale(torch.Tensor output)
        {
            if (!Enabled)
                return output;
            if (_scale.is_null())
                LazyInitScaleGrowthTracker(output.device);
            Debug.Assert(!_scale.is_null());
            return output * _scale.to(output.device, output.dtype, true);
        }

        public IList<torch.Tensor> scale(IList<torch.Tensor> outputs)
        {
            apply_scale(outputs);
            return outputs;
        }
        private class MultiDeviceReplicator
        {
            private readonly torch.Tensor master;

            internal readonly Dictionary<torch.Device, torch.Tensor> per_device_tensors = new Dictionary<torch.Device, torch.Tensor>();
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

            check_scale_growth_tracker(nameof(unscale));
            //if(_per_optimizer_states.ContainsKey(optimizer.GetHashCode()))

            var optimizer_state = _per_optimizer_states[optimizer.GetHashCode()];
            if (optimizer_state["state"] is OptState state) {
                if (state == OptState.Unscaled) {
                    throw new Exception($"{nameof(unscale)} has already been called on this optimizer since the last update()");
                }
                else if(state == OptState.Stepped)
                    throw new Exception($"{nameof(unscale)} is being called after step()");
            }

            Debug.Assert(!_scale.is_null());
            var inv_scale = _scale.@double().reciprocal().@float();
            var found_inf = torch.full(new ReadOnlySpan<long>(new long[] { 0 }), 0.0f, torch.ScalarType.Float32,_scale.device);

            optimizer_state["found_inf_per_device"] = unscale_grads(optimizer, inv_scale, found_inf, false);

            optimizer_state["stage"] = OptState.Unscaled;
        }

        private float? maybe_opt_step(torch.optim.Optimizer optimizer, UnorderedMap<string, object> optimizer_state)
        {
            //https://github.com/pytorch/pytorch/blob/a00fad017719346bac6e08da0819358146e647e3/torch/amp/grad_scaler.py#L351
            throw new NotImplementedException();
        }

        public float? step(torch.optim.Optimizer optimizer, params object[] obj)
        {
            if (obj.Length == 0)
                throw new Exception("The obj param cannot be empty");
            if (!Enabled) {
                if(obj.Length == 1 && obj[0] is Func<torch.Tensor> closure)
                    return optimizer.step(closure).item<float>();
                return null;
            }

            check_scale_growth_tracker(nameof(step));
            var optimizer_state = _per_optimizer_states[optimizer.GetHashCode()];
            if (optimizer_state["stage"] is OptState state && state == OptState.Stepped)
                throw new Exception($"{nameof(step)} has already been called since the last update()");
            float? retval;

            //https://github.com/pytorch/pytorch/blob/a00fad017719346bac6e08da0819358146e647e3/torch/amp/grad_scaler.py#L398
            var f = optimizer.GetType().GetField("_step_support_amp_scaling");
            if (f != null && f.GetValue(optimizer) is bool b && !b) {

            }
            if (optimizer_state["stage"] is OptState state1 && state1 == OptState.Ready)
                unscale(optimizer);
            Debug.Assert((optimizer_state["found_inf_per_device"] as float[]).Length > 0, "(optimizer_state['found_inf_per_device'] as float[]).Length > 0");

            retval = maybe_opt_step(optimizer, optimizer_state);
            optimizer_state["stage"] = OptState.Stepped;
            return retval;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="new_scale">only float or torch.Tensor</param>
        public void update(object new_scale = null)
        {
            if (!Enabled)
                return;
            var tup = check_scale_growth_tracker("update");
            _scale = tup.Item1;
            _growth_tracker = tup.Item2;
            if (new_scale != null) {
                Debug.Assert(!_scale.is_null());
                if (new_scale is float f)
                    _scale.fill_(f);
                else if(new_scale is torch.Tensor t) {
                    string reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor or torch.FloatTensor with requires_grad = False.";
                    Debug.Assert(t.device == this.device, reason);
                    Debug.Assert(t.numel() == 1, reason);
                    Debug.Assert(!t.requires_grad, reason);
                    _scale.copy_(t);
                }
            } else {
                //var found_infs = 
            }
            
        }
    }
}