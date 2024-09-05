using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorboard;
using TorchSharp.Modules;
using TorchSharp.Utils;

namespace TorchSharp.Amp
{
    public class GradScaler : IDisposable
    {
        private bool Enabled;
        public torch.Device device;
        private torch.Tensor _scale, _growth_tracker;
        private float InitScale, InitGrowthTracker;
        public float _growth_factor { set; get; }
        public float _backoff_factor { set; get; }
        private int _growth_interval { set; get; }
        private UnorderedMap<int, UnorderedMap<string, object>> _per_optimizer_states = new UnorderedMap<int, UnorderedMap<string, object>>();
        bool disposedValue;

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
            this._growth_factor = growth_factor;
            _backoff_factor = backoff_factor;
            _growth_interval = growth_interval;
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
            float? retval=0;
            foreach(var d in optimizer_state)
                if (d.Value is torch.Tensor t)
                    retval += t.item<float>();
            if (retval==0)
                retval = optimizer.step().item<float>();
            return retval;
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
                bool has_grad_scaler = false;//I dont know how deal this...
                if (has_grad_scaler) {

                } else {
                    if (optimizer_state["stage"] is OptState optstate && optstate == OptState.Ready)
                        check_inf_per_device(optimizer);
                    var scaler = _get_scale_async();
                    Debug.Assert(!scaler.is_null(), "!scaler.is_null()");
                    torch.Tensor found_inf;
                    if (optimizer_state["found_inf_per_device"] is torch.Tensor[] ts) {
                        for (int i = 0; i < ts.Length; i++)
                            ts[i].to(scaler.device, true);
                        found_inf=torch.sum(torch.cat(ts));
                    }
                    //if(optimizer is SGD ad)
                    //Info: All optimizer have grad_scale and found_inf //https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py, etc.
                    //DANGER: Optimizer in TorchShapr not have grad_scaler or found_inf, we need grad_scale for https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/amp/grad_scaler.py#L440

                    //optimizer.GetType().GetField("grad_scale").GetValue(optimizer) as torch.Tensor t
                }
                retval = optimizer.step().item<float>();
                optimizer_state["stage"] = OptState.Stepped;
                //https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/amp/grad_scaler.py#L445
                return retval;
            }
            if (optimizer_state["stage"] is OptState state1 && state1 == OptState.Ready)
                unscale(optimizer);
            Debug.Assert((optimizer_state["found_inf_per_device"] as torch.Tensor[]).Length > 0, "(optimizer_state['found_inf_per_device'] as torch.Tensor).size(0) > 0");
            retval = maybe_opt_step(optimizer, optimizer_state);
            optimizer_state["stage"] = OptState.Stepped;
            return retval;
        }

        private torch.Tensor _get_scale_async()
        {
            return _scale;
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
                IList<torch.Tensor> found_infs = new List<torch.Tensor>();
                foreach (var state in _per_optimizer_states)
                    foreach (var found_inf in state.Value)
                        if(found_inf.Value is torch.Tensor t)
                            found_infs.Add(t);
                Debug.Assert(found_infs.Count > 0, "No inf checks were recorded prior to update.");
                torch.Tensor found_inf_combined = found_infs[0];
                if (found_infs.Count > 1)
                    for (int i = 1; i < found_infs.Count; i++)
                        found_inf_combined += found_infs[i];
                torch.amp_update_scale_(_scale, _growth_tracker, found_inf_combined, (double)_growth_factor, (double)_backoff_factor, (long)_growth_interval);

            }
            //TODO: Implement defaultdict https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/torch/amp/grad_scaler.py#L531
        }

        public float get_scale()
        {
            if (this.Enabled) {

                var scale = _get_scale_async();
                if (scale.is_null())
                    return InitScale;
                return scale.item<float>();
            }
            return 1.0f;
        }

        public bool IsEnabled()
        {
            return this.Enabled;
        }

        public UnorderedMap<string, object> state_dict()
        {
            if (Enabled) {
                var res = new UnorderedMap<string, object>();
                res["scale"] = get_scale();
                res[nameof(_growth_factor)] = _growth_factor;
                res[nameof(_backoff_factor)] = _backoff_factor;
                res[nameof(_growth_interval)] = _growth_interval;
                res[nameof(_growth_tracker)] = _growth_tracker;
                return res;
            }
            return null;
        }

        public void load_state_dict(Dictionary<string, object> state_dict)
        {
            if (!Enabled)
                return;
            if (state_dict.Count == 0)
                throw new Exception("The source state dict is empty, possibly because it was saved from a disabled instance of GradScaler.");
            //TODO: implement reflection to set field/properties based on state_dict
        }

        torch.Tensor check_inf_per_device(torch.optim.Optimizer optimizer)
        {
            _scale = check_scale_growth_tracker(nameof(check_inf_per_device)).Item1;
            var dummy_inv_scale = torch.full(new ReadOnlySpan<long>(new long[] { 0 }), 1.0f, torch.ScalarType.Float32, _scale.device);
            var foundd_inf = torch.full(new ReadOnlySpan<long>(new long[] { 0 }), 0.0f, torch.ScalarType.Float32, _scale.device);
            _per_optimizer_states[optimizer.GetHashCode()]["found_inf_per_device"] = unscale_grads(optimizer, dummy_inv_scale, foundd_inf, true);
            return _per_optimizer_states[optimizer.GetHashCode()]["found_inf_per_device"] as torch.Tensor;
        }

        private object _found_inf_per_device(torch.optim.Optimizer optimizer)
        {
            return _per_optimizer_states[optimizer.GetHashCode()]["found_inf_per_device"];
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue) {
                if (disposing) {
                    _per_optimizer_states.Dispose();
                    _growth_tracker.Dispose();
                    _scale.Dispose();
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~GradScaler()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}