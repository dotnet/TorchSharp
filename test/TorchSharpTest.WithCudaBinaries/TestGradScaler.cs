using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Amp;
using TorchSharp.Modules;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace TorchSharpTest.WithCudaBinaries
{
    public class TestGradScaler
    {
        internal DeviceType device = DeviceType.CUDA;
        internal ScalarType dtype = ScalarType.Float32;

        private (Sequential modctrl, Sequential modscal, torch.optim.Optimizer optctrl, torch.optim.Optimizer optscal) create_scaling_model_optimizer(DeviceType dev = DeviceType.CUDA)
        {
            var mod_control =Sequential(torch.nn.Linear(8,8), torch.nn.Linear(8, 8));
            mod_control.to(dev);
            var mod_scaling = Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8));
            mod_scaling.to(dev);

            using (torch.no_grad()) {

                using (var enumer = mod_control.parameters().Zip(mod_scaling.parameters()).GetEnumerator())
                    while (enumer.MoveNext())
                        enumer.Current.Second.copy_(enumer.Current.First);

                var opt_control = torch.optim.SGD(mod_control.parameters(), 1.0f);
                var opt_scaling = torch.optim.SGD(mod_scaling.parameters(), 1.0f);
                return (mod_control, mod_scaling, opt_control, opt_scaling);
            }
        }
        internal (Sequential modctrl, Sequential modscal, torch.optim.Optimizer optctrl, torch.optim.Optimizer optscal, List<KeyValuePair<torch.Tensor, torch.Tensor>> data, MSELoss loss_fn, int skip_iter) create_scaling_case(DeviceType dev = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
        {
            var data = new List<KeyValuePair<torch.Tensor, torch.Tensor>>() {
                new(torch.randn(new long[]{8,8}, dtype, new Device(dev)),torch.randn(new long[]{8,8}, dtype, new Device(dev))),
                new(torch.randn(new long[]{8,8}, dtype, new Device(dev)),torch.randn(new long[]{8,8}, dtype, new Device(dev))),
                new(torch.randn(new long[]{8,8}, dtype, new Device(dev)),torch.randn(new long[]{8,8}, dtype, new Device(dev))),
                new(torch.randn(new long[]{8,8}, dtype, new Device(dev)),torch.randn(new long[]{8,8}, dtype, new Device(dev))),
            };

            var loss_fn = MSELoss();
            loss_fn.to(DeviceType.CUDA);
            const int skip_iter = 2;
            var csmo = create_scaling_model_optimizer(dev);
            return (csmo.modctrl, csmo.modscal, csmo.optctrl, csmo.optscal, data, loss_fn, skip_iter);
        }
        internal void run_scaling_case(Action<List<KeyValuePair<torch.Tensor, torch.Tensor>>, Sequential, torch.optim.Optimizer, GradScaler, MSELoss, int, bool> run, int unskipped, int skipped, double atol = 1e07)
        {
            const double rtol = 1e-7d;
            bool[] enableds = new bool[] { true, false };
            foreach (var enabled in enableds) {
                var res =create_scaling_case();
                var scaler = new GradScaler(new Device(DeviceType.CUDA), 128.0f, 2.0f, growth_interval: 1);
                run.Invoke(res.data, res.modctrl, res.optctrl, scaler, res.loss_fn, res.skip_iter, false);
                run.Invoke(res.data, res.modscal, res.optscal, scaler, res.loss_fn, res.skip_iter, true);
                if (enabled) {
                    var net_growth = unskipped > 0 ? MathF.Pow(scaler.get_growth_factor(), unskipped) : 1.0f;
                    var net_backoff = skipped> 0 ? MathF.Pow(scaler.get_backoff_factor(), skipped) : 1.0f;
                    Assert.Equal((128.0f * net_growth * net_backoff), scaler.get_scale());
                    
                } else {
                    Assert.Equal(1.0f, scaler.get_scale());
                }

                foreach(var seq in res.modctrl.parameters().Zip(res.modscal.parameters())){
                    var c_grad = seq.First.grad;
                    var s_grad = seq.Second.grad;
                    if(!(c_grad is null) && !(s_grad is null))
                        Assert.True(torch.allclose(seq.First.grad, seq.Second.grad, rtol, atol));
                    var c_state = res.optctrl.ParamGroups;
                    var s_state = res.optscal.ParamGroups;
                    foreach(var c_s_state in c_state.Zip(s_state)) {
                        if (c_s_state.First is ParamGroup pg_c_state && c_s_state.Second is ParamGroup pg_s_state) {
                            foreach (var c_s_state_p in pg_c_state.Parameters.Zip(pg_s_state.Parameters))
                                Assert.True(torch.allclose(c_s_state_p.First, c_s_state_p.Second, rtol, atol));
                        }
                    }
                    Assert.True(torch.allclose(seq.First, seq.Second, rtol, atol));
                }
            }
        }
       
        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingUnscaleSparse()
        {
            var scaler = new GradScaler(new Device(device));
            var inv_scale = torch.full(1, 0.25, dtype, new Device(device));
            var found_inf = torch.empty(1, dtype, new Device(device));
            var cur = found_inf.device;
            var i = torch.tensor(new long[,] { { 0, 1, 1 }, { 2, 0, 2 } }, ScalarType.Int64, new Device(DeviceType.CUDA));
            var v = torch.tensor(new float[] { 16.0f,32.0f,64.0f}, ScalarType.Float32, new Device(DeviceType.CUDA));
            var s = torch.sparse_coo_tensor(i,v, new long[]{2,3}, dtype, new Device(DeviceType.CUDA));

            var p = s.clone();
            Assert.True(p.is_sparse);
            var optA = torch.optim.SGD(new[] { new Parameter(p) }, 1.0);
            p.grad = s.clone();
            found_inf.zero_();
            found_inf = scaler.unscale_grads(optA, inv_scale, found_inf, false)[cur];

            Assert.Equal(0.0f, found_inf.item<float>());
            Assert.True(torch.equal(p.grad.to_dense(), (s/4).to_dense()).item<bool>());

            v = torch.tensor(new float[] { 16.0f, 32.0f, float.PositiveInfinity });
            p.grad = torch.sparse_coo_tensor(i, v, new long[] { 2, 3 }, dtype, new Device(DeviceType.CUDA));
            found_inf.zero_();
            found_inf = scaler.unscale_grads(optA, inv_scale, found_inf, false)[cur];
            Assert.Equal(1.0f, found_inf.item<float>());

            v = torch.tensor(new float[] { 16.0f, 32.0f, float.NaN });
            p.grad = torch.sparse_coo_tensor(i, v, new long[] { 2, 3 }, dtype, new Device(DeviceType.CUDA));
            found_inf.zero_();
            found_inf = scaler.unscale_grads(optA, inv_scale, found_inf, false)[cur];
            Assert.Equal(1.0f, found_inf.item<float>());

            p = s.clone().to(ScalarType.Float16);
            Assert.True(p.is_sparse);
            var optB = torch.optim.SGD(new Parameter[] { new Parameter(p) }, 1.0);

            p.grad = s.clone().to(ScalarType.Float16);
            found_inf.zero_();
            found_inf = scaler.unscale_grads(optB, inv_scale, found_inf, true)[cur];
            Assert.Equal(0.0f, found_inf.item<float>());
            Assert.True(torch.equal(p.grad.to_dense(), (s.to(ScalarType.Float16) / 4).to_dense()).item<bool>());

            i = torch.tensor(new long[,] { { 0, 1, 0 }, { 2, 0, 2 } });
            v = torch.tensor(new float[] { 64000.0f, 32.0f, 64000.0f });
            p.grad = torch.sparse_coo_tensor(i, v, new long[] { 2, 3 }, dtype, new Device(DeviceType.CUDA));
            found_inf.zero_();
            found_inf = scaler.unscale_grads(optB, inv_scale, found_inf, true)[cur];
            Assert.Equal(0.0f, found_inf.item<float>());
        }

        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingStateDict()
        {
            bool[] lazy_init_scale = new[] { true, false };
            foreach (var l in lazy_init_scale) {
                var s0 = new GradScaler(new Device(DeviceType.CUDA), 3.0f, 4.0f, 0.5f, 2);
                var s1 = new GradScaler(new Device(DeviceType.CUDA), 6.0f, 7.0f, 0.8f, 1);
                s1.set_init_growth_tracker(7);
                if (l) {
                    s1.scale(torch.full(1, 4.0f, ScalarType.Float32, new Device(DeviceType.CUDA, 0)));
                    Assert.Equal(ScalarType.Float32, s1.get_scale_async().dtype);
                }

                var re = s0.state_dict();
                s1.load_state_dict(re);

                Assert.Equal(3.0f, s1.get_scale());
                Assert.Equal(0.5f, s1.get_growth_factor());
                Assert.Equal(2, s1.get_growth_interval());
                Assert.Equal(0.0f, s1.get_init_growth_tracker());
            }
        }

        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScaleWillNotOverflow()
        {
            var model = torch.nn.Linear(5, 1).to(DeviceType.CUDA);
            var optimizer = torch.optim.Adam(model.parameters());
            var scaler = new GradScaler(new Device(DeviceType.CUDA), 1e38f, MathF.Pow(2.0f, 4), growth_interval:1);
            optimizer.zero_grad();
            var x = torch.randn(new long[]{1,5}).to(DeviceType.CUDA);
            var y = 1e-30 * torch.randn(new long[]{1,1}).to(DeviceType.CUDA);
            var l = torch.pow(model.forward(x) - y, 2).mean();
            scaler.scale(l).backward();
            scaler.step(optimizer);
            scaler.update();
            Assert.True(!scaler.get_scale_async().isinf().item<bool>() && !scaler.get_scale_async().isnan().item<bool>());
        }
        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingClipping()
        {
            run_scaling_case(new Action<List<KeyValuePair<Tensor, Tensor>>, Sequential, optim.Optimizer, GradScaler, MSELoss, int, bool>((
                (data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api) => {
                    const float max_norm = 0.2f;
                    int idx = 0;
                    foreach (var ipair in data) {
                        //ipair.
                        optimizer.zero_grad();
                        var output = model.forward(ipair.Key);
                        var loss = loss_fn.forward(output, ipair.Value);
                        if (try_scaling_api) {
                            scaler.scale(loss).backward();
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale());
                            if (idx == skip_iter && scaler.IsEnabled()) {
                                var weight = (model[1] as Linear)?.weight;
                                if (weight.is_null())
                                    throw new ArgumentNullException(nameof(weight));
                                weight.grad.fill_(float.PositiveInfinity);
                            }

                            scaler.step(optimizer);
                            scaler.update();
                        } else {
                            loss.backward();
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm);
                            if (!scaler.IsEnabled() || (idx != skip_iter))
                                optimizer.step();
                        }

                        idx++;
                    }
                })),
                3, 1, 1e-5);
        }
        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingClippingSeparateUnscale()
        {
            run_scaling_case(new Action<List<KeyValuePair<Tensor, Tensor>>, Sequential, optim.Optimizer, GradScaler, MSELoss, int, bool>((
                (data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api) => {
                    const float max_norm = 0.2f;
                    int idx = 0;
                    foreach (var ipair in data) {
                        //ipair.
                        optimizer.zero_grad();
                        var output = model.forward(ipair.Key);
                        var loss = loss_fn.forward(output, ipair.Value);
                        if (try_scaling_api) {
                            scaler.scale(loss).backward();
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm);
                            if (idx == skip_iter && scaler.IsEnabled()) {
                                var weight = (model[1] as Linear)?.weight;
                                weight.grad.fill_(float.PositiveInfinity);
                            }

                            scaler.step(optimizer);
                            scaler.update();
                        } else {
                            loss.backward();
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm);
                            if (!scaler.IsEnabled() || (idx != skip_iter))
                                optimizer.step();
                        }

                        idx++;
                    }
                })),
            3, 1);
        }
        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingPenalty()
        {
            
            run_scaling_case(new Action<List<KeyValuePair<Tensor, Tensor>>, Sequential, optim.Optimizer, GradScaler, MSELoss, int, bool>((
                (data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api) => {
                    //const float max_norm = 0.2f;
                    int idx = 0;
                    foreach (var ipair in data) {
                        //ipair.
                        optimizer.zero_grad();
                        var output = model.forward(ipair.Key);
                        var loss = loss_fn.forward(output, ipair.Value);
                        List<Tensor> grad_params = new List<Tensor>();
                        if (try_scaling_api) {
                            //throw new NotImplementedException();
                            //TODO: RESEARCH TORCH::AUTOGRAD:GRAD THE SECOND ARGUMENT SHOULD HAVE model->parameters();
                            //grad_params = torch.autograd.grad(new List<Tensor>(){scaler.scale(loss)}, model.parameters())
                            var inv_scale = 1.0f / scaler.get_scale();
                            for (int i = 0; i < grad_params.Count; i++)
                                grad_params[i] *= inv_scale;
                        } else {
                            //throw new NotImplementedException();
                            //TODO: RESEARCH TORCH::AUTOGRAD:GRAD THE SECOND ARGUMENT SHOULD HAVE model->parameters();
                            //grad_params = torch.autograd.grad(new List<Tensor>(){scaler.scale(loss)}, model.parameters())
                        }

                        var grad_norm = torch.zeros(new long[] { 1 }).to(ipair.Key.device);
                        for (int i = 0; i < grad_params.Count; i++)
                            grad_norm += grad_params[i].pow(2).sum();
                        grad_norm = grad_norm.sqrt();
                        loss = loss + grad_norm;
                        if (try_scaling_api) {
                            scaler.scale(loss).backward();
                            if (idx == skip_iter && scaler.IsEnabled()) {
                                var weight = (model[1] as Linear)?.weight;
                                weight.grad.fill_(float.PositiveInfinity);
                            }

                            scaler.step(optimizer);
                            scaler.update();
                        } else {
                            loss.backward();
                            if (!scaler.IsEnabled() || (idx != skip_iter)) {
                                optimizer.step();
                            }
                        }
                        idx++;
                    }
                })),
            3, 1);
        }
        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingAccumulation()
        {
            run_scaling_case(new Action<List<KeyValuePair<Tensor, Tensor>>, Sequential, optim.Optimizer, GradScaler, MSELoss, int, bool>((
                (data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api) => {
                    const int iters_to_accumulate= 2;
                    int idx = 0;
                    foreach (var ipair in data) {
                        //ipair.
                        optimizer.zero_grad();
                        var output = model.forward(ipair.Key);
                        var loss = loss_fn.forward(output, ipair.Value);
                        loss /= iters_to_accumulate;

                        if (try_scaling_api) {
                            scaler.scale(loss).backward();
                        } else {
                            loss.backward();
                        }

                        if ((idx + 1) % iters_to_accumulate == 0) {
                            if (try_scaling_api) {
                                scaler.step(optimizer);
                                scaler.update();
                                optimizer.zero_grad();
                            } else {
                                optimizer.step();
                                optimizer.zero_grad();
                            }
                        }
                        idx++;
                    }
                })),
            2, 0);
        }
        [Fact]
        [TestOf(nameof(GradScaler))]
        public void TestGradScalingMultiple()
        {
            throw new NotImplementedException();
        }
    }
}
