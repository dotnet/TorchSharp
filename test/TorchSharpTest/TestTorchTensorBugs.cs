// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

using System.Threading;
using System.Runtime.CompilerServices;

using static TorchSharp.torch.nn;
using Xunit;

using static TorchSharp.torch;
using static TorchSharp.torchvision.models;


using System.Numerics;

#nullable enable

namespace TorchSharp
{
    // The tests in this file are all derived from reported GitHub Issues, serving
    // as regression tests.

#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestTorchTensorBugs
    {

        [Fact]
        public void ValidateAddInplace()
        {
            using var _ = NewDisposeScope();

            var x = torch.zeros(10, 10);
            var y = torch.ones(10).expand(10, 10);

            x.add_(y, 1);

            Assert.Equal(x, y);
        }

        [Fact]
        public void ValidateIssue145()
        {
            // Tensor.DataItem gives a hard crash on GPU tensor

            if (torch.cuda.is_available()) {

                using var _ = NewDisposeScope();

                var scalar = torch.tensor(3.14f, torch.CUDA);
                Assert.Throws<InvalidOperationException>(() => scalar.item<float>());
                var tensor = torch.zeros(new long[] { 10, 10 }, device: torch.CUDA);
                Assert.Throws<InvalidOperationException>(() => tensor.data<float>());
                Assert.Throws<InvalidOperationException>(() => { var _ = tensor.bytes; });
            }
        }

        class DoubleIt : nn.Module<Tensor, Tensor>
        {
            public DoubleIt() : base("double") { }

            public override Tensor forward(Tensor t) => t * 2;
        }

        [Fact]
        public void ValidateIssue315_1()
        {
            using var _ = NewDisposeScope();

            // https://github.com/dotnet/TorchSharp/issues/315
            // custom module crash in GC thread

            // make Torch call our custom module by adding a ReLU in front of it
            using var net = nn.Sequential(
                ("relu", nn.ReLU()),
                ("double", new DoubleIt())
            );

            using var @in = torch.tensor(3);
            using var @out = net.call(@in);
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        [Fact]
        public void ValidateIssue315_2()
        {
            using var _ = NewDisposeScope();

            Func<Tensor, Tensor, Tensor> distance =
                (x, y) => {
                    return (x - y).abs();
                };

            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var output = nn.TripletMarginWithDistanceLoss(distance);
                using (var result = output.call(anchor, positive, negative)) { }
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }



        [Fact]
        public void ValidateIssue315_3()
        {
            using var _ = NewDisposeScope();

            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            using var x = torch.randn(new long[] { 64, 1000 });
            using var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate);
            var loss = nn.MSELoss(Reduction.Sum);

            Func<Tensor> closure = () => {
                using var eval = seq.call(x);
                var output = loss.call(eval, y);

                var l = output.ToSingle();

                optimizer.zero_grad();

                output.backward();
                return output;
            };

            optimizer.step(closure);

            GC.Collect();
            GC.WaitForPendingFinalizers();
        }


        private void ThreadFunc()
        {
            using var _ = NewDisposeScope();

            using var net = nn.Sequential(
                ("relu", nn.ReLU()),
                ("double", new DoubleIt())
            );

            using var @in = torch.tensor(3);

            for (var i = 0; i < 1000; i++) {
                using var @out = net.call(@in);
            }
        }

        [Fact]
        public void ValidateIssue315_4()
        {
            // Is CustomModule thread-safe?
            // See: https://github.com/pytorch/pytorch/issues/19029

            var threads = new List<Thread>();

            for (var i = 0; i < 10; i++) {
                var t = new Thread(ThreadFunc);
                threads.Add(t);
            }

            foreach (var t in threads) {
                t.Start();
            }
            foreach (var t in threads) {
                t.Join();
            }
        }

        class TestModule : Module<Tensor, Tensor>
        {
            public TestModule() : base(nameof(TestModule)) { }

            public override torch.Tensor forward(torch.Tensor t) => t.clone();

            public static void Reproduce()
            {
                Tensor t = torch.zeros(10);

                var seq = Make();
                GC.Collect();
                GC.WaitForPendingFinalizers();
                for (var i = 0; i < 100; i++) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    t = seq.call(t);
                }
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static Module<Tensor, Tensor> Make() => Sequential(("t", new TestModule()), ("d", Linear(10, 10)));
        }

        [Fact]
        void ValidateIssue321()
        {
            TestModule.Reproduce();
        }

        [Fact]
        public void ValidateIssue353()
        {
            using var _ = NewDisposeScope();

            //
            // Just validating that the methods are there.
            //
            using var x = torch.zeros(3, 3);
            using var y = torch.ones(3, 3);

            var mx1 = x.max();
            Assert.Equal(0, mx1.item<float>());

            var mx2 = x.maximum(y);
            Assert.True(mx2.allclose(y));

            var mn1 = x.min();
            Assert.Equal(0, mn1.item<float>());

            var mn2 = x.minimum(y);
            Assert.True(mn2.allclose(x));
        }

        [Fact]
        public void ValidateIssue399_1()
        {
            using var _ = NewDisposeScope();

            // Create a contiguous 3x4 matrix and fill with auto-inc values starting at zero.
            // 0 1 2 3
            // 4 5 6 7
            // 8 9 10 11
            using var contig = torch.arange(12, int32).reshape(3, 4).contiguous();
            var data1 = contig.data<int>();

            // Create a 3x2 slice of this, which should contain:
            // 1 2
            // 5 6
            // 9 10
            using var sub = contig.slice(1, 1, 3, 1);
            var data2 = sub.data<int>();

            Assert.Equal(12, data1.Count);
            Assert.Equal(6, data2.Count);

            Assert.False(sub.is_contiguous());
            Assert.True(sub.contiguous().data<int>() == data2);
            Assert.False(sub.contiguous().data<int>() != data2);
            Assert.Equal(sub.contiguous().data<int>(), data2);
            Assert.Equal(sub.contiguous().data<int>().ToArray(), data2.ToArray());
        }

        [Fact]
        public void ValidateIssue399_2()
        {
            using var _ = NewDisposeScope();

            // Create a contiguous 3x4 matrix and fill with auto-inc values starting at zero.
            // 0 1 2 3
            // 4 5 6 7
            // 8 9 10 11
            using var contig = torch.arange(12, int32).reshape(3, 4).contiguous();
            var data1 = contig.data<int>();

            // Creating the transpose, which should look like:
            // 0 4 8
            // 1 5 9
            // 2 6 10
            // 3 7 11
            using var trans = contig.t();
            var data2 = trans.data<int>();

            Assert.False(trans.is_contiguous());

            Assert.Equal(12, data1.Count);
            Assert.Equal(12, data2.Count);

            Assert.Equal(6, data2[2, 1]);
            Assert.Equal(7, data2[3, 1]);

            Assert.True(trans.contiguous().data<int>() == data2);
            Assert.False(trans.contiguous().data<int>() != data2);
            Assert.Equal(trans.contiguous().data<int>(), data2);
            Assert.Equal(trans.contiguous().data<int>().ToArray(), data2.ToArray());
        }

        [Fact]
        public void ValidateIssue399_3()
        {
            using var _ = NewDisposeScope();

            using var contig = torch.arange(27, int32).reshape(3, 3, 3).contiguous();
            using var trans = contig.permute(2, 0, 1);

            Assert.False(trans.is_contiguous());
            Assert.Equal<int>(trans.contiguous().data<int>(), trans.data<int>());
            Assert.Equal<int>(trans.contiguous().data<int>().ToArray(), trans.data<int>().ToArray());
        }

        [Fact]
        public void ValidateIssue399_4()
        {
            using var _ = NewDisposeScope();

            using var contig = torch.arange(12, int32).reshape(3, 4).contiguous();
            using var flipped = contig.t().flip(1);
            var strides = flipped.stride();

            Assert.False(flipped.is_contiguous());
            Assert.Equal<int>(flipped.contiguous().data<int>(), flipped.data<int>());
            Assert.Equal<int>(flipped.contiguous().data<int>().ToArray(), flipped.data<int>().ToArray());
        }

        [Fact]
        public void ValidateIssue399_5()
        {
            using var _ = NewDisposeScope();

            using var contig = torch.arange(12, int32).reshape(3, 4).contiguous();
            using var strided = contig.as_strided(new long[] { 3, 2, 4 }, new long[] { 4, 0, 1 });

            Assert.False(strided.is_contiguous());
            Assert.Equal<int>(strided.contiguous().data<int>(), strided.data<int>());
            Assert.Equal(new long[] { 3, 2, 4 }, strided.shape);
        }

        [Fact]
        public void ValidateIssue399_6()
        {
            using var _ = NewDisposeScope();

            using var contig = torch.arange(3, int32).reshape(3, 1).contiguous();
            using var strided = contig.expand(3, 4);

            Assert.False(strided.is_contiguous());
            Assert.Equal<int>(strided.contiguous().data<int>(), strided.data<int>());
            Assert.Equal(new long[] { 3, 4 }, strided.shape);
        }

        [Fact]
        public void ValidateIssue399_7()
        {
            using var _ = NewDisposeScope();

            using var contig = torch.arange(27, int32).reshape(3, 3, 3).contiguous();
            using var trans = contig.permute(2, 0, 1);

            Assert.False(trans.is_contiguous());

            // Test the enumerators.
            var data1 = trans.contiguous().data<int>();
            var data2 = trans.data<int>();

            var expected = new int[] { 0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 4, 7, 10, 13, 16, 19, 22, 25, 2, 5, 8, 11, 14, 17, 20, 23, 26 };

            long idx = 0;
            foreach (var value in data1) {
                Assert.Equal(expected[idx], value);
                idx += 1;
            }
            Assert.Equal(27, idx);

            idx = 0;
            foreach (var value in data2) {
                Assert.Equal(expected[idx], value);
                idx += 1;
            }
            Assert.Equal(27, idx);

            var arr1 = data1.AsEnumerable<int>().ToArray();
            var arr2 = data2.AsEnumerable<int>().ToArray();

            Assert.Equal(expected, arr1);
            Assert.Equal(arr1, arr2);
        }

        [Fact]
        public void ValidateIssue399_8()
        {
            using var _ = NewDisposeScope();

            // We need to test something that has rank 1, because the TensorAccessor uses
            // seprate enumeration logic for that.
            using var contig = torch.arange(48, int32).reshape(12, 4).contiguous();
            using var sub = contig.slice(1, 1, 2, 1).squeeze(1);

            var data1 = sub.contiguous().data<int>();
            var data2 = sub.data<int>();

            Assert.True(data1 == data2);
            Assert.False(data1 != data2);

            Assert.Equal(data1.ToArray(), data2.ToArray());

            var expected = new int[] { 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45 };

            long idx = 0;
            foreach (var value in data1) {
                Assert.Equal(expected[idx], value);
                idx += 1;
            }
            Assert.Equal(12, idx);

            idx = 0;
            foreach (var value in data2) {
                Assert.Equal(expected[idx], value);
                idx += 1;
            }
            Assert.Equal(12, idx);

            var arr1 = data1.AsEnumerable<int>().ToArray();
            var arr2 = data2.AsEnumerable<int>().ToArray();

            Assert.Equal(expected, arr1);
            Assert.Equal(arr1, arr2);
        }

        [Fact]
        public void ValidateRandThrowsOnWrongType()
        {
            using var _ = NewDisposeScope();

            Assert.Throws<ArgumentException>(() => torch.rand(3, 4, dtype: torch.int8));
            Assert.Throws<ArgumentException>(() => torch.rand(3, 4, dtype: torch.uint8));
            Assert.Throws<ArgumentException>(() => torch.rand(3, 4, dtype: torch.int16));
            Assert.Throws<ArgumentException>(() => torch.rand(3, 4, dtype: torch.int32));
            Assert.Throws<ArgumentException>(() => torch.rand(3, 4, dtype: torch.int64));
            Assert.Throws<ArgumentException>(() => torch.rand(3, 4, dtype: torch.@bool));
            Assert.Throws<ArgumentException>(() => torch.randn(3, 4, dtype: torch.int8));
            Assert.Throws<ArgumentException>(() => torch.randn(3, 4, dtype: torch.uint8));
            Assert.Throws<ArgumentException>(() => torch.randn(3, 4, dtype: torch.int16));
            Assert.Throws<ArgumentException>(() => torch.randn(3, 4, dtype: torch.int32));
            Assert.Throws<ArgumentException>(() => torch.randn(3, 4, dtype: torch.int64));
            Assert.Throws<ArgumentException>(() => torch.randn(3, 4, dtype: torch.@bool));
        }

        [Fact]
        public void ValidateIssue496()
        {
            using var _ = NewDisposeScope();

            var c2 = torch.nn.Conv2d(3, 16, kernelSize: (1, 7), stride: (1, 1), padding: (0, 3));
            var Win = torch.rand(16, 3, 8, 8);
            var s = c2.call(Win).shape;
            Assert.Equal(new long[] { 16, 16, 8, 8 }, s);
        }

        [Fact]
        public void ValidateIssue500()
        {
            using var _ = NewDisposeScope();

            using (var pool = BatchNorm1d(28)) {
                pool.eval();
                pool.call(torch.ones(1, 28));
            }
            using (var pool = BatchNorm1d(28))
            using (var seq = Sequential(pool)) {
                seq.eval();
                seq.call(torch.ones(1, 28));
            }
            using (var seq = new Module500()) {
                seq.eval();
                seq.call(torch.ones(1, 28));
            }
        }

        class Module500 : Module<Tensor, Tensor>
        {
            private Module<Tensor, Tensor> bn1 = BatchNorm1d(28);

            public Module500() : base(nameof(TestModule)) { RegisterComponents(); }

            public override torch.Tensor forward(torch.Tensor t) => bn1.call(t);
        }

        [Fact]
        public void ValidateIssue510()
        {
            using var _ = NewDisposeScope();

            var model = new Module510(1, 32);
            model.call(torch.randn(16, 1, 32));

            var w0 = model.get_parameter("stack.0.weight").clone();
            var w1 = model.get_parameter("stack.1.weight").clone();
            var b1 = model.get_parameter("stack.1.bias").clone();
            var rm = model.get_buffer("stack.1.running_mean").clone();
            var rv = model.get_buffer("stack.1.running_var").clone();
            var nm = model.get_buffer("stack.1.num_batches_tracked").clone();

            model.load("bug510.dat");

            var w0_ = model.get_parameter("stack.0.weight");
            var w1_ = model.get_parameter("stack.1.weight");
            var b1_ = model.get_parameter("stack.1.bias");
            var rm_ = model.get_buffer("stack.1.running_mean");
            var rv_ = model.get_buffer("stack.1.running_var");
            var nm_ = model.get_buffer("stack.1.num_batches_tracked");

            Assert.NotEqual(w0, w0_);
            Assert.NotEqual(w1, w1_);
            Assert.NotEqual(b1, b1_);
            Assert.NotEqual(rm, rm_);
            Assert.NotEqual(rv, rv_);
            Assert.Equal(1, nm.item<long>());
            Assert.Equal(0, nm_.item<long>());
        }

        internal class Module510 : Module<Tensor, Tensor>
        {
            private readonly Module<Tensor, Tensor> stack;

            public Module510(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 0) : base(String.Empty)
            {
                var temp = BatchNorm1d(out_channels);
                this.stack = Sequential(
                    Conv1d(in_channels, out_channels, 3, stride: stride, padding: padding, bias: false),
                    temp,
                    ReLU(inplace: true)
                );

                temp.weight = Parameter(torch.randn(temp.weight!.shape));
                temp.bias = Parameter(torch.randn(temp.bias!.shape));
                if (temp.running_mean is not null) temp.running_mean = torch.randn(temp.running_mean.shape);
                if (temp.running_var is not null) temp.running_var = torch.randn(temp.running_var.shape);

                this.RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor t)
            {
                return this.stack.call(t);
            }
        }

        [Fact]
        public void ValidateIssue516()
        {
            if (torch.cuda.is_available()) {
                using var _ = NewDisposeScope();

                var model = new TestGradWarningModel();
                model.cuda();

                var optimizer = torch.optim.Adam(model.parameters());
                optimizer.zero_grad();

                var x = torch.ones(5, 3).cuda();
                var y = torch.ones(5, 4).cuda();

                var z = model.call(x);
                var lossFunc = torch.nn.CrossEntropyLoss();
                var loss = lossFunc.call(y, z);
                loss.backward();
                optimizer.step();

                var grad1 = optimizer.parameters().ToArray()[0].grad();
                Assert.NotNull(grad1);

                var grad2 = model.Weight.grad();
                Assert.NotNull(grad2);
            }
        }

        internal abstract class BaseModule : torch.nn.Module<Tensor, Tensor>
        {
            public int? InstanceId = null;

            protected BaseModule(string name) : base(name)
            {
            }
        }

        public class TestGradWarningModel : torch.nn.Module<Tensor, Tensor>
        {
            public readonly Modules.Parameter Weight;

            public TestGradWarningModel() : base(nameof(TestGradWarningModel))
            {
                Weight = torch.zeros(new long[] { 3, 4 }).AsParameter();
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor t)
            {
                return torch.matmul(t, Weight);
            }
        }

        [Fact]
        public void Validate532()
        {
            using var _ = NewDisposeScope();
            var module = new Module532(1000, 100);

            var p = module.parameters().ToArray();
            var pB = module.batch.parameters().ToArray();
            var pC = module.conv.parameters().ToArray();

            Assert.Equal(pB.Length + pC.Length, p.Length);
        }

        internal class Module532 : Module<Tensor, Tensor>
        {
            public Module<Tensor, Tensor> conv;
            public Module<Tensor, Tensor> batch;
            private Module<Tensor, Tensor> seq;

            public Module532(int in_channels, int out_channels) : base(String.Empty)
            {
                conv = Conv1d(in_channels, out_channels, 3);
                batch = BatchNorm1d(out_channels);
                seq = Sequential(
                    conv,
                    batch,
                    ReLU(inplace: true)
                );

                this.RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor t)
            {
                return this.seq.call(t);
            }
        }



        [Fact]
        public void Validate538()
        {
            var module = new Module538(1000, 100);

            var sd = module.state_dict();

            Assert.Equal(7, sd.Count);

            if (File.Exists("bug538.dat")) File.Delete("bug538.dat");

            module.save("bug538.dat");
            module.load("bug538.dat");

            File.Delete("bug538.dat");
        }

        internal class Module538 : Module<Tensor, Tensor>
        {
            private Module<Tensor, Tensor> seq;

            public Module538(int in_channels, int out_channels) : base(String.Empty)
            {
                seq = Sequential(Conv2d(1, 32, 3),
                     BatchNorm2d(32),
                     ReLU(),
                     Flatten(),
                     LogSoftmax(1)
                );

                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor t)
            {
                return this.seq.call(t);
            }
        }

        [Fact]
        public void GruModuleWorksWithoutPassingH0()
        {
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            using var _ = NewDisposeScope();
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }, device: device),
                   h0 = torch.randn(new long[] { 1, 3, 20 }, device: device))
            using (var gru = GRU(10, 20)) {
                gru.to(device);
                var (output, hN) = gru.call(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }
        }

        [Fact]
        public void LstmModuleWorksWithoutPassingH0C0()
        {
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            using var _ = NewDisposeScope();
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }, device: device),
                   h0 = torch.randn(new long[] { 1, 3, 20 }, device: device))
            using (var lstm = LSTM(10, 20)) {
                lstm.to(device);
                var (output, hN, hX) = lstm.call(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(h0.shape, hX.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }
        }

        [Fact]
        public void RnnModuleWorksWithoutPassingH0()
        {
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            using var _ = NewDisposeScope();
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }, device: device),
               h0 = torch.randn(new long[] { 1, 3, 20 }, device: device))
            using (var rnn = RNN(10, 20)) {
                rnn.to(device);
                var (output, hN) = rnn.call(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }
        }

        [Fact]
        public void ValidateBug618()
        {
            if (torch.cuda.is_available()) {
                using var _ = NewDisposeScope();
                var mean = torch.randn(new long[] { 32, 3, 3 }, device: torch.CUDA);
                var tmp = torch.randn(new long[] { 3, 3 }, device: torch.CUDA);
                var log_std = tmp.expand_as(mean);
                var std = exp(log_std);
                var dist = torch.distributions.Normal(mean, std);
                dist.sample();// error
            }
        }

        [Fact]
        public void ValidateBug653()
        {
            var x = torch.linspace(0, 1, 100, dtype: torch.float32);
            var y = torch.linspace(0, 1, 100, dtype: torch.float64);
            var z = x.to(y);
            Assert.Equal(torch.float64, z.dtype);
        }

        [Fact]
        public void ValidateBug679()
        {
            // Per bug report.
            // https://github.com/dotnet/TorchSharp/issues/679
            //
            var dtype = torch.float32;
            var spec = torch.rand(1, 1024, 500, 2, dtype: dtype);
            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() => torch.istft(spec, 512, 160, 400, null));

            spec = torch.rand(1, 512, 500, 2, dtype: dtype);
            var x = torch.istft(spec, 512, 160, 400, null);

            spec = torch.rand(1, 257, 500, 2, dtype: dtype);
            x = torch.istft(spec, 512, 160, 400, null);

            dtype = torch.complex64;
            spec = torch.rand(1, 1024, 500, dtype: dtype);
            Assert.Throws<System.Runtime.InteropServices.ExternalException>(() => torch.istft(spec, 512, 160, 400, null));

            spec = torch.rand(1, 512, 500, dtype: dtype);
            x = torch.istft(spec, 512, 160, 400, null);

            spec = torch.rand(1, 257, 500, dtype: dtype);
            x = torch.istft(spec, 512, 160, 400, null);
        }

        [Fact]
        public void ValidateBug715()
        {
            var resnet = resnet18();
            var resnetlist = resnet.named_children();
            var list = resnetlist.Take(6).Select(x => (x.name, (nn.Module<Tensor, Tensor>)x.module));
            var bone = nn.Sequential(list);

            var x = torch.zeros(1, 3, 64, 160);

            // This should not blow up.
            var tmp = bone.call(x);
        }

        [Fact]
        [TestOf(nameof(Modules.Categorical.probs))]
        public void ValidateBug836()
        {
            int nSamples = Convert.ToInt32(5e3);
            random.manual_seed(1);
            Tensor actionLogits = ones(nSamples, 2);
            Modules.Categorical distribution = new(logits: actionLogits);
            Tensor actions = distribution.sample();
            Tensor entropy = distribution.entropy();
            Tensor log_prob = distribution.log_prob(actions);
            var eMean = entropy.mean().ToSingle();
            var lMean = log_prob.mean().ToSingle();

            Assert.Equal(0.693147, eMean, 0.0001);
            Assert.Equal(-0.693147, lMean, 0.0001);
        }

        [Fact]
        [TestOf(nameof(torch.distributions.Bernoulli))]
        public void ValidateBug838()
        {
            int nSamples = Convert.ToInt32(5e6);
            random.manual_seed(1);
            Tensor actionLogits = rand(nSamples, 2);
            // This should not blow up.
            var distribution = distributions.Bernoulli(logits: actionLogits);
        }

        [Fact]
        public void ValidateMultiStepLR()
        {
            var gen = new Generator(4711);
            TestTraining.CreateLinearLayers(gen, out var lin1, out var lin2);
            TestTraining.CreateDataAndLabels(gen, out var x, out var y);

            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.1;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, new int[] { 3, 5, 7 }, 0.9);

            optimizer.zero_grad();
            optimizer.step();
            scheduler.step(1);
            Assert.Equal(0.1, optimizer.ParamGroups.First().LearningRate);
            scheduler.step(2);
            Assert.Equal(0.1, optimizer.ParamGroups.First().LearningRate);
            scheduler.step(3);
            Assert.Equal(0.09, optimizer.ParamGroups.First().LearningRate, 0.00001);
            scheduler.step(4);
            Assert.Equal(0.09, optimizer.ParamGroups.First().LearningRate, 0.00001);
            scheduler.step(5);
            Assert.Equal(0.09 * 0.9, optimizer.ParamGroups.First().LearningRate, 0.00001);
            scheduler.step(6);
            scheduler.step(7);
            Assert.Equal(0.09 * 0.9 * 0.9, optimizer.ParamGroups.First().LearningRate, 0.00001);
            scheduler.step(8);
            Assert.Equal(0.09 * 0.9 * 0.9, optimizer.ParamGroups.First().LearningRate, 0.00001);
            scheduler.step(9);
            Assert.Equal(0.09 * 0.9 * 0.9, optimizer.ParamGroups.First().LearningRate, 0.00001);
            scheduler.step(10);
            Assert.Equal(0.09 * 0.9 * 0.9, optimizer.ParamGroups.First().LearningRate, 0.00001);
        }



        [Fact]
        public void ValidatePolynomialLR()
        {
            {
                // Linear decay
                var gen = new Generator(4711);
                TestTraining.CreateLinearLayers(gen, out var lin1, out var lin2);
                TestTraining.CreateDataAndLabels(gen, out var x, out var y);

                var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

                double learning_rate = 0.1;
                var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
                var scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 10, 1);

                optimizer.zero_grad();
                optimizer.step();
                scheduler.step(1);
                Assert.Equal(0.09, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(2);
                Assert.Equal(0.08, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(3);
                Assert.Equal(0.07, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(4);
                Assert.Equal(0.06, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(5);
                Assert.Equal(0.05, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(6);
                scheduler.step(7);
                Assert.Equal(0.03, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(8);
                Assert.Equal(0.02, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(9);
                Assert.Equal(0.01, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(10);
                Assert.Equal(0.0, optimizer.ParamGroups.First().LearningRate, 0.00001);
            }
            {
                // Squared decay
                var gen = new Generator(4711);
                TestTraining.CreateLinearLayers(gen, out var lin1, out var lin2);
                TestTraining.CreateDataAndLabels(gen, out var x, out var y);

                var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

                double learning_rate = 0.1;
                var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
                var scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 10, 2);

                optimizer.zero_grad();
                optimizer.step();
                scheduler.step(1);
                Assert.Equal(0.081, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(2);
                Assert.Equal(0.064, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(3);
                Assert.Equal(0.049, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(4);
                Assert.Equal(0.036, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(5);
                Assert.Equal(0.025, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(6);
                scheduler.step(7);
                Assert.Equal(0.009, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(8);
                Assert.Equal(0.004, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(9);
                Assert.Equal(0.001, optimizer.ParamGroups.First().LearningRate, 0.00001);
                scheduler.step(10);
                Assert.Equal(0.0, optimizer.ParamGroups.First().LearningRate, 0.00001);
            }
        }


        [Fact]
        public void Validate845()
        {
            var module1 = new Module845(10, 10);
            var module2 = new Module845(10, 10);
            var module3 = new Module845(10, 10);

            var dev = torch.cuda.is_available() ? CUDA : CPU;

            module1.to(dev);
            module1.validate(float32, dev.type);

            module2.to(dev, float64);
            module2.validate(float64, dev.type);

            module3.to(float64);
            module3.validate(float64, CPU.type);
        }

        internal class Module845 : Module<Tensor, Tensor>
        {
            private Module<Tensor, Tensor> seq;

            public Module845(int in_channels, int out_channels) : base(String.Empty)
            {
                seq = Sequential(Conv2d(1, 32, 3),
                     ReLU(),
                     Flatten(),
                     LogSoftmax(1)
                );

                register_buffer("test", torch.ones(10, 10));
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor t)
            {
                return this.seq.call(t);
            }

            public void validate(ScalarType expected, DeviceType devType)
            {
                foreach (var (name, buffer) in named_buffers()) {
                    Assert.Equal(expected, buffer.dtype);
                    Assert.Equal(devType, buffer.device_type);
                }
            }
        }

        [Fact]
        public void Validate851()
        {
            var a = torch.tensor(new long[] { 100, 200, 300, 400 }, new long[] { 1, 4 });
            a.print();
            var str = a.ToString(TorchSharp.TensorStringStyle.Numpy);
            Assert.Equal("[[100 200 300 400]]", str);
        }

        [Fact]
        public void Validate852()
        {
            float[] a = new float[12];
            var x = torch.as_tensor(a);
        }

        [Fact]
        public void Validate877()
        {
            if (torch.cuda.is_available()) {
                var device = torch.CUDA;
                torch.TryInitializeDeviceType(device.type);
                var train = new GitBlockTest("test", device);

                var p0 = train.named_parameters().Select(p => p.name).ToArray();

                train.to(device);

                int named_parameters_1 = train.named_parameters().Count();
                train.to(torch.CUDA);
                int named_parameters_2 = train.named_parameters().Count();

                Assert.Equal(named_parameters_1, named_parameters_2);
            }
        }


        class GitTestCnn : Module<Tensor, Tensor>
        {
            private readonly TorchSharp.Modules.Sequential layers0;

            public GitTestCnn(string name, Device? device = null) : base(name)
            {
                var modules = new List<(string, Module<Tensor, Tensor>)>();
                modules.Add(($"{name}-conv2d-1", Conv2d(1, 4, kernelSize: (1L, 1L), stride: (1L, 1L), padding: (0L, 0L), paddingMode: PaddingModes.Replicate, bias: false)));
                layers0 = Sequential(modules);

                RegisterComponents();
            }

            public override Tensor forward(Tensor t)
            {
                var t1 = layers0.call(t).squeeze_(3);
                return t1;
            }
        }

        class GitTestGru : Module<Tensor, Tensor>
        {
            private TorchSharp.Modules.GRU layers1;
            private TorchSharp.Modules.GRU layers2;
            private Tensor init_h0;
            private Tensor init_h1;

            public GitTestGru(string name, Device? device = null) : base(name)
            {
                layers1 = nn.GRU(1, 4, batchFirst: true);
                layers2 = nn.GRU(4, 4, batchFirst: true);

                var state_size = new long[] { 1, 1, 4 };
                init_h0 = torch.nn.Parameter(torch.zeros(state_size, device: device));
                init_h1 = torch.nn.Parameter(torch.zeros(state_size, device: device));

                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var (rnn_output, states_h0) = layers1.call(input, init_h0);
                init_h0 = states_h0.detach_();
                var (_, states_h1) = layers2.call(rnn_output, init_h1);
                init_h1 = states_h1.detach_();
                var x2 = states_h1[-1];
                return x2;
            }
        }

        class GitBlockTest : Module<Tensor, Tensor>
        {
            private readonly Module<Tensor, Tensor> sequence_layers;

            public GitBlockTest(string name, Device? device = null) : base(name)
            {
                var modules = nn.ModuleDict<Module<Tensor, Tensor>>();
                modules.Add(("cnn-1", new GitTestCnn("GitTest", device)));
                modules.Add(("rnn-2", new GitTestGru("GitTestGru", device)));
                sequence_layers = Sequential(modules.values());
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var t0 = sequence_layers.call(input);
                return t0;
            }
        }


        [Fact]
        public void Validate912()
        {
            var test = new Test_912(2);
            Dictionary<string, Tensor> sd = test.state_dict(); // No layer2.modules keyword
            Assert.Contains(sd.Keys, k => k.StartsWith("layer2.modules"));
        }

        public class Test_912 : nn.Module
        {
            public nn.Module layer;
            public nn.Module layer2;

            public Test_912(int layernum) : base("Test_912")
            {
                layer = nn.Linear(16, 2);
                layer2 = new Test2_912((from l in Enumerable.Range(0, layernum)
                                    select new Test3_912()).ToArray(),
                                    (from l in Enumerable.Range(0, layernum)
                                     select new Test3_912()).ToArray());
                this.RegisterComponents();
            }
        }
        public class Test2_912 : nn.Module
        {
            public new Modules.ModuleList<Test3_912> modules;
            public Modules.ModuleList<nn.Module> modules2;
            public nn.Module layer;
            public Test2_912(Test3_912[] ms, nn.Module[] ms2) : base("Test2_912")
            {
                layer = nn.Linear(16, 2);
                modules = nn.ModuleList(ms);
                modules2 = nn.ModuleList(ms2);
                this.RegisterComponents();
            }
        }
        public class Test3_912 : nn.Module
        {
            public nn.Module layer;
            public Test3_912() : base("Test3_912")
            {
                layer = nn.Linear(16, 2);
                this.RegisterComponents();
            }
        }
    }
}