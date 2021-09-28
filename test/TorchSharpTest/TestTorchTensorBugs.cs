// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

using System.Threading;

using static TorchSharp.torch.nn;
using Xunit;

using static TorchSharp.torch;
using System.Runtime.CompilerServices;

#nullable enable

namespace TorchSharp
{
    // The tests in this file are all derived from reported GitHub Issues, serving
    // as regression tests.

    public class TestTorchTensorBugs
    {

        [Fact]
        public void ValidateAddInplace()
        {
            var x = torch.zeros(10,10);
            var y = torch.ones(10).expand(10, 10);

            x.add_(y, 1);

            Assert.Equal(x, y);
        }

        [Fact]
        public void ValidateIssue145()
        {
            // Tensor.DataItem gives a hard crash on GPU tensor

            if (torch.cuda.is_available()) {
                var scalar = torch.tensor(3.14f, torch.CUDA);
                Assert.Throws<InvalidOperationException>(() => scalar.DataItem<float>());
                var tensor = torch.zeros(new long[] { 10, 10 }, device: torch.CUDA);
                Assert.Throws<InvalidOperationException>(() => tensor.Data<float>());
                Assert.Throws<InvalidOperationException>(() => tensor.Bytes());
            }
        }

        class DoubleIt : nn.Module
        {
            public DoubleIt() : base("double") { }

            public override Tensor forward(Tensor t) => t * 2;
        }

        [Fact]
        public void ValidateIssue315_1()
        {
            // https://github.com/dotnet/TorchSharp/issues/315
            // custom module crash in GC thread

            // make Torch call our custom module by adding a ReLU in front of it
            using var net = nn.Sequential(
                ("relu", nn.ReLU()),
                ("double", new DoubleIt())
            );

            using var @in = torch.tensor(3);
            using var @out = net.forward(@in);
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        [Fact]
        public void ValidateIssue315_2()
        {
            Func<Tensor, Tensor, Tensor> distance =
                (x, y) => {
                    return (x - y).abs();
                };

            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requiresGrad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requiresGrad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var output = nn.functional.triplet_margin_with_distance_loss(distance);
                using (var result = output(anchor, positive, negative)) { }
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }



        [Fact]
        public void ValidateIssue315_3()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            using var x = torch.randn(new long[] { 64, 1000 });
            using var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate);
            var loss = nn.functional.mse_loss(Reduction.Sum);

            Func<Tensor> closure = () => {
                using var eval = seq.forward(x);
                var output = loss(eval, y);

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
            using var net = nn.Sequential(
                ("relu", nn.ReLU()),
                ("double", new DoubleIt())
            );

            using var @in = torch.tensor(3);

            for (var i = 0; i < 1000; i++) {
                using var @out = net.forward(@in);
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

        class TestModule : Module
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
                    t = seq.forward(t);
                }
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static Module Make() => Sequential(("t", new TestModule()), ("d", Linear(10, 10)));
        }

        [Fact]
        void ValidateIssue321()
        {
            TestModule.Reproduce();
        }

        [Fact]
        public void ValidateIssue353()
        {
            //
            // Just validating that the methods are there.
            //
            using var x = torch.zeros(3,3);
            using var y = torch.ones(3,3);

            var mx1 = x.max();
            Assert.Equal(0, mx1.DataItem<float>());

            var mx2 = x.maximum(y);
            Assert.True(mx2.allclose(y));

            var mn1 = x.min();
            Assert.Equal(0, mn1.DataItem<float>());

            var mn2 = x.minimum(y);
            Assert.True(mn2.allclose(x));
        }
    }
}