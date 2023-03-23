// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using Xunit;
using System.Collections.Generic;

#nullable enable

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestLoadSave
    {
        [Fact]
        public void TestSaveLoadLinear1()
        {
            var location = "TestSaveLoadLinear1.ts";

            if (File.Exists(location)) File.Delete(location);

            try {
                var linear = Linear(100, 10, true);
                var params0 = linear.parameters();
                linear.save(location);

                var loadedLinear = Linear(100, 10, true);
                loadedLinear.load(location);

                var params1 = loadedLinear.parameters();
                Assert.Equal(params0, params1);

                loadedLinear = Linear(100, 10, true);
                loadedLinear.load(location, skip: new[] { "weight" });
                var params2 = loadedLinear.parameters();

                Assert.NotEqual(params0.First(), params2.First());
                Assert.Equal(params0.Skip(1).First(), params2.Skip(1).First());
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadLinear2()
        {
            var location = "TestSaveLoadLinear2.ts";

            if (File.Exists(location)) File.Delete(location);
            try {
                var linear = Linear(100, 10, true);
                var params0 = linear.parameters();
                linear.save(location, skip: new[] { "weight" });

                var loadedLinear = Linear(100, 10, true);
                Assert.Throws<ArgumentException>(() => loadedLinear.load(location, strict: true));
                loadedLinear.load(location, strict: false);
                var params2 = loadedLinear.parameters();

                Assert.NotEqual(params0.First(), params2.First());
                Assert.Equal(params0.Skip(1).First(), params2.Skip(1).First());
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadLinear3()
        {
            var location = "TestSaveLoadLinear3.ts";
            if (File.Exists(location)) File.Delete(location);

            try {
                using var linear = Linear(100, 10, true);
                var params0 = linear.parameters();
                linear.save(location);

                var loadedLinear = Linear(10, 10, true);    // Mismatched shape, shouldn't matter when skipped.
                Assert.Throws<ArgumentException>(() => loadedLinear.load(location));

                loadedLinear.load(location, skip: new[] { "weight" });
                var params2 = loadedLinear.parameters();

                Assert.NotEqual(params0.First(), params2.First());
                Assert.Equal(params0.Skip(1).First(), params2.Skip(1).First());
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadConv2D()
        {
            var location = "TestSaveLoadConv2D.ts";
            if (File.Exists(location)) File.Delete(location);
            try {
                using var conv = Conv2d(100, 10, 5);
                var params0 = conv.parameters();
                conv.save(location);
                using var loaded = Conv2d(100, 10, 5);
                loaded.load(location);
                var params1 = loaded.parameters();

                Assert.Equal(params0, params1);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadConv2D_sd()
        {
            using var conv = Conv2d(100, 10, 5);
            var params0 = conv.parameters();

            var sd = conv.state_dict();

            using var loaded = Conv2d(100, 10, 5);
            Assert.NotEqual(params0, loaded.parameters());

            loaded.load_state_dict(sd);
            Assert.Equal(params0, loaded.parameters());
        }

        [Fact]
        public void TestSaveLoadConv2DStream()
        {
            var location = "TestSaveLoadConv2D.ts";
            if (File.Exists(location)) File.Delete(location);
            try {
                using var conv = Conv2d(100, 10, 5);
                var params0 = conv.parameters();

                using (var stream = System.IO.File.OpenWrite(location)) {
                    conv.save(stream);
                }

                using var loaded = Conv2d(100, 10, 5);

                using (var stream = System.IO.File.OpenRead(location)) {
                    loaded.load(stream);
                }
                var params1 = loaded.parameters();

                Assert.Equal(params0, params1);

            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadSequential()
        {
            var location = "TestSaveLoadSequential.ts";
            if (File.Exists(location)) File.Delete(location);
            try {
                using var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
                var params0 = conv.parameters();
                conv.save(location);
                using var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
                loaded.load(location);
                var params1 = loaded.parameters();

                Assert.Equal(params0, params1);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadSequentialStream()
        {
            var location = "TestSaveLoadSequential.ts";
            if (File.Exists(location)) File.Delete(location);
            try {
                using var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
                var params0 = conv.parameters();
                using (var stream = System.IO.File.OpenWrite(location)) {
                    conv.save(stream);
                }
                using var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
                using (var stream = System.IO.File.OpenRead(location)) {
                    loaded.load(stream);
                }
                var params1 = loaded.parameters();

                Assert.Equal(params0, params1);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadSequential_sd()
        {
            using var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            var params0 = conv.parameters();

            var sd = conv.state_dict();

            using var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            Assert.NotEqual(params0, loaded.parameters());

            loaded.load_state_dict(sd);
            Assert.Equal(params0, loaded.parameters());
        }

        [Fact]
        public void TestSaveLoadSequential_error1()
        {
            using var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            var params0 = conv.parameters();

            var sd = conv.state_dict();
            sd.Remove("0.bias");

            using var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            Assert.NotEqual(params0, loaded.parameters());

            Assert.Throws<InvalidOperationException>(() => loaded.load_state_dict(sd));
        }

        [Fact]
        public void TestSaveLoadSequential_error2()
        {
            using var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            var params0 = conv.parameters();

            var sd = conv.state_dict();
            var t = sd["0.bias"];

            sd.Add("2.bias", t);

            using var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            Assert.NotEqual(params0, loaded.parameters());

            Assert.Throws<InvalidOperationException>(() => loaded.load_state_dict(sd));
        }

        [Fact]
        public void TestSaveLoadSequential_lax()
        {
            using var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            var params0 = conv.parameters();

            var sd = conv.state_dict();
            var t = sd["0.bias"];
            sd.Remove("0.bias");

            sd.Add("2.bias", t);

            using var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            Assert.NotEqual(params0, loaded.parameters());

            var (m, u) = loaded.load_state_dict(sd, false);
            Assert.NotEqual(params0, loaded.parameters());

            Assert.NotEmpty(m);
            Assert.NotEmpty(u);
        }

        [Fact]
        public void TestSaveLoadCustomWithParameters()
        {
            var location = "TestSaveLoadCustomWithParameters.ts";
            if (File.Exists(location)) File.Delete(location);

            try {
                using var original = new TestModule1();
                Assert.True(original.has_parameter("test"));

                var params0 = original.parameters();
                Assert.True(params0.ToArray().ToArray()[0].requires_grad);
                original.save(location);

                using var loaded = new TestModule1();
                Assert.True(loaded.has_parameter("test"));

                var params1 = loaded.parameters();
                Assert.True(params1.ToArray()[0].requires_grad);
                Assert.NotEqual(params0.ToArray(), params1);

                loaded.load(location);
                var params2 = loaded.parameters();
                Assert.True(params2.ToArray()[0].requires_grad);
                Assert.Equal(params0, params2);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        private class TestModule1 : Module<torch.Tensor, torch.Tensor>
        {
            public TestModule1() : base("TestModule1")
            {
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor input)
            {
                throw new NotImplementedException();
            }

            private Parameter test = Parameter(torch.randn(new long[] { 2, 2 }));
        }


        [Fact]
        public void TestSaveLoadError_1()
        {
            var location = "TestSaveLoadError_1.ts";
            if (File.Exists(location)) File.Delete(location);

            try {
                var linear = Linear(100, 10, true);
                var params0 = linear.parameters();
                linear.save(location);
                var loaded = Conv2d(100, 10, 5);
                Assert.Throws<ArgumentException>(() => loaded.load(location));
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadError_2()
        {
            var location = "TestSaveLoadError_2.ts";

            // Submodule count mismatch
            if (File.Exists(location)) File.Delete(location);
            try {
                var linear = Sequential(("linear1", Linear(100, 10, true)));
                var params0 = linear.parameters();
                linear.save(location);
                var loaded = Sequential(("linear1", Linear(100, 10, true)), ("conv1", Conv2d(100, 10, 5)));
                Assert.Throws<ArgumentException>(() => loaded.load(location));
                // Shouldn't get an error for this:
                loaded.load(location, strict: false);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadError_3()
        {
            var location = "TestSaveLoadError_3.ts";
            // Submodule name mismatch
            if (File.Exists(location)) File.Delete(location);
            try {
                var linear = Sequential(("linear1", Linear(100, 10, true)));
                var params0 = linear.parameters();
                linear.save(location);
                var loaded = Sequential(("linear2", Linear(100, 10, true)));
                Assert.Throws<ArgumentException>(() => loaded.load(location));
                File.Delete(location);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadSequence()
        {
            if (File.Exists(".model-list.txt")) File.Delete(".model-list.txt");

            var lin1 = Linear(100, 10, true);
            var lin2 = Linear(10, 5, true);
            var seq = Sequential(("lin1", lin1), ("lin2", lin2));
            var params0 = seq.parameters();
            seq.save(".model-list.txt");

            var lin3 = Linear(100, 10, true);
            var lin4 = Linear(10, 5, true);
            var seq1 = Sequential(("lin1", lin3), ("lin2", lin4));
            seq1.load(".model-list.txt");
            File.Delete("model-list.txt");

            var params1 = seq1.parameters();
            Assert.Equal(params0, params1);
        }

        [Fact]
        public void TestSaveLoadGruOnCPU()
        {
            var location = "TestSaveLoadGruOnCPU.ts";
            // Works on CPU
            if (File.Exists(location)) File.Delete(location);
            try {
                var gru = GRU(2, 2, 2);
                var params0 = gru.parameters();
                gru.save(location);

                var loadedGru = GRU(2, 2, 2);
                loadedGru.load(location);
                var params1 = loadedGru.parameters();
                File.Delete(location);
                Assert.Equal(params0, params1);
            } finally {
                if (File.Exists(location)) File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveLoadGruOnCUDA()
        {
            var location = "TestSaveLoadGruOnCUDA.ts";
            if (torch.cuda.is_available()) {

                // Fails on CUDA
                if (File.Exists(location)) File.Delete(location);
                try {
                    var gru = GRU(2, 2, 2);
                    gru.to(DeviceType.CUDA);
                    var params0 = gru.parameters().ToArray();
                    Assert.Equal(DeviceType.CUDA, params0[0].device_type);

                    gru.save(location);

                    // Make sure the model is still on the GPU when we come back.

                    params0 = gru.parameters().ToArray();
                    Assert.Equal(DeviceType.CUDA, params0[0].device_type);

                    var loadedGru = GRU(2, 2, 2);
                    loadedGru.to(DeviceType.CUDA);
                    loadedGru.load(location);
                    var params1 = loadedGru.parameters().ToArray();

                    Assert.Equal(DeviceType.CUDA, params1[0].device_type);

                    File.Delete(location);
                    Assert.Equal(params0, params1);
                } finally {
                    if (File.Exists(location)) File.Delete(location);
                }

            }
        }

        [Fact]
        public void TestSummaryWriterLogDir()
        {
            var w1 = torch.utils.tensorboard.SummaryWriter("runs/123");
            Assert.Equal("runs/123", w1.LogDir);
            if (Directory.Exists(w1.LogDir)) {
                Directory.Delete(w1.LogDir, recursive: true);
            }

            var w2 = torch.utils.tensorboard.SummaryWriter("garbage", createRunName: true);
            Assert.NotEqual("garbage", w2.LogDir);
            Assert.StartsWith("garbage", w2.LogDir);
            if (Directory.Exists(w2.LogDir)) {
                Directory.Delete(w2.LogDir, recursive: true);
            }
        }

        [Fact]
        public void TestTensorBoardScalar()
        {
            var writer = torch.utils.tensorboard.SummaryWriter();
            Assert.StartsWith("runs", writer.LogDir);

            for (var i = 0; i < 100; i++) {
                writer.add_scalar("a/b", MathF.Sin(i * MathF.PI / 8), i);
            }

            // Comment this out to look at the output data in tensorboard
            if (Directory.Exists(writer.LogDir)) {
                Directory.Delete(writer.LogDir, recursive: true);
            }
        }

        [Fact]
        public void TestTensorBoardScalars1()
        {
            var writer = torch.utils.tensorboard.SummaryWriter();
            for (var i = 0; i < 100; i++) {
                float f = i;
                writer.add_scalars("run_14h", new Dictionary<string, float> {
                    { "sin",i* MathF.Sin(f / 5) },
                    { "cos", i* MathF.Cos(f / 5) },
                    { "tan", MathF.Tan(f / 5) }
                }, i);
            }

            // Comment this out to look at the output data in tensorboard
            if (Directory.Exists(writer.LogDir)) {
                Directory.Delete(writer.LogDir, recursive: true);
            }
        }

        [Fact]
        public void TestTensorBoardScalars2()
        {
            var writer = torch.utils.tensorboard.SummaryWriter();
            for (var i = 0; i < 100; i++) {
                float f = i;
                writer.add_scalars("run_14h", new[] {
                    ("sin",i* MathF.Sin(f / 5)),
                    ("cos", i* MathF.Cos(f / 5)),
                    ("tan", MathF.Tan(f / 5))
                }, i);
            }

            // Comment this out to look at the output data in tensorboard
            if (Directory.Exists(writer.LogDir)) {
                Directory.Delete(writer.LogDir, recursive: true);
            }
        }

        [Fact]
        public void TestSaveRprop()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Rprop(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Rprop(seq.parameters(), 0.5, 0.01, 1.25, 1e-8, 35);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.Rprop.Options;
                var opt2 = sd2.Options[i] as Modules.Rprop.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.etaminus, opt2!.etaminus),
                () => Assert.Equal(opt!.etaplus, opt2!.etaplus),
                () => Assert.Equal(opt!.min_step, opt2!.min_step),
                () => Assert.Equal(opt!.max_step, opt2!.max_step)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.Rprop.State;
                var opt2 = sd2.State[i] as Modules.Rprop.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveRpropFile()
        {
            var location = ".TestSaveRpropFile1.ts";

            if (File.Exists(location)) File.Delete(location);
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Rprop(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Rprop(seq.parameters(), 0.5, 0.01, 1.25, 1e-8, 35);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Rprop.Options;
                    var opt2 = sd2.Options[i] as Modules.Rprop.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.etaminus, opt2!.etaminus),
                    () => Assert.Equal(opt!.etaplus, opt2!.etaplus),
                    () => Assert.Equal(opt!.min_step, opt2!.min_step),
                    () => Assert.Equal(opt!.max_step, opt2!.max_step)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Rprop.State;
                    var opt2 = sd2.State[i] as Modules.Rprop.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveSGD()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.SGD(seq.parameters(), 0.01);

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.SGD(seq.parameters(), 0.5, 0.01, 1.25, 1e-8);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.SGD.Options;
                var opt2 = sd2.Options[i] as Modules.SGD.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.momentum, opt2!.momentum),
                () => Assert.Equal(opt!.dampening, opt2!.dampening),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.SGD.State;
                var opt2 = sd2.State[i] as Modules.SGD.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveSGDFile()
        {
            var location = ".TestSaveSGDFile.ts";

            if (File.Exists(location)) File.Delete(location);
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.SGD(seq.parameters(), 0.01);

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.SGD(seq.parameters(), 0.5, 0.01, 1.25, 1e-8);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.SGD.Options;
                    var opt2 = sd2.Options[i] as Modules.SGD.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.momentum, opt2!.momentum),
                    () => Assert.Equal(opt!.dampening, opt2!.dampening),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.SGD.State;
                    var opt2 = sd2.State[i] as Modules.SGD.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveSGDFile2()
        {
            var location = ".TestSaveSGDFile2.ts";

            if (File.Exists(location)) File.Delete(location);
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.SGD(seq.parameters(), 0.01);

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.SGD(seq.parameters(), 0.5, 0.01, 1.25, 1e-8);

            // Force the SGD momentum_buffer to be created.

            using var x = torch.randn(new long[] { 64, 10 });
            using var y = torch.randn(new long[] { 64, 10 });

            var loss = torch.nn.MSELoss(Reduction.Sum);

            using var eval = seq.call(x);
            var output = loss.call(eval, y);

            var l = output.ToSingle();

            optim2.zero_grad();

            output.backward();

            optim2.step();

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.SGD.Options;
                    var opt2 = sd2.Options[i] as Modules.SGD.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.momentum, opt2!.momentum),
                    () => Assert.Equal(opt!.dampening, opt2!.dampening),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.SGD.State;
                    var opt2 = sd2.State[i] as Modules.SGD.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveSGDFile3()
        {
            var location = ".TestSaveSGDFile3.ts";

            if (File.Exists(location)) File.Delete(location);
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.SGD(seq.parameters(), 0.01, 0.025);

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.SGD(seq.parameters(), 0.5, 0.01, 1.25, 1e-8);

            // Force the SGD momentum_buffer to be created.

            using var x = torch.randn(new long[] { 64, 10 });
            using var y = torch.randn(new long[] { 64, 10 });

            var loss = torch.nn.MSELoss(Reduction.Sum);

            using var eval = seq.call(x);
            var output = loss.call(eval, y);

            var l = output.ToSingle();

            optim.zero_grad();

            output.backward();

            optim.step();

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.SGD.Options;
                    var opt2 = sd2.Options[i] as Modules.SGD.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.momentum, opt2!.momentum),
                    () => Assert.Equal(opt!.dampening, opt2!.dampening),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.SGD.State;
                    var opt2 = sd2.State[i] as Modules.SGD.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveASGD()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.ASGD(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.ASGD(seq.parameters(), 0.5, 0.01, 1.25, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.ASGD.Options;
                var opt2 = sd2.Options[i] as Modules.ASGD.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.lambd, opt2!.lambd),
                () => Assert.Equal(opt!.alpha, opt2!.alpha),
                () => Assert.Equal(opt!.t0, opt2!.t0),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.ASGD.State;
                var opt2 = sd2.State[i] as Modules.ASGD.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveASGDFile()
        {
            var location = ".TestSaveASGDFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.ASGD(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.ASGD(seq.parameters(), 0.5, 0.01, 1.25, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.ASGD.Options;
                    var opt2 = sd2.Options[i] as Modules.ASGD.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.lambd, opt2!.lambd),
                    () => Assert.Equal(opt!.alpha, opt2!.alpha),
                    () => Assert.Equal(opt!.t0, opt2!.t0),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.ASGD.State;
                    var opt2 = sd2.State[i] as Modules.ASGD.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveRMSProp()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.RMSProp(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.RMSProp(seq.parameters(), 0.5, 0.01, 1.25, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.RMSProp.Options;
                var opt2 = sd2.Options[i] as Modules.RMSProp.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.momentum, opt2!.momentum),
                () => Assert.Equal(opt!.alpha, opt2!.alpha),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.RMSProp.State;
                var opt2 = sd2.State[i] as Modules.RMSProp.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveRMSPropFile()
        {
            var location = ".TestSaveRMSPropFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.RMSProp(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.RMSProp(seq.parameters(), 0.5, 0.01, 1.25, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.RMSProp.Options;
                    var opt2 = sd2.Options[i] as Modules.RMSProp.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.momentum, opt2!.momentum),
                    () => Assert.Equal(opt!.alpha, opt2!.alpha),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.RMSProp.State;
                    var opt2 = sd2.State[i] as Modules.RMSProp.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveRAdam()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.RAdam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.RAdam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.RAdam.Options;
                var opt2 = sd2.Options[i] as Modules.RAdam.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.beta1, opt2!.beta1),
                () => Assert.Equal(opt!.beta2, opt2!.beta2),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.RAdam.State;
                var opt2 = sd2.State[i] as Modules.RAdam.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveRAdamFile()
        {
            var location = ".TestSaveRAdamFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.RAdam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.RAdam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.RAdam.Options;
                    var opt2 = sd2.Options[i] as Modules.RAdam.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.RAdam.State;
                    var opt2 = sd2.State[i] as Modules.RAdam.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveNAdam()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.NAdam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.NAdam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.NAdam.Options;
                var opt2 = sd2.Options[i] as Modules.NAdam.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.beta1, opt2!.beta1),
                () => Assert.Equal(opt!.beta2, opt2!.beta2),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                () => Assert.Equal(opt!.momentum_decay, opt2!.momentum_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.NAdam.State;
                var opt2 = sd2.State[i] as Modules.NAdam.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveNAdamFile()
        {
            var location = ".TestSaveNAdamFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.NAdam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.NAdam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.NAdam.Options;
                    var opt2 = sd2.Options[i] as Modules.NAdam.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.momentum_decay, opt2!.momentum_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.NAdam.State;
                    var opt2 = sd2.State[i] as Modules.NAdam.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdam()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.Adam.Options;
                var opt2 = sd2.Options[i] as Modules.Adam.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.beta1, opt2!.beta1),
                () => Assert.Equal(opt!.beta2, opt2!.beta2),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                () => Assert.Equal(opt!.maximize, opt2!.maximize)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.Adam.State;
                var opt2 = sd2.State[i] as Modules.Adam.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveAdamFile()
        {
            var location = ".TestSaveAdamFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Adam.Options;
                    var opt2 = sd2.Options[i] as Modules.Adam.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                    () => Assert.Equal(opt!.maximize, opt2!.maximize)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Adam.State;
                    var opt2 = sd2.State[i] as Modules.Adam.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdamFile2()
        {
            var location = ".TestSaveAdamFile2.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adam(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25, true);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Adam.Options;
                    var opt2 = sd2.Options[i] as Modules.Adam.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                    () => Assert.Equal(opt!.maximize, opt2!.maximize)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Adam.State;
                    var opt2 = sd2.State[i] as Modules.Adam.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdamFile3()
        {
            var location = ".TestSaveAdamFile3.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adam(seq.parameters(), amsgrad: true);

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adam(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Adam.Options;
                    var opt2 = sd2.Options[i] as Modules.Adam.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                    () => Assert.Equal(opt!.maximize, opt2!.maximize)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Adam.State;
                    var opt2 = sd2.State[i] as Modules.Adam.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdamW()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.AdamW(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.AdamW(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.AdamW.Options;
                var opt2 = sd2.Options[i] as Modules.AdamW.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.beta1, opt2!.beta1),
                () => Assert.Equal(opt!.beta2, opt2!.beta2),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                () => Assert.Equal(opt!.maximize, opt2!.maximize)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.AdamW.State;
                var opt2 = sd2.State[i] as Modules.AdamW.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveAdamWFile()
        {
            var location = ".TestSaveAdamWFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.AdamW(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.AdamW(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.AdamW.Options;
                    var opt2 = sd2.Options[i] as Modules.AdamW.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                    () => Assert.Equal(opt!.maximize, opt2!.maximize)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.AdamW.State;
                    var opt2 = sd2.State[i] as Modules.AdamW.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdamWFile2()
        {
            var location = ".TestSaveAdamWFile2.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.AdamW(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.AdamW(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25, true);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.AdamW.Options;
                    var opt2 = sd2.Options[i] as Modules.AdamW.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                    () => Assert.Equal(opt!.maximize, opt2!.maximize)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.AdamW.State;
                    var opt2 = sd2.State[i] as Modules.AdamW.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdamWFile3()
        {
            var location = ".TestSaveAdamWFile3.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.AdamW(seq.parameters(), amsgrad: true);

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.AdamW(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.AdamW.Options;
                    var opt2 = sd2.Options[i] as Modules.AdamW.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay),
                    () => Assert.Equal(opt!.amsgrad, opt2!.amsgrad),
                    () => Assert.Equal(opt!.maximize, opt2!.maximize)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.AdamW.State;
                    var opt2 = sd2.State[i] as Modules.AdamW.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdamax()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adamax(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adamax(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.Adamax.Options;
                var opt2 = sd2.Options[i] as Modules.Adamax.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.beta1, opt2!.beta1),
                () => Assert.Equal(opt!.beta2, opt2!.beta2),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.Adamax.State;
                var opt2 = sd2.State[i] as Modules.Adamax.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveAdamaxFile()
        {
            var location = ".TestSaveAdamaxFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adamax(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adamax(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Adamax.Options;
                    var opt2 = sd2.Options[i] as Modules.Adamax.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.beta1, opt2!.beta1),
                    () => Assert.Equal(opt!.beta2, opt2!.beta2),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Adamax.State;
                    var opt2 = sd2.State[i] as Modules.Adamax.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdagrad()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adagrad(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adagrad(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.Adagrad.Options;
                var opt2 = sd2.Options[i] as Modules.Adagrad.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.lr_decay, opt2!.lr_decay),
                () => Assert.Equal(opt!.initial_accumulator_value, opt2!.initial_accumulator_value),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.Adagrad.State;
                var opt2 = sd2.State[i] as Modules.Adagrad.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveAdagradFile()
        {
            var location = ".TestSaveAdagradFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adagrad(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adagrad(seq.parameters(), 0.5, 0.01, 0.75, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Adagrad.Options;
                    var opt2 = sd2.Options[i] as Modules.Adagrad.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.lr_decay, opt2!.lr_decay),
                    () => Assert.Equal(opt!.initial_accumulator_value, opt2!.initial_accumulator_value),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Adagrad.State;
                    var opt2 = sd2.State[i] as Modules.Adagrad.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestSaveAdadelta()
        {
            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adadelta(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adadelta(seq.parameters(), 0.5, 0.01, 1e-8, 0.25);
            optim2.load_state_dict(sd);

            var sd2 = optim2.state_dict();

            Assert.Multiple(
            () => Assert.Equal(4, sd2.State.Count),
            () => Assert.Single(sd2.Options)
            );

            for (int i = 0; i < sd.Options.Count; i++) {
                var opt = sd.Options[i] as Modules.Adadelta.Options;
                var opt2 = sd2.Options[i] as Modules.Adadelta.Options;
                Assert.Multiple(
                () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                () => Assert.Equal(opt!.rho, opt2!.rho),
                () => Assert.Equal(opt!.eps, opt2!.eps),
                () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                );
            }
            for (int i = 0; i < sd.State.Count; i++) {
                var opt = sd.State[i] as Modules.Adadelta.State;
                var opt2 = sd2.State[i] as Modules.Adadelta.State;
                Assert.True(opt!.ApproximatelyEquals(opt2));
            }
        }

        [Fact]
        public void TestSaveAdadeltaFile()
        {
            var location = ".TestSaveAdadeltaFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim = torch.optim.Adadelta(seq.parameters());

            var sd = optim.state_dict();
            Assert.Multiple(
            () => Assert.Equal(4, sd.State.Count),
            () => Assert.Single(sd.Options)
            );

            var optim2 = torch.optim.Adadelta(seq.parameters(), 0.5, 0.01, 1e-8, 0.25);

            try {
                optim.save_state_dict(location);
                optim2.load_state_dict(location);

                var sd2 = optim2.state_dict();

                Assert.Multiple(
                () => Assert.Equal(4, sd2.State.Count),
                () => Assert.Single(sd2.Options)
                );

                for (int i = 0; i < sd.Options.Count; i++) {
                    var opt = sd.Options[i] as Modules.Adadelta.Options;
                    var opt2 = sd2.Options[i] as Modules.Adadelta.Options;
                    Assert.Multiple(
                    () => Assert.Equal(opt!.InitialLearningRate, opt2!.InitialLearningRate),
                    () => Assert.Equal(opt!.LearningRate, opt2!.LearningRate),
                    () => Assert.Equal(opt!.rho, opt2!.rho),
                    () => Assert.Equal(opt!.eps, opt2!.eps),
                    () => Assert.Equal(opt!.weight_decay, opt2!.weight_decay)
                    );
                }
                for (int i = 0; i < sd.State.Count; i++) {
                    var opt = sd.State[i] as Modules.Adadelta.State;
                    var opt2 = sd2.State[i] as Modules.Adadelta.State;
                    Assert.True(opt!.ApproximatelyEquals(opt2));
                }
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestMisMatchedFile()
        {

            var location = ".TestMisMatchedFile.ts";
            if (File.Exists(location)) File.Delete(location);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            var optim1 = torch.optim.SGD(seq.parameters(), 0.05);

            try {
                optim1.save_state_dict(location);
                Assert.Throws<InvalidDataException>(() => torch.optim.ASGD(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.Rprop(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.RMSProp(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.RAdam(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.NAdam(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.Adam(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.AdamW(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.Adamax(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.Adagrad(seq.parameters()).load_state_dict(location));
                Assert.Throws<InvalidDataException>(() => torch.optim.Adadelta(seq.parameters()).load_state_dict(location));
            } finally {
                File.Delete(location);
            }
        }

        [Fact]
        public void TestLoadingSGDStateFromPython()
        {
            var lin = torch.nn.Linear(10, 10);

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(lin.parameters(), learning_rate);

            optimizer.load_state_dict("sgd1.dat");

            var sd = optimizer.state_dict();

            Assert.Single(sd.Options);
            Assert.Equal(2, sd.State.Count);

            foreach (var opts in sd.Options) {
                var options = opts as Modules.SGD.Options;
                Assert.Equal(0.1, options!.momentum);
                Assert.NotEqual(learning_rate, options!.LearningRate);
            }

            foreach (var st in sd.State) {
                var state = st as Modules.SGD.State;
                Assert.NotNull(state!.momentum_buffer);
            }
        }

        [Fact]
        public void TestLoadingASGDStateFromPython()
        {
            var lin = torch.nn.Linear(10, 10);

            double learning_rate = 0.004f;
            var optimizer = torch.optim.ASGD(lin.parameters(), learning_rate);

            optimizer.load_state_dict("asgd1.dat");

            var sd = optimizer.state_dict();

            Assert.Single(sd.Options);
            Assert.Equal(2, sd.State.Count);

            foreach (var opts in sd.Options) {
                var options = opts as Modules.ASGD.Options;
                Assert.Equal(0.65, options!.alpha);
                Assert.Equal(1e-3, options!.lambd);
                Assert.Equal(1e5, options!.t0);
                Assert.NotEqual(learning_rate, options!.LearningRate);
            }

            foreach (var st in sd.State) {
                var state = st as Modules.ASGD.State;
                Assert.Equal(1, state!.step);
                Assert.NotNull(state!.ax);
            }
        }

        [Fact]
        public void TestLoadingRMSpropStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new RMSProp.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, centered: false, momentum: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(pgs, learning_rate);

            optimizer.load_state_dict("rmsprop1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.RMSProp.Options;
            Assert.Multiple(
                () => Assert.Equal(0.1, options!.momentum),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.False(options!.centered)
            );

            options = sd.Options[1] as Modules.RMSProp.Options;
            Assert.Multiple(
                () => Assert.Equal(0, options!.momentum),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.True(options!.centered)
            );

            var state = sd.State[0] as Modules.RMSProp.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.square_avg),
                () => Assert.NotNull(state!.momentum_buffer),
                () => Assert.Null(state!.grad_avg)
            );

            state = sd.State[1] as Modules.RMSProp.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.square_avg),
                () => Assert.NotNull(state!.momentum_buffer),
                () => Assert.Null(state!.grad_avg)
            );

            state = sd.State[2] as Modules.RMSProp.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.square_avg),
                () => Assert.Null(state!.momentum_buffer),
                () => Assert.NotNull(state!.grad_avg)
            );

            state = sd.State[3] as Modules.RMSProp.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.square_avg),
                () => Assert.Null(state!.momentum_buffer),
                () => Assert.NotNull(state!.grad_avg)
            );
        }

        [Fact]
        public void TestLoadingRpropStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new Rprop.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, maximize: false)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Rprop(pgs, learning_rate);

            optimizer.load_state_dict("rprop1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.Rprop.Options;
            Assert.Multiple(
                () => Assert.Equal(0.35, options!.etaminus),
                () => Assert.Equal(1.5, options!.etaplus),
                () => Assert.Equal(1e-5, options!.min_step),
                () => Assert.Equal(5, options!.max_step),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.False(options!.maximize)
            );

            options = sd.Options[1] as Modules.Rprop.Options;
            Assert.Multiple(
                () => Assert.Equal(0.45, options!.etaminus),
                () => Assert.Equal(1.5, options!.etaplus),
                () => Assert.Equal(1e-5, options!.min_step),
                () => Assert.Equal(5, options!.max_step),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.True(options!.maximize)
            );

            foreach (var st in sd.State) {
                var state = sd.State[0] as Modules.Rprop.State;
                Assert.Multiple(
                    () => Assert.Equal(1, state!.step),
                    () => Assert.NotNull(state!.prev),
                    () => Assert.NotNull(state!.step_size)
                );
            }
        }

        [Fact]
        public void TestLoadingAdamStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new Adam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, amsgrad: false, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adam(pgs, learning_rate);

            optimizer.load_state_dict("adam1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.Adam.Options;
            Assert.Multiple(
                () => Assert.Equal(0.8, options!.beta1),
                () => Assert.Equal(0.9, options!.beta2),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.False(options!.amsgrad)
            );

            options = sd.Options[1] as Modules.Adam.Options;
            Assert.Multiple(
                () => Assert.Equal(0.7, options!.beta1),
                () => Assert.Equal(0.79, options!.beta2),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.True(options!.amsgrad)
            );

            var state = sd.State[0] as Modules.Adam.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.Null(state!.max_exp_avg_sq)
            );

            state = sd.State[1] as Modules.Adam.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.Null(state!.max_exp_avg_sq)
            );

            state = sd.State[2] as Modules.Adam.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.NotNull(state!.max_exp_avg_sq)
            );

            state = sd.State[3] as Modules.Adam.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.NotNull(state!.max_exp_avg_sq)
            );
        }

        [Fact]
        public void TestLoadingAdamWStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new AdamW.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, amsgrad: false, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.AdamW(pgs, learning_rate);

            optimizer.load_state_dict("adamw1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.AdamW.Options;
            Assert.Multiple(
                () => Assert.Equal(0.8, options!.beta1),
                () => Assert.Equal(0.9, options!.beta2),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.False(options!.amsgrad)
            );

            options = sd.Options[1] as Modules.AdamW.Options;
            Assert.Multiple(
                () => Assert.Equal(0.7, options!.beta1),
                () => Assert.Equal(0.79, options!.beta2),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.True(options!.amsgrad)
            );

            var state = sd.State[0] as Modules.AdamW.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.Null(state!.max_exp_avg_sq)
            );

            state = sd.State[1] as Modules.AdamW.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.Null(state!.max_exp_avg_sq)
            );

            state = sd.State[2] as Modules.AdamW.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.NotNull(state!.max_exp_avg_sq)
            );

            state = sd.State[3] as Modules.AdamW.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg_sq),
                () => Assert.NotNull(state!.max_exp_avg_sq)
            );
        }

        [Fact]
        public void TestLoadingNAdamStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new NAdam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, beta1: 0.25, weight_decay: 0.1)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.NAdam(pgs, learning_rate);

            optimizer.load_state_dict("nadam1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.NAdam.Options;
            Assert.Multiple(
                () => Assert.Equal(0.8, options!.beta1),
                () => Assert.Equal(0.9, options!.beta2),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.Equal(0, options!.weight_decay)
            );

            options = sd.Options[1] as Modules.NAdam.Options;
            Assert.Multiple(
                () => Assert.Equal(0.7, options!.beta1),
                () => Assert.Equal(0.79, options!.beta2),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.Equal(0.3, options!.weight_decay)
            );

            foreach (var st in sd.State) {
                var state = st as Modules.NAdam.State;
                Assert.Multiple(
                    () => Assert.Equal(1, state!.step),
                    () => Assert.NotNull(state!.exp_avg),
                    () => Assert.NotNull(state!.exp_avg_sq)
                );
            }
        }

        [Fact]
        public void TestLoadingRAdamStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new RAdam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RAdam(pgs, learning_rate);

            optimizer.load_state_dict("radam1.dat");

            var sd = optimizer.state_dict();

            var options = sd.Options[0] as Modules.RAdam.Options;
            Assert.Multiple(
                () => Assert.Equal(0.8, options!.beta1),
                () => Assert.Equal(0.9, options!.beta2),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.Equal(0, options!.weight_decay)
            );

            options = sd.Options[1] as Modules.RAdam.Options;
            Assert.Multiple(
                () => Assert.Equal(0.7, options!.beta1),
                () => Assert.Equal(0.79, options!.beta2),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.Equal(0.3, options!.weight_decay)
            );

            foreach (var st in sd.State) {
                var state = st as Modules.RAdam.State;
                Assert.Multiple(
                    () => Assert.Equal(1, state!.step),
                    () => Assert.NotNull(state!.exp_avg),
                    () => Assert.NotNull(state!.exp_avg_sq)
                );
            }
        }

        [Fact]
        public void TestLoadingAdadeltaStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new Adadelta.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, maximize: false, rho: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adadelta(pgs, learning_rate);

            optimizer.load_state_dict("adadelta1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.Adadelta.Options;
            Assert.Multiple(
                () => Assert.Equal(0.85, options!.rho),
                () => Assert.Equal(0.3, options!.weight_decay),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.False(options!.maximize)
            );

            options = sd.Options[1] as Modules.Adadelta.Options;
            Assert.Multiple(
                () => Assert.Equal(0.79, options!.rho),
                () => Assert.Equal(0.3, options!.weight_decay),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.True(options!.maximize)
            );

            foreach (var st in sd.State) {
                var state = st as Modules.Adadelta.State;
                Assert.Multiple(
                    () => Assert.Equal(1, state!.step),
                    () => Assert.NotNull(state!.square_avg),
                    () => Assert.NotNull(state!.acc_delta)
               );
            }
        }

        [Fact]
        public void TestLoadingAdagradStateFromPython()
        {
            var lin = torch.nn.Linear(10, 10);

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adagrad(lin.parameters(), learning_rate);

            optimizer.load_state_dict("adagrad1.dat");

            var sd = optimizer.state_dict();

            Assert.Single(sd.Options);
            Assert.Equal(2, sd.State.Count);

            foreach (var opts in sd.Options) {
                var options = opts as Modules.Adagrad.Options;
                Assert.Equal(0.85, options!.lr_decay);
                Assert.Equal(0.3, options!.weight_decay);
                Assert.NotEqual(learning_rate, options!.LearningRate);
            }

            foreach (var st in sd.State) {
                var state = st as Modules.Adagrad.State;
                Assert.Equal(1, state!.step);
                Assert.NotNull(state!.sum);
            }
        }

        [Fact]
        public void TestLoadingAdamaxStateFromPython()
        {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new Adamax.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, weight_decay: 0.25, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adamax(pgs, learning_rate);

            optimizer.load_state_dict("adamax1.dat");

            var sd = optimizer.state_dict();

            Assert.Equal(2, sd.Options.Count);
            Assert.Equal(4, sd.State.Count);

            var options = sd.Options[0] as Modules.Adamax.Options;
            Assert.Multiple(
                () => Assert.Equal(0.8, options!.beta1),
                () => Assert.Equal(0.9, options!.beta2),
                () => Assert.Equal(0.001, options!.LearningRate),
                () => Assert.Equal(0, options!.weight_decay)
            );

            options = sd.Options[1] as Modules.Adamax.Options;
            Assert.Multiple(
                () => Assert.Equal(0.7, options!.beta1),
                () => Assert.Equal(0.79, options!.beta2),
                () => Assert.Equal(0.01, options!.LearningRate),
                () => Assert.Equal(0.3, options!.weight_decay)
            );

            var state = sd.State[0] as Modules.Adamax.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_inf)
            );

            state = sd.State[1] as Modules.Adamax.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_inf)
            );

            state = sd.State[2] as Modules.Adamax.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_inf)
            );

            state = sd.State[3] as Modules.Adamax.State;
            Assert.Multiple(
                () => Assert.Equal(1, state!.step),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_avg),
                () => Assert.NotNull(state!.exp_inf)
            );
        }
    }
}
