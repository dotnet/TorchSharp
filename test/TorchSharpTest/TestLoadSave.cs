// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using Xunit;

#nullable enable

namespace TorchSharp
{
    public class TestLoadSave
    {
        [Fact]
        public void TestSaveLoadLinear()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var linear = Linear(100, 10, true);
            var params0 = linear.parameters();
            linear.save(".model.ts");
            var loadedLinear = Linear(100, 10, true);
            loadedLinear.load(".model.ts");
            var params1 = loadedLinear.parameters();
            File.Delete(".model.ts");

            Assert.Equal(params0, params1);
        }

        [Fact]
        public void TestSaveLoadConv2D()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var conv = Conv2d(100, 10, 5);
            var params0 = conv.parameters();
            conv.save(".model.ts");
            var loaded = Conv2d(100, 10, 5);
            loaded.load(".model.ts");
            var params1 = loaded.parameters();
            File.Delete(".model.ts");

            Assert.Equal(params0, params1);
        }

        [Fact]
        public void TestSaveLoadSequential()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var conv = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            var params0 = conv.parameters();
            conv.save(".model.ts");
            var loaded = Sequential(Conv2d(100, 10, 5), Linear(100, 10, true));
            loaded.load(".model.ts");
            var params1 = loaded.parameters();
            File.Delete(".model.ts");

            Assert.Equal(params0, params1);
        }

        [Fact]
        public void TestSaveLoadCustomWithParameters()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");

            var original = new TestModule1();
            Assert.True(original.has_parameter("test"));

            var params0 = original.parameters();
            Assert.True(params0[0].requires_grad);
            original.save(".model.ts");

            var loaded = new TestModule1();
            Assert.True(loaded.has_parameter("test"));

            var params1 = loaded.parameters();
            Assert.True(params1[0].requires_grad);
            Assert.NotEqual(params0, params1);

            loaded.load(".model.ts");
            var params2 = loaded.parameters();
            Assert.True(params2[0].requires_grad);

            File.Delete(".model.ts");

            Assert.Equal(params0, params2);
        }

        private class TestModule1 : Module
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
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var linear = Linear(100, 10, true);
            var params0 = linear.parameters();
            linear.save(".model.ts");
            var loaded = Conv2d(100, 10, 5);
            Assert.Throws<ArgumentException>(() => loaded.load(".model.ts"));
            File.Delete(".model.ts");
        }

        [Fact]
        public void TestSaveLoadError_2()
        {
            // Submodule count mismatch
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var linear = Sequential(("linear1", Linear(100, 10, true)));
            var params0 = linear.parameters();
            linear.save(".model.ts");
            var loaded = Sequential(("linear1", Linear(100, 10, true)), ("conv1", Conv2d(100, 10, 5)));
            Assert.Throws<ArgumentException>(() => loaded.load(".model.ts"));
            // Shouldn't get an error for this:
            loaded.load(".model.ts", strict: false);
            File.Delete(".model.ts");
        }

        [Fact]
        public void TestSaveLoadError_3()
        {
            // Submodule name mismatch
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var linear = Sequential(("linear1", Linear(100, 10, true)));
            var params0 = linear.parameters();
            linear.save(".model.ts");
            var loaded = Sequential(("linear2", Linear(100, 10, true)));
            Assert.Throws<ArgumentException>(() => loaded.load(".model.ts"));
            File.Delete(".model.ts");
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

        [Fact(Skip = "The native saving/loading of models does not seem to work right now.")]
        public void TestNativeSaveLoad()
        {
            if (File.Exists(".model.native.bin")) File.Delete(".model.native.bin");
            var linear = Linear(100, 10, true);
            var params0 = linear.parameters();
            linear.Save(".model.native.bin");

            var lin3 = Modules.Linear.Load(".model.native.bin");

            File.Delete(".model.native.bin");

            var params1 = lin3.parameters();
            Assert.Equal(params0, params1);
        }

        [Fact(Skip = "CIFAR10 data too big to keep in repo")]
        public void TestCIFAR10Loader()
        {
            using (var train = Data.Loader.CIFAR10("../../../../src/Examples/Data", 16)) {
                Assert.NotNull(train);

                var size = train.Size();
                int i = 0;

                foreach (var (data, target) in train) {
                    i++;

                    Assert.Equal(data.shape, new long[] { 16, 3, 32, 32 });
                    Assert.Equal(target.shape, new long[] { 16 });
                    Assert.True(target.data<int>().ToArray().Where(x => x >= 0 && x < 10).Count() == 16);

                    data.Dispose();
                    target.Dispose();
                }

                Assert.Equal(size, i * 16);
            }
        }

        [Fact(Skip = "MNIST data too big to keep in repo")]
        public void TestMNISTLoaderWithEpochs()
        {
            using (var train = Data.Loader.MNIST("../../../../test/data/MNIST", 32)) {
                var size = train.Size();
                var epochs = 10;

                int i = 0;

                for (int e = 0; e < epochs; e++) {
                    foreach (var (data, target) in train) {
                        i++;

                        Assert.Equal(data.shape, new long[] { 32, 1, 28, 28 });
                        Assert.Equal(target.shape, new long[] { 32 });

                        data.Dispose();
                        target.Dispose();
                    }
                }

                Assert.Equal(size * epochs, i * 32);
            }
        }
    }
}
