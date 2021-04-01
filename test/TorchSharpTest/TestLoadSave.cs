// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using TorchSharp.Tensor;
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
            linear.Save(".model.ts");
            var loadedLinear = NN.Linear.Load(".model.ts");
            File.Delete(".model.ts");
            Assert.NotNull(loadedLinear);
        }

        [Fact]
        public void TestSaveLoadConv2D()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var conv = Conv2D(100, 10, 5);
            conv.Save(".model.ts");
            var loaded = NN.Conv2d.Load(".model.ts");
            File.Delete(".model.ts");
            Assert.NotNull(loaded);
        }

        [Fact]
        public void TestSaveLoadSequence()
        {
            if (File.Exists(".model-list.txt")) File.Delete(".model-list.txt");
            var lin1 = Linear(100, 10, true);
            var lin2 = Linear(10, 5, true);
            var seq = Sequential(("lin1", lin1), ("lin2", lin2));
            seq.Save(".model-list.txt");
            var loaded = NN.Sequential.Load(".model-list.txt");
            File.Delete("model-list.txt");
            Assert.NotNull(loaded);
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
                    Assert.True(target.Data<int>().ToArray().Where(x => x >= 0 && x < 10).Count() == 16);

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
