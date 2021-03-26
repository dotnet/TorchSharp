// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Tensor;
using Xunit;

#nullable enable

namespace TorchSharp
{
    public class TestTorch
    {
        [Fact]
        public void TestDeviceCount()
        {
            //var shape = new long[] { 2, 2 };

            var isCudaAvailable = Torch.IsCudaAvailable();
            var isCudnnAvailable = Torch.IsCudnnAvailable();
            var deviceCount = Torch.CudaDeviceCount();
            if (isCudaAvailable) {
                Assert.True(deviceCount > 0);
                Assert.True(isCudnnAvailable);
            } else {
                Assert.Equal(0, deviceCount);
                Assert.False(isCudnnAvailable);
            }

            //TorchTensor t = Float32Tensor.ones(shape);
            //Assert.Equal(shape, t.Shape);
            //Assert.Equal(1.0f, t[0, 0].ToSingle());
            //Assert.Equal(1.0f, t[1, 1].ToSingle());
        }

        [Fact]
        public void ExplicitDisposal()
        {
            // Allocate many 256MB tensors. Without explicit disposal memory use relies on finalization.
            // This will often succeed but not reliably
            int n = 25;
            for (int i = 0; i < n; i++) {
                Console.WriteLine("ExplicitDisposal: Loop iteration {0}", i);

                using (var x = Float32Tensor.empty(new long[] { 64000, 1000 }, device: Device.CPU)) { }
            }
            Console.WriteLine("Hello World!");
        }

        [Fact]
        public void TestDefaultGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going,
            long a, b, c;
            using(var gen = Torch.ManualSeed(4711)) {
                a = gen.InitialSeed;
            }
            using (var gen = TorchGenerator.Default) {
                b = gen.InitialSeed;
            }
            Assert.Equal(a, b);
            using (var gen = Torch.ManualSeed(17)) {
                c = gen.InitialSeed;
            }
            Assert.NotEqual(a, c);

            var x = Float32Tensor.rand(new long[] { 10, 10, 10 });
            Assert.Equal(new long[] { 10, 10, 10 }, x.shape);
        }
    }
}
