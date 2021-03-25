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

    }
}
