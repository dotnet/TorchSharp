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

            //TorchTensor t = FloatTensor.Ones(shape);
            //Assert.Equal(shape, t.Shape);
            //Assert.Equal(1.0f, t[0, 0].DataItem<float>());
            //Assert.Equal(1.0f, t[1, 1].DataItem<float>());
        }

        [Fact]
        public void ExplicitDisposal()
        {
            // Allocate many 256MB tensors. Without explicit disposal memory use relies on finalization.
            // This will often succeed but not reliably
            int n = 50;
            for (int i = 0; i < n; i++) {
                Console.WriteLine("ExplicitDisposal: Loop iteration {0}", i);

                using (var x = FloatTensor.Empty(new long[] { 64000, 1000 }, deviceType: DeviceType.CPU)) { }
            }
            Console.WriteLine("Hello World!");
        }

        [Fact]
        public void FinalizeWithExplicitMemoryPressure()
        {
            // 
            // Allocate many 512MB tensors. Without explicit disposal memory use relies on finalization.
            // Use explicit memory pressure for large tensors makes this succeed reliably.
            int n = 50;
            for (int i = 0; i < n; i++) {
                Console.WriteLine("FinalizeWithExplicitMemoryPressure: Loop iteration {0}", i);

                // Allocate a 256MB tensor
                var x = FloatTensor.Empty(new long[] { 64000, 1000 }, deviceType: DeviceType.CPU);
                x.RegisterAsMemoryPressure();
            }
            Console.WriteLine("Hello World!");
        }

    }
}
