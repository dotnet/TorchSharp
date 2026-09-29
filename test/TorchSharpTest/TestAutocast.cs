// Copyright (c) .NET Foundation and Contributors. All Rights Reserved. See LICENSE in the project root for license information.
using System;
using System.Threading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpTest
{
    [Collection("Sequential")]
    public class TestAutocast
    {
        [Fact]
        public void CpuBFloat16AutocastKeepsParametersFloat32()
        {
            using var scope = NewDisposeScope();
            using var linear = Linear(8, 8);
            using var input = randn(2, 8);
            Assert.Equal(ScalarType.Float32, linear.weight.dtype);
            var wasEnabled = is_autocast_enabled(DeviceType.CPU);

            Tensor loss;
            using (autocast(DeviceType.CPU, ScalarType.BFloat16))
            {
                Assert.True(is_autocast_enabled(DeviceType.CPU));
                using (autocast(DeviceType.CPU, ScalarType.BFloat16))
                {
                    using var nestedOutput = linear.forward(input);
                    Assert.Equal(ScalarType.BFloat16, nestedOutput.dtype);
                }
                Assert.True(is_autocast_enabled(DeviceType.CPU));
                using var output = linear.forward(input);
                loss = output.to_type(ScalarType.Float32).pow(2).mean();
            }

            Assert.Equal(wasEnabled, is_autocast_enabled(DeviceType.CPU));
            loss.backward();
            Assert.Equal(ScalarType.Float32, linear.weight.dtype);
            Assert.Equal(ScalarType.Float32, linear.weight.grad!.dtype);
            using var beforeStep = linear.weight.detach().clone();
            using var optimizer = torch.optim.AdamW(linear.parameters(), lr: 0.01);
            optimizer.step();
            Assert.True((linear.weight - beforeStep).abs().max().item<float>() > 0);
        }

        [Fact]
        public void AutocastScopeMustBeDisposedOnItsCreatingThread()
        {
            var wasEnabled = is_autocast_enabled(DeviceType.CPU);
            using var guard = autocast(DeviceType.CPU, ScalarType.BFloat16);
            Exception disposeError = null;
            var otherThread = new Thread(() =>
            {
                try { guard.Dispose(); }
                catch (Exception error) { disposeError = error; }
            }) { IsBackground = true };
            otherThread.Start();
            Assert.True(otherThread.Join(TimeSpan.FromSeconds(5)));
            Assert.IsType<InvalidOperationException>(disposeError);
            Assert.True(is_autocast_enabled(DeviceType.CPU));
            guard.Dispose();
            Assert.Equal(wasEnabled, is_autocast_enabled(DeviceType.CPU));
        }

#if TORCHSHARP_CUDA_TESTS
        [Fact]
        public void CudaBFloat16AutocastKeepsParametersFloat32()
        {
            Assert.True(cuda.is_available(), "CUDA is required for this test.");

            using var scope = NewDisposeScope();
            using var linear = Linear(8, 8).to(CUDA);
            using var input = randn(new long[] { 2, 8 }, device: CUDA);
            var wasEnabled = is_autocast_enabled(DeviceType.CUDA);
            Assert.Equal(ScalarType.Float32, linear.weight.dtype);
            Tensor loss;
            using (autocast(DeviceType.CUDA, ScalarType.BFloat16))
            {
                Assert.True(is_autocast_enabled(DeviceType.CUDA));
                using (autocast(DeviceType.CUDA, ScalarType.BFloat16))
                {
                    using var nestedOutput = linear.forward(input);
                    Assert.Equal(ScalarType.BFloat16, nestedOutput.dtype);
                }
                Assert.True(is_autocast_enabled(DeviceType.CUDA));
                using var output = linear.forward(input);
                Assert.Equal(ScalarType.BFloat16, output.dtype);
                loss = output.to_type(ScalarType.Float32).pow(2).mean();
            }

            Assert.Equal(wasEnabled, is_autocast_enabled(DeviceType.CUDA));
            loss.backward();
            Assert.Equal(ScalarType.Float32, linear.weight.dtype);
            Assert.Equal(ScalarType.Float32, linear.weight.grad!.dtype);
            using var beforeStep = linear.weight.detach().clone();
            using var optimizer = torch.optim.AdamW(linear.parameters(), lr: 0.01);
            optimizer.step();
            Assert.True((linear.weight - beforeStep).abs().max().item<float>() > 0);
        }

        [Fact]
        public void CudaNestedDtypeRestoresOuterBFloat16()
        {
            Assert.True(cuda.is_available(), "CUDA is required for this test.");

            using var scope = NewDisposeScope();
            using var linear = Linear(8, 8).to(CUDA);
            using var input = randn(new long[] { 2, 8 }, device: CUDA);
            var wasEnabled = is_autocast_enabled(DeviceType.CUDA);

            using (autocast(DeviceType.CUDA, ScalarType.BFloat16))
            {
                using var before = linear.forward(input);
                Assert.Equal(ScalarType.BFloat16, before.dtype);
                using (autocast(DeviceType.CUDA, ScalarType.Float16))
                {
                    using var inner = linear.forward(input);
                    Assert.Equal(ScalarType.Float16, inner.dtype);
                }
                using var after = linear.forward(input);
                Assert.Equal(ScalarType.BFloat16, after.dtype);
            }

            Assert.Equal(wasEnabled, is_autocast_enabled(DeviceType.CUDA));
            Assert.Equal(ScalarType.Float32, linear.weight.dtype);
        }
#endif
    }
}
