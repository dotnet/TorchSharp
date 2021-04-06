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
            lock (_lock) {
                using (var gen = Torch.ManualSeed(4711)) {
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

        [Fact]
        public void TestExplicitGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going,
            long a, b, c;
            using (var gen = Torch.ManualSeed(4711)) {
                a = gen.InitialSeed;
            }
            using (TorchGenerator gen = TorchGenerator.Default, genA = new TorchGenerator(4355)) {
                b = gen.InitialSeed;
                c = genA.InitialSeed;
            }
            Assert.Equal(a, b);
            Assert.NotEqual(a, c);
            Assert.Equal(4355, c);
        }

        [Fact]
        public void TestGeneratorState()
        {
            // After restoring a saved RNG state, the next number should be the
            // same as right after the snapshot.

            lock (_lock) {
                using (var gen = Torch.ManualSeed(4711)) {

                    // Take a snapshot
                    var state = gen.State;
                    Assert.NotNull(state);

                    // Generate a number
                    var val1 = Float32Tensor.randn(new long[] { 1 });
                    var value1 = val1[0].ToSingle();

                    // Genereate a different number
                    var val2 = Float32Tensor.randn(new long[] { 1 });
                    var value2 = val2[0].ToSingle();
                    Assert.NotEqual(value1, value2);

                    // Restore the state
                    gen.State = state;

                    // Generate the first number again.
                    var val3 = Float32Tensor.randn(new long[] { 1 });
                    var value3 = val3[0].ToSingle();
                    Assert.Equal(value1, value3);
                }
            }
        }

        // Because some of the tests mess with global state, and are run in parallel, we need to
        // acquire a lock before testing setting the default RNG see.
        private static object _lock = new object();
    }
}
