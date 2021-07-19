// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp;
using Xunit;

using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
    public class TestTorch
    {
        [Fact]
        public void TestEnumEquivalence()
        {
            Assert.Equal((int)InterpolationMode.Nearest, (int)UpsampleMode.Nearest);
            Assert.Equal((int)InterpolationMode.Linear, (int)UpsampleMode.Linear);
            Assert.Equal((int)InterpolationMode.Bilinear, (int)UpsampleMode.Bilinear);
            Assert.Equal((int)InterpolationMode.Bicubic, (int)UpsampleMode.Bicubic);
            Assert.Equal((int)InterpolationMode.Trilinear, (int)UpsampleMode.Trilinear);
            Assert.Equal((int)InterpolationMode.Nearest, (int)GridSampleMode.Nearest);
            Assert.Equal((int)InterpolationMode.Bilinear, (int)GridSampleMode.Bilinear);
        }

        [Fact]
        public void TestDeviceCount()
        {
            //var shape = new long[] { 2, 2 };

            var isCudaAvailable = torch.cuda.is_available();
            var isCudnnAvailable = torch.cuda.is_cudnn_available();
            var deviceCount = torch.cuda.device_count();
            if (isCudaAvailable) {
                Assert.True(deviceCount > 0);
                Assert.True(isCudnnAvailable);
            } else {
                Assert.Equal(0, deviceCount);
                Assert.False(isCudnnAvailable);
            }

            //Tensor t = Float32Tensor.ones(shape);
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

                using (var x = Float32Tensor.empty(new long[] { 64000, 1000 }, device: torch.CPU)) { }
            }
            Console.WriteLine("Hello World!");
        }

        [Fact]
        public void TestDefaultGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going,
            long a, b, c;
            lock (_lock) {
                using (var gen = torch.random.manual_seed(4711)) {
                    a = gen.initial_seed();
                }
                using (var gen = torch.Generator.Default) {
                    b = gen.initial_seed();
                }
                Assert.Equal(a, b);
                using (var gen = torch.random.manual_seed(17)) {
                    c = gen.initial_seed();
                }
                Assert.NotEqual(a, c);

                var x = Float32Tensor.rand(new long[] { 10, 10, 10 });
                Assert.Equal(new long[] { 10, 10, 10 }, x.shape);
            }
        }

        [Fact]
        public void TestExplicitGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going.
            lock (_lock) {

                long a, b, c;
                using (var gen = torch.random.manual_seed(4711)) {
                    a = gen.initial_seed();
                }
                using (torch.Generator gen = torch.Generator.Default, genA = new torch.Generator(4355)) {
                    b = gen.initial_seed();
                    c = genA.initial_seed();
                }
                Assert.Equal(a, b);
                Assert.NotEqual(a, c);
                Assert.Equal(4355, c);
            }
        }

        [Fact(Skip = "Some non-deterministic race condition causes this test to fail every now and then. The lock help, but not completely.")]
        public void TestGeneratorState()
        {
            // This test fails intermittently with CUDA. Just skip it.
            if (torch.cuda.is_available()) return;

            // After restoring a saved RNG state, the next number should be the
            // same as right after the snapshot.

            lock (_lock) {
                using (var gen = torch.random.manual_seed(4711)) {

                    // Take a snapshot
                    var state = gen.get_state();
                    Assert.NotNull(state);

                    // Generate a number
                    var val1 = Float32Tensor.randn(new long[] { 1 });
                    var value1 = val1[0].ToSingle();

                    // Genereate a different number
                    var val2 = Float32Tensor.randn(new long[] { 1 });
                    var value2 = val2[0].ToSingle();
                    Assert.NotEqual(value1, value2);

                    // Restore the state
                    gen.set_state(state);

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
