// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using Xunit;

using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
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
        }

        [Fact]
        public void ExplicitDisposal()
        {
            // Allocate many 256MB tensors. Without explicit disposal memory use relies on finalization.
            // This will often succeed but not reliably
            int n = 25;
            for (int i = 0; i < n; i++) {
                using (var x = torch.empty(new long[] { 64000, 1000 }, device: torch.CPU)) { }
            }
            if (torch.cuda.is_available())
            {
                for (int i = 0; i < n; i++) {
                    using (var x = torch.empty(new long[] { 64000, 1000 }, device: torch.CUDA)) { }
                }
            }
            Assert.True(true); // Just make sure we got here.
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

                var x = torch.rand(new long[] { 10, 10, 10 });
                Assert.Equal(new long[] { 10, 10, 10 }, x.shape);
            }
        }

        [Fact]
        public void TestExplicitGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going.
            lock (_lock) {

                long a, b, c, d;
                var gen = torch.random.manual_seed(4711);
                a = gen.initial_seed();

                torch.Generator genA = torch.Generator.Default;
                torch.Generator genB = new torch.Generator(4355);
                torch.Generator genC = new torch.Generator(4355);

                b = genA.initial_seed();
                c = genB.initial_seed();
                d = genC.initial_seed();

                Assert.Equal(a, b);
                Assert.Equal(c, d);
                Assert.NotEqual(a, c);
                Assert.Equal(4355, c);

                {
                    var x = torch.rand(100, generator: genB);
                    var y = torch.rand(100, generator: genC);
                    Assert.Equal(new long[] { 100 }, x.shape);
                    Assert.True(x.allclose(y));
                }

                {
                    var x = torch.randn(100, generator: genB);
                    var y = torch.randn(100, generator: genC);
                    Assert.Equal(new long[] { 100 }, x.shape);
                    Assert.True(x.allclose(y));
                }

                {
                    var x = torch.randint(1000, 100, generator: genB);
                    var y = torch.randint(1000, 100, generator: genC);
                    Assert.Equal(new long[] { 100 }, x.shape);
                    Assert.True(x.allclose(y));
                }

                {
                    var x = torch.randperm(1000, generator: genB);
                    var y = torch.randperm(1000, generator: genC);
                    Assert.Equal(new long[] { 1000 }, x.shape);
                    Assert.True(x.allclose(y));
                }
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
                    var val1 = torch.randn(new long[] { 1 });
                    var value1 = val1[0].ToSingle();

                    // Genereate a different number
                    var val2 = torch.randn(new long[] { 1 });
                    var value2 = val2[0].ToSingle();
                    Assert.NotEqual(value1, value2);

                    // Restore the state
                    gen.set_state(state);

                    // Generate the first number again.
                    var val3 = torch.randn(new long[] { 1 });
                    var value3 = val3[0].ToSingle();
                    Assert.Equal(value1, value3);
                }
            }
        }

        [Fact]
        public void TestUtilsPtoV()
        {
            var lin1 = nn.Linear(1000, 100);
            var lin2 = nn.Linear(100, 10);

            var submodules = new List<(string name, torch.nn.Module submodule)>();
            submodules.Add(("lin1", lin1));
            submodules.Add(("lin2", lin2));

            var seq = nn.Sequential(submodules);

            var vec = torch.nn.utils.parameters_to_vector(seq.parameters());

            Assert.Equal(101110, vec.NumberOfElements);
        }
        [Fact]
        public void TestUtilsVtoP()
        {
            var lin1 = nn.Linear(1000, 100);
            var lin2 = nn.Linear(100, 10);

            var submodules = new List<(string name, torch.nn.Module submodule)>();
            submodules.Add(("lin1", lin1));
            submodules.Add(("relu1", nn.ReLU()));
            submodules.Add(("lin2", lin2));

            var seq = nn.Sequential(submodules);

            var data = torch.rand(101110);

            torch.nn.utils.vector_to_parameters(data, seq.parameters());

            var data1 = torch.nn.utils.parameters_to_vector(seq.parameters());

            Assert.Equal(data.shape, data1.shape);
            Assert.Equal(data, data1);
        }

        // Because some of the tests mess with global state, and are run in parallel, we need to
        // acquire a lock before testing setting the default RNG see.
        private static object _lock = new object();
    }
}
