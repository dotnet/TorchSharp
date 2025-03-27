// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;
using System.Security.Cryptography;
using Tensorboard;
using Xunit;

using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestTorch
    {
        [Fact]
        public void FInfoTest()
        {
            static void AssertScalarEqual(Tensor expected, Tensor actual)
            {
                Assert.Equal((double)expected, (double)actual);
            }
            static void AssertScalarNotEqual(Tensor expected, Tensor actual)
            {
                Assert.NotEqual((double)expected, (double)actual);
            }

            var floatingTypes = new[] {
                ScalarType.Float16, ScalarType.Float32, ScalarType.Float64, ScalarType.BFloat16
            };
            foreach (var scalarType in floatingTypes) {
                var info = finfo(scalarType);

                var zeroPointFour = tensor(0.4, scalarType);
                var zeroPointNine = tensor(0.9, scalarType);
                var one = tensor(1, scalarType);
                var zero = tensor(1, scalarType);
                Assert.Equal(one.dtype.ElementSize() * 8, info.bits);
                
                var eps = tensor(info.eps, scalarType);
                AssertScalarNotEqual(one, one + eps);
                AssertScalarEqual(one + eps, one + eps * zeroPointNine);
                AssertScalarEqual(one, one + eps * zeroPointFour);

                var max = tensor(info.max, scalarType);
                AssertScalarEqual(max, max + eps);

                var min = tensor(info.min, scalarType);
                AssertScalarEqual(-max, min);
                AssertScalarEqual(min, min - eps);

                var tiny = tensor(info.tiny, scalarType);
                // not sure how to test for tiny.

                var smallest_normal = tensor(info.smallest_normal, scalarType);
                AssertScalarEqual(tiny, smallest_normal);

                var resolution = tensor(info.resolution, scalarType);
                // not sure how to test for resolution.
            }

            var complexTypes = new[] {
                (ScalarType.ComplexFloat32, ScalarType.Float32),
                (ScalarType.ComplexFloat64, ScalarType.Float64)
            };
            foreach (var (complex, floating) in complexTypes) {
                var c = finfo(complex);
                var f = finfo(floating);

                Assert.Equal(f.bits, c.bits);
                Assert.Equal(f.eps, c.eps);
                Assert.Equal(f.max, c.max);
                Assert.Equal(f.min, c.min);
                Assert.Equal(f.tiny, c.tiny);
                Assert.Equal(f.smallest_normal, c.smallest_normal);
                Assert.Equal(f.resolution, c.resolution);
            }
        }

        [Fact]
        public void EnumEquivalence()
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
        public void DeviceCount()
        {
            //var shape = new long[] { 2, 2 };

            var isCudaAvailable = cuda.is_available();
            var isCudnnAvailable = cuda.is_cudnn_available();
            var deviceCount = cuda.device_count();
            if (isCudaAvailable) {
                Assert.True(deviceCount > 0);
                Assert.True(isCudnnAvailable);
            } else {
                Assert.Equal(0, deviceCount);
                Assert.False(isCudnnAvailable);
            }
        }

        [Fact]
        public void Synchronize()
        {
            // Will throw if the underlying native function fails to link at runtime
            cuda.synchronize();
        }

        [Fact]
        public void ExplicitDisposal()
        {
            // Allocate many 256MB tensors. Without explicit disposal memory use relies on finalization.
            // This will often succeed but not reliably
            int n = 25;
            for (int i = 0; i < n; i++) {
                using (var x = empty(new long[] { 64000, 1000 }, device: CPU)) { }
            }
            if (cuda.is_available())
            {
                for (int i = 0; i < n; i++) {
                    using (var x = empty(new long[] { 64000, 1000 }, device: CUDA)) { }
                }
            }
            Assert.True(true); // Just make sure we got here.
        }

        [Fact]
        public void DefaultGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going,
            long a, b, c;
            lock (_lock) {
                using (var g = Generator.Default) {
                    a = g.initial_seed();
                }
                using (var g = Generator.Default) {
                    b = g.initial_seed();
                    Assert.Equal(a, b);
                }

                using var gen = random.manual_seed(4711);
                a = gen.initial_seed();
                Assert.Equal(4711, a);

                using (var g = new torch.Generator()) {
                    b = g.initial_seed();
                    Assert.NotEqual(a, b);
                }
                using (var g = Generator.Default) {
                    b = g.initial_seed();
                    Assert.Equal(a, b);
                }
                using var gen1 = random.manual_seed(17);
                c = gen.initial_seed();
                Assert.Equal(17, c);
                Assert.NotEqual(a, c);

                var x = rand(new long[] { 10, 10, 10 });
                Assert.Equal(new long[] { 10, 10, 10 }, x.shape);
            }
        }

        [Fact]
        public void ExplicitGenerators()
        {
            // This tests that the default generator can be disposed, but will keep on going.
            lock (_lock) {

                long a, b, c, d;
                var gen = random.manual_seed(4711);
                a = gen.initial_seed();

                Generator genA = Generator.Default;
                Generator genB = new Generator(4355);
                Generator genC = new Generator(4355);

                b = genA.initial_seed();
                c = genB.initial_seed();
                d = genC.initial_seed();

                Assert.Equal(a, b);
                Assert.Equal(c, d);
                Assert.NotEqual(a, c);
                Assert.Equal(4355, c);

                {
                    var x = rand(100, generator: genB);
                    var y = rand(100, generator: genC);
                    Assert.Equal(new long[] { 100 }, x.shape);
                    Assert.True(x.allclose(y));
                }

                {
                    var x = randn(100, generator: genB);
                    var y = randn(100, generator: genC);
                    Assert.Equal(new long[] { 100 }, x.shape);
                    Assert.True(x.allclose(y));
                }

                {
                    var x = randint(1000, 100, generator: genB);
                    var y = randint(1000, 100, generator: genC);
                    Assert.Equal(new long[] { 100 }, x.shape);
                    Assert.True(x.allclose(y));
                }

                {
                    var x = randperm(1000, generator: genB);
                    var y = randperm(1000, generator: genC);
                    Assert.Equal(new long[] { 1000 }, x.shape);
                    Assert.True(x.allclose(y));
                }
            }
        }

        [Fact]//(Skip = "Some non-deterministic race condition causes this test to fail every now and then. The lock helps, but not completely.")]
        public void GeneratorState()
        {
            // This test fails intermittently with CUDA. Just skip it.
            if (cuda.is_available()) return;

            // After restoring a saved RNG state, the next number should be the
            // same as right after the snapshot.

            lock (_lock) {
                using (var gen = random.manual_seed(4711)) {

                    // Take a snapshot
                    var state = gen.get_state();
                    Assert.NotNull(state);

                    // Generate a number
                    var val1 = randn(new long[] { 1 });
                    var value1 = val1[0].ToSingle();

                    // Genereate a different number
                    var val2 = randn(new long[] { 1 });
                    var value2 = val2[0].ToSingle();
                    Assert.NotEqual(value1, value2);

                    // Restore the state
                    gen.set_state(state);

                    // Generate the first number again.
                    var val3 = randn(new long[] { 1 });
                    var value3 = val3[0].ToSingle();
                    Assert.Equal(value1, value3);
                }
            }
        }

        [Fact]
        public void UtilsPtoV()
        {
            var lin1 = nn.Linear(1000, 100);
            var lin2 = nn.Linear(100, 10);

            var submodules = new List<(string name, nn.Module<Tensor, Tensor> submodule)>();
            submodules.Add(("lin1", lin1));
            submodules.Add(("lin2", lin2));

            var seq = nn.Sequential(submodules);

            var vec = nn.utils.parameters_to_vector(seq.parameters());

            Assert.Equal(101110, vec.NumberOfElements);

            var array = vec.detach().data<float>().ToArray();
            torch.nn.utils.vector_to_parameters(torch.as_tensor(array).cpu(), seq.parameters());
        }
        [Fact]
        public void UtilsVtoP()
        {
            var lin1 = nn.Linear(1000, 100);
            var lin2 = nn.Linear(100, 10);

            var submodules = new List<(string name, nn.Module<Tensor, Tensor> submodule)>();
            submodules.Add(("lin1", lin1));
            submodules.Add(("relu1", nn.ReLU()));
            submodules.Add(("lin2", lin2));

            var seq = nn.Sequential(submodules);

            var data = rand(101110);

            nn.utils.vector_to_parameters(data, seq.parameters());

            var data1 = nn.utils.parameters_to_vector(seq.parameters());

            Assert.Equal(data.shape, data1.shape);
            Assert.Equal(data, data1);
        }

        [Fact]
        public void UtilsFusion()
        {
            static void SetRandomParameter<T>(
                T module,
                Expression<Func<T, Modules.Parameter>> parameterProperty)
            {
                var propertyExpression = (MemberExpression)parameterProperty.Body;
                var property = (PropertyInfo)propertyExpression.Member;
                var parameter = (Modules.Parameter)property.GetValue(module)!;
                var randomTensor = rand_like(
                    parameter,
                    parameter.dtype,
                    parameter.device) * 100;
                var newParameter = new Modules.Parameter(randomTensor, parameter.requires_grad);
                property.SetValue(module, newParameter);
            }

            static void SetRandomTensor<T>(
                T module,
                Expression<Func<T, Tensor>> tensorProperty)
            {
                var propertyExpression = (MemberExpression)tensorProperty.Body;
                var property = (PropertyInfo)propertyExpression.Member;
                var tensor = (Tensor)property.GetValue(module)!;
                var newTensor = rand_like(
                    tensor,
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad) * 100;
                property.SetValue(module, newTensor);
            }

            static void AssertRelativelyEqual(
                Tensor expected, Tensor actual, double tolerance = 1e-5)
            {
                Assert.Equal(expected.size(), actual.size());
                var difference = (expected - actual) / expected;
                var maxDifference = (double)difference.abs().max();
                Assert.InRange(maxDifference, -tolerance, tolerance);
            }

            {
                // linear
                var x = rand(new long[] { 20, 20 }) * 100;

                var linear = nn.Linear(20, 5);
                linear.eval();
                SetRandomParameter(linear, x => x.weight!);
                SetRandomParameter(linear, x => x.bias!);

                var batchNorm1d = nn.BatchNorm1d(5, eps: 1);
                batchNorm1d.eval();
                SetRandomParameter(batchNorm1d, x => x.weight!);
                SetRandomParameter(batchNorm1d, x => x.bias!);
                SetRandomTensor(batchNorm1d, x => x.running_mean!);
                SetRandomTensor(batchNorm1d, x => x.running_var!);

                (var weight, var bias) = nn.utils.fuse_linear_bn_weights(
                    linear.weight!, linear.bias,
                    batchNorm1d.running_mean!, batchNorm1d.running_var!,
                    bn_eps: 1, batchNorm1d.weight!, batchNorm1d.bias!);

                var newLinear = nn.Linear(20, 5);
                newLinear.eval();
                newLinear.weight = weight;
                newLinear.bias = bias;

                AssertRelativelyEqual(
                    batchNorm1d.call(linear.call(x)),
                    newLinear.call(x));
            }

            {
                // conv
                var x = rand(new long[] { 20, 20, 20, 20 }) * 100;
                var conv = nn.Conv2d(20, 5, 3);
                conv.eval();
                SetRandomParameter(conv, x => x.weight!);
                SetRandomParameter(conv, x => x.bias!);

                var batchNorm2d = nn.BatchNorm2d(5, eps: 13);
                batchNorm2d.eval();
                SetRandomParameter(batchNorm2d, x => x.weight!);
                SetRandomParameter(batchNorm2d, x => x.bias!);
                SetRandomTensor(batchNorm2d, x => x.running_mean!);
                SetRandomTensor(batchNorm2d, x => x.running_var!);

                (var weight, var bias) = nn.utils.fuse_conv_bn_weights(
                    conv.weight!, conv.bias,
                    batchNorm2d.running_mean!, batchNorm2d.running_var!,
                    bn_eps: 13, batchNorm2d.weight!, batchNorm2d.bias!);

                var newConv = nn.Conv2d(20, 5, 3);
                newConv.eval();
                newConv.weight = weight;
                newConv.bias = bias;

                AssertRelativelyEqual(
                    batchNorm2d.call(conv.call(x)),
                    newConv.call(x));
            }
        }

        [Fact(Skip = "Intermittently fails")]
        public void AllowTF32()
        {
            Assert.False(torch.backends.cuda.matmul.allow_tf32);
            Assert.True(torch.backends.cudnn.allow_tf32);

            torch.backends.cuda.matmul.allow_tf32 = true;
            torch.backends.cudnn.allow_tf32 = false;

            Assert.True(torch.backends.cuda.matmul.allow_tf32);
            Assert.False(torch.backends.cudnn.allow_tf32);
        }

        [Fact(Skip = "Intermittently fails")]
        public void EndableSDP()
        {
            Assert.True(torch.backends.cuda.flash_sdp_enabled());
            Assert.True(torch.backends.cuda.math_sdp_enabled());

            torch.backends.cuda.enable_flash_sdp(false);
            torch.backends.cuda.enable_math_sdp(false);

            Assert.False(torch.backends.cuda.flash_sdp_enabled());
            Assert.False(torch.backends.cuda.math_sdp_enabled());
        }

        [Fact]
        public void EnableInferenceMode()
        {
            Assert.False(torch.is_inference_mode_enabled());

            using (var d = torch.inference_mode()) {
                Assert.True(torch.is_inference_mode_enabled());
            }

            Assert.False(torch.is_inference_mode_enabled());
        }

        [Fact(Skip ="Intermittently fails")]
        public void AllowFP16ReductionCuBLAS()
        {
            Assert.True(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction);
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
            Assert.False(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction);
        }

        [Fact]
        public void CheckVersionStrings()
        {
            Assert.Equal("2.5.1", torch.NormalizeNuGetVersion("2.5.1.0"));
            Assert.Equal("0.105.0", torch.NormalizeNuGetVersion("0.105.0.0"));
            Assert.Equal("0.1.0-alpha", torch.NormalizeNuGetVersion("0.1.0-alpha"));
            Assert.Equal("0.1.0", torch.NormalizeNuGetVersion("0.1.0"));
            Assert.Throws<ArgumentException>(() => NormalizeNuGetVersion(""));
            Assert.Throws<ArgumentException>(() => NormalizeNuGetVersion("1.2.3.4.5"));
        }

        // Because some of the tests mess with global state, and are run in parallel, we need to
        // acquire a lock before testing setting the default RNG see.
        private static object _lock = new object();
    }
}
