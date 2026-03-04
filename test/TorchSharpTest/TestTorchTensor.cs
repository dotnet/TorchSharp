// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Sdk;
using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
    [TraitDiscoverer("CategoryDiscoverer", "TraitExtensibility")]
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = true)]
    public sealed class TestOfAttribute : Attribute, ITraitAttribute
    {
        public TestOfAttribute(string name) { Name = name; }
        public string Name { get; }
    }

    [Collection("Sequential")]
    public class TestTensor
    {
        [Fact]
        public void ScalarCreation()
        {
            using (var scalar = false.ToScalar()) {
                Assert.Equal(ScalarType.Bool, scalar.Type);
            }
            using (var scalar = ((byte)0).ToScalar()) {
                Assert.Equal(ScalarType.Int64, scalar.Type);
            }
            using (var scalar = ((short)0).ToScalar()) {
                Assert.Equal(ScalarType.Int64, scalar.Type);
            }
            using (var scalar = ((int)0).ToScalar()) {
                Assert.Equal(ScalarType.Int64, scalar.Type);
            }
            using (var scalar = ((long)0).ToScalar()) {
                Assert.Equal(ScalarType.Int64, scalar.Type);
            }
            using (var scalar = ((float)0).ToScalar()) {
                Assert.Equal(ScalarType.Float64, scalar.Type);
            }
            using (var scalar = ((double)0).ToScalar()) {
                Assert.Equal(ScalarType.Float64, scalar.Type);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.tolist))]
        public void Test1DToList()
        {
            {
                Tensor t = torch.zeros(4);
                var lst = t.tolist();
                var list = lst as System.Collections.IList;
                Assert.NotNull(list);
                if (list is not null) {
                    Assert.Equal(4, list.Count);
                    for (var idx = 0; idx < list.Count; idx++)
                        Assert.IsType<Scalar>(list[idx]);
                }
            }
            {
                Tensor t = torch.zeros(4, 4);
                var lst = t.tolist();
                var list = lst as System.Collections.IList;
                Assert.NotNull(list);

                if (list is not null) {
                    Assert.Equal(4, list.Count);
                    for (var idx = 0; idx < list.Count; idx++)
                        Assert.IsType<System.Collections.ArrayList>(list[idx]);
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        [TestOf(nameof(TensorExtensionMethods.jlstr))]
        public void ScalarToString()
        {
            {
                Tensor t = (Tensor)3.14f;
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[], type = Float32, device = cpu, value = 3.14", str);
            }
            {
                Tensor t = torch.tensor(3.14f);
                var str = t.jlstr("E2", cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[], type = Float32, device = cpu, value = 3.14E+000", str);
            }
            {
                Tensor t = torch.tensor((3.14f, 6.28f), torch.complex64);
                var str = t.jlstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[], type = ComplexFloat32, device = cpu, value = 3.14+6.28i", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void TestBFloat16ScalarToString()
        {
            // Scalar (0-d tensor)
            {
                var t = torch.tensor(3.14f, torch.bfloat16);
                var str = t.jlstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[], type = BFloat16, device = cpu, value = 3.1406", str);
            }
            {
                var t = torch.tensor(3.14f, torch.bfloat16);
                var str = t.npstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("3.1406", str);
            }
            {
                var t = torch.tensor(3.14f, torch.bfloat16);
                var str = t.cstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[], type = BFloat16, device = cpu, value = 3.1406", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void TestBFloat16TensorToString()
        {
            // 1-D tensor
            {
                var t = torch.zeros(4, torch.bfloat16);
                var str = t.ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[0, 0, 0, 0]", str);
            }
            // 1-D Julia
            {
                var t = torch.zeros(4, torch.bfloat16);
                var str = t.jlstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[4], type = BFloat16, device = cpu{Environment.NewLine} 0 0 0 0{Environment.NewLine}", str);
            }
            // 1-D CSharp
            {
                var t = torch.zeros(4, torch.bfloat16);
                var str = t.cstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[4], type = BFloat16, device = cpu, value = bfloat16 [] {0f, 0f, 0f, 0f}", str);
            }
            // 2-D tensor
            {
                var t = torch.ones(2, 3, torch.bfloat16);
                var str = t.ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[[1, 1, 1]{Environment.NewLine} [1, 1, 1]]", str);
            }
            // print() should not throw
            {
                var t = torch.randn(3, 3).to(torch.bfloat16);
                var originalOut = Console.Out;
                using (var sw = new StringWriter()) {
                    try {
                        Console.SetOut(sw);
                        t.print(cultureInfo: CultureInfo.InvariantCulture);
                        var result = sw.ToString();
                        Assert.False(string.IsNullOrEmpty(result));
                    } finally {
                        Console.SetOut(originalOut);
                    }
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void TestFloat16TensorToString()
        {
            // 1-D CSharp
            {
                var t = torch.zeros(4, torch.float16);
                var str = t.cstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[4], type = Float16, device = cpu, value = float16 [] {0f, 0f, 0f, 0f}", str);
            }
            // 1-D Numpy
            {
                var t = torch.zeros(4, torch.float16);
                var str = t.npstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal("[0, 0, 0, 0]", str);
            }
        }

        private string _sep = Environment.NewLine;

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        [TestOf(nameof(TensorExtensionMethods.jlstr))]
        public void Test1DToJuliaString()
        {
            {
                Tensor t = torch.zeros(4);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture, newLine: _sep);
                Assert.Equal($"[4], type = Float32, device = cpu{_sep} 0 0 0 0{_sep}", str);
            }
            {
                Tensor t = torch.zeros(4);
                var str = t.jlstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[4], type = Float32, device = cpu{Environment.NewLine} 0 0 0 0{Environment.NewLine}", str);
            }
            {
                Tensor t = torch.zeros(4, torch.complex64);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[4], type = ComplexFloat32, device = cpu{_sep} 0 0 0 0{_sep}", str);
            }
            {
                Tensor t = torch.ones(4, torch.complex64, torch.META);
                for (int i = 0; i < t.shape[0]; i++) t[i] = torch.tensor((1.0f * i, 2.43f * i * 2), torch.complex64);
                var str = t.jlstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[4], type = ComplexFloat32, device = meta", str);
            }
            {
                Tensor t = torch.ones(4, torch.complex64);
                for (int i = 0; i < t.shape[0]; i++) t[i] = torch.tensor((1.0f * i, 2.43f * i * 2), torch.complex64);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[4], type = ComplexFloat32, device = cpu{_sep} 0 1+4.86i 2+9.72i 3+14.58i{_sep}", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        [TestOf(nameof(TensorExtensionMethods.jlstr))]
        public void Test2DToJuliaString()
        {
            {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4);
                var str = t.ToString(true, cultureInfo: CultureInfo.InvariantCulture, newLine: _sep);
                Assert.Equal($"[2x4], type = Float32, device = cpu{_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}", str);
            }
            {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4);
                var str = t.str(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = Float32, device = cpu{Environment.NewLine}        0   3.141 6.2834 3.1415{Environment.NewLine} 6.28e-06 -13.142   0.01 4713.1{Environment.NewLine}", str);
            }
            if (torch.cuda.is_available()) {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4, device: torch.CUDA);
                var str = t.str(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = Float32, device = cuda:0{Environment.NewLine}        0   3.141 6.2834 3.1415{Environment.NewLine} 6.28e-06 -13.142   0.01 4713.1{Environment.NewLine}", str);
            }
            {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4, device: torch.META);
                var str = t.str(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = Float32, device = meta", str);
            }
            {
                Tensor t = torch.zeros(2, 4, torch.complex64);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = ComplexFloat32, device = cpu{_sep} 0 0 0 0{_sep} 0 0 0 0{_sep}", str);
            }
            {
                Tensor t = torch.ones(2, 4, torch.complex64);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = ComplexFloat32, device = cpu{_sep} 1 1 1 1{_sep} 1 1 1 1{_sep}", str);
            }
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.jlstr))]
        public void Test3DToJuliaString()
        {
            {
                Tensor t = torch.tensor(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, 2, 2, 4);
                var str = t.jlstr("0.0000000", cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal(($"[2x2x4], type = Float32, device = cpu\n[0,..,..] =\n 0.0000000   3.1410000 6.2834000    3.1415200\n" +
                             $" 0.0000063 -13.1415300 0.0100000 4713.1400000\n\n[1,..,..] =\n 0.0100000 0.0000000 0.0000000 0.0000000\n" +
                             $" 0.0000000 0.0000000 0.0000000 0.0000000\n").Replace("\n", Environment.NewLine),
                             str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test4DToJuliaString()
        {
            {
                Tensor t = torch.tensor(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, 2, 2, 2, 4);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x2x2x4], type = Float32, device = cpu{_sep}[0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}" +
                             $"[1,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[1,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test5DToJuliaString()
        {
            {
                Tensor t = torch.tensor(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, new long[] { 2, 2, 2, 2, 4 });
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x2x2x2x4], type = Float32, device = cpu{_sep}[0,0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,0,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[0,1,0,..,..] ={_sep}" +
                             $"        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,1,1,..,..] ={_sep} 0.01 0 0 0{_sep}   " +
                             $" 0 0 0 0{_sep}{_sep}[1,0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[1,0,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,1,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,1,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test6DToJuliaString()
        {
            {
                Tensor t = torch.tensor(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, new long[] { 2, 2, 2, 2, 2, 4 });
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x2x2x2x2x4], type = Float32, device = cpu{_sep}[0,0,0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,0,0,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[0,0,1,0,..,..] ={_sep}" +
                             $"        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,0,1,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}" +
                             $"[0,1,0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,1,0,1,..,..] ={_sep} 0.01 0 0 0{_sep}" +
                             $"    0 0 0 0{_sep}{_sep}[0,1,1,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[0,1,1,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,0,0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,0,0,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,0,1,0,..,..] ={_sep}" +
                             $"        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,0,1,1,..,..] ={_sep} 0.01 0 0 0{_sep}" +
                             $"    0 0 0 0{_sep}{_sep}[1,1,0,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[1,1,0,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,1,1,0,..,..] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,1,1,1,..,..] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void TestToString1()
        {
            Tensor t = torch.zeros(2, 2);
            var str = t.ToString();
            Assert.Equal("[2x2], type = Float32, device = cpu", str);
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.print))]
        public void TestTensorDefaultPrint()
        {
            Tensor t = torch.zeros(2, 2);
            string expectedOutput = t.ToString(TensorStringStyle.Default) + Environment.NewLine;
            var originalOut = Console.Out;
            using (var sw = new StringWriter()) {
                try {
                    Console.SetOut(sw);
                    t.print();
                    var result = sw.ToString();
                    Assert.Equal(expectedOutput, result);
                } finally {
                    Console.SetOut(originalOut);
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test1DToCSharpString()
        {
            var hundreds = torch.tensor(Enumerable.Range(0, 100).Select(x => 2 * x + 1).ToList(), ScalarType.Float32);

            Assert.Multiple(
            () => Assert.Equal("[4], type = Float32, device = cpu, value = float [] {0f, 0f, 0f, 0f}", torch.zeros(4).ToString(torch.csharp)),
            () => Assert.Equal("[100], type = Int32, device = cpu, value = int [] {1, 1, 1, ... 1, 1, 1}", torch.ones(100, ScalarType.Int32).ToString(torch.csharp)),
            () => Assert.Equal("[100], type = Float32, device = cpu, value = float [] {1f, 3f, 5f, ... 195f, 197f, 199f}", hundreds.ToString(torch.csharp))
            );
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test2DToCSharpString()
        {
            var hundreds = torch.tensor(Enumerable.Range(0, 100).Select(x => 2 * x + 1).ToList(), ScalarType.Float32);

            Assert.Multiple(
            () => Assert.Equal($"[2x4], type = Float32, device = cpu, value = {_sep}float [,] {{{_sep} {{0f, 0f, 0f, 0f}},{_sep} {{0f, 0f, 0f, 0f}}{_sep}}}", torch.zeros(2, 4).ToString(torch.csharp)),
            () => Assert.Equal($"[2x10], type = Int32, device = cpu, value = {_sep}int [,] {{{_sep} {{1, 1, 1, ... 1, 1, 1}},{_sep} {{1, 1, 1, ... 1, 1, 1}}{_sep}}}", torch.ones(2, 10, ScalarType.Int32).ToString(torch.csharp)),
            () => Assert.Equal($"[2x50], type = Float32, device = cpu, value = {_sep}float [,] {{{_sep} {{1f, 3f, 5f, ... 95f, 97f, 99f}},{_sep} {{101f, 103f, 105f, ... 195f, 197f, 199f}}{_sep}}}", hundreds.reshape(2, 50).ToString(torch.csharp)),
            () => Assert.Equal($"[10x10], type = Float32, device = cpu, value = {_sep}float [,] {{{_sep} {{1f, 3f, 5f, ... 15f, 17f, 19f}},{_sep} {{21f, 23f, 25f, ... 35f, 37f, 39f}},{_sep} {{41f, 43f, 45f, ... 55f, 57f, 59f}},{_sep} ...{_sep} {{141f, 143f, 145f, ... 155f, 157f, 159f}},{_sep} {{161f, 163f, 165f, ... 175f, 177f, 179f}},{_sep} {{181f, 183f, 185f, ... 195f, 197f, 199f}}{_sep}}}", hundreds.reshape(10, 10).ToString(torch.csharp))
            );
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test3DToCSharpString()
        {
            var hundreds = torch.tensor(Enumerable.Range(0, 100).Select(x => 2 * x + 1).ToList(), ScalarType.Float32);
            Assert.Multiple(
            () => Assert.Equal($"[2x2x4], type = Float32, device = cpu, value = {_sep}float [,,] {{{_sep} {{{{0f, 0f, 0f, 0f}},{_sep}  {{0f, 0f, 0f, 0f}}}},{_sep} {{{{0f, 0f, 0f, 0f}},{_sep}  {{0f, 0f, 0f, 0f}}}}{_sep}}}", torch.zeros(2, 2, 4).ToString(torch.csharp)),
            () => Assert.Equal($"[2x2x5], type = Int32, device = cpu, value = {_sep}int [,,] {{{_sep} {{{{1, 1, 1, 1, 1}},{_sep}  {{1, 1, 1, 1, 1}}}},{_sep} {{{{1, 1, 1, 1, 1}},{_sep}  {{1, 1, 1, 1, 1}}}}{_sep}}}", torch.ones(2, 2, 5, ScalarType.Int32).ToString(torch.csharp)),
            () => Assert.Equal($"[2x5x10], type = Float32, device = cpu, value = {_sep}float [,,] {{{_sep} {{{{1f, 3f, 5f, ... 15f, 17f, 19f}},{_sep}  {{21f, 23f, 25f, ... 35f, 37f, 39f}},{_sep}  {{41f, 43f, 45f, ... 55f, 57f, 59f}},{_sep}  {{61f, 63f, 65f, ... 75f, 77f, 79f}},{_sep}  {{81f, 83f, 85f, ... 95f, 97f, 99f}}}},{_sep} {{{{101f, 103f, 105f, ... 115f, 117f, 119f}},{_sep}  {{121f, 123f, 125f, ... 135f, 137f, 139f}},{_sep}  {{141f, 143f, 145f, ... 155f, 157f, 159f}},{_sep}  {{161f, 163f, 165f, ... 175f, 177f, 179f}},{_sep}  {{181f, 183f, 185f, ... 195f, 197f, 199f}}}}{_sep}}}", hundreds.reshape(2, 5, 10).ToString(torch.csharp)),
            () => Assert.Equal($"[10x2x5], type = Float32, device = cpu, value = {_sep}float [,,] {{{_sep} {{{{1f, 3f, 5f, 7f, 9f}},{_sep}  {{11f, 13f, 15f, 17f, 19f}}}},{_sep} {{{{21f, 23f, 25f, 27f, 29f}},{_sep}  {{31f, 33f, 35f, 37f, 39f}}}},{_sep} {{{{41f, 43f, 45f, 47f, 49f}},{_sep}  {{51f, 53f, 55f, 57f, 59f}}}},{_sep} ...{_sep} {{{{141f, 143f, 145f, 147f, 149f}},{_sep}  {{151f, 153f, 155f, 157f, 159f}}}},{_sep} {{{{161f, 163f, 165f, 167f, 169f}},{_sep}  {{171f, 173f, 175f, 177f, 179f}}}},{_sep} {{{{181f, 183f, 185f, 187f, 189f}},{_sep}  {{191f, 193f, 195f, 197f, 199f}}}}{_sep}}}", hundreds.reshape(10, 2, 5).ToString(torch.csharp))
            );
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test1DToNumpyString()
        {
            Assert.Equal("[0, 0, 0, 0]", torch.zeros(4).ToString(torch.numpy));
            Assert.Equal("[0, 0, 0, 0]", torch.zeros(4, torch.complex64).ToString(torch.numpy));
            {
                Tensor t = torch.ones(4, torch.complex64);
                for (int i = 0; i < t.shape[0]; i++) t[i] = torch.tensor((1.0f * i, 2.43f * i * 2), torch.complex64);
                var str = t.ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[0, 1+4.86i, 2+9.72i, 3+14.58i]", str);
            }
            Assert.Equal("[1, 1, 1, ... 1, 1, 1]", torch.ones(100, ScalarType.Int32).ToString(torch.numpy));
            Assert.Equal("[1, 3, 5, ... 195, 197, 199]", torch.tensor(Enumerable.Range(0, 100).Select(x => 2 * x + 1).ToList(), ScalarType.Float32).ToString(torch.numpy));
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test2DToNumpyString()
        {
            string str = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
            Assert.Equal($"[[0, 3.141, 6.2834, 3.1415]{_sep} [6.28e-06, -13.142, 0.01, 4713.1]]", str);
            {
                Tensor t = torch.zeros(5, 5, torch.complex64);
                for (int i = 0; i < t.shape[0]; i++)
                    for (int j = 0; j < t.shape[1]; j++)
                        t[i][j] = torch.tensor((1.24f * i, 2.491f * i * 2), torch.complex64);
                str = t.ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[[0, 0, 0, 0, 0]{_sep} [1.24+4.982i, 1.24+4.982i, 1.24+4.982i, 1.24+4.982i, 1.24+4.982i]{_sep} [2.48+9.964i, 2.48+9.964i, 2.48+9.964i, 2.48+9.964i, 2.48+9.964i]{_sep} [3.72+14.946i, 3.72+14.946i, 3.72+14.946i, 3.72+14.946i, 3.72+14.946i]{_sep} [4.96+19.928i, 4.96+19.928i, 4.96+19.928i, 4.96+19.928i, 4.96+19.928i]]", str);
            }
            Assert.Equal($"[[0, 0, 0, 0]{_sep} [0, 0, 0, 0]]", torch.zeros(2, 4, torch.complex64).ToString(torch.numpy));
            Assert.Equal($"[[1, 1, 1, 1]{_sep} [1, 1, 1, 1]]", torch.ones(2, 4, torch.complex64).ToString(torch.numpy));
            Assert.Equal($"[[7, 9, 11, ... 101, 103, 105]{_sep} [107, 109, 111, ... 201, 203, 205]]", torch.tensor(Enumerable.Range(1, 100).Select(x => x * 2 + 5).ToList(), new long[] { 2, 50 }, ScalarType.Float32).ToString(torch.numpy));
            Assert.Equal($"[[7, 9, 11, ... 201, 203, 205]{_sep} [207, 209, 211, ... 401, 403, 405]{_sep} [407, 409, 411, ... 601, 603, 605]{_sep} ...{_sep} [19407, 19409, 19411, ... 19601, 19603, 19605]{_sep} [19607, 19609, 19611, ... 19801, 19803, 19805]{_sep} [19807, 19809, 19811, ... 20001, 20003, 20005]]",
                torch.tensor(Enumerable.Range(1, 10000).Select(x => x * 2 + 5).ToList(), new long[] { 100, 100 },
                    ScalarType.Float32).ToString(torch.numpy));
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test3DToNumpyString()
        {
            {
                Assert.Equal(
                    $"[[[0, 3.141, 6.2834, 3.1415]{_sep}  [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep} [[0.01, 0, 0, 0]{_sep}  [0, 0, 0, 0]]]",
                    torch.tensor(
                        new float[] {
                            0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        }, 2, 2, 4).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
                var actual = torch.tensor(Enumerable.Range(1, 250).Select(x => x * 2 + 5).ToList(), new long[] { 5, 5, 10 }, ScalarType.Float32).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[[[7, 9, 11, ... 21, 23, 25]{_sep}  [27, 29, 31, ... 41, 43, 45]{_sep}  [47, 49, 51, ... 61, 63, 65]{_sep}  [67, 69, 71, ... 81, 83, 85]{_sep}  [87, 89, 91, ... 101, 103, 105]]{_sep}{_sep} [[107, 109, 111, ... 121, 123, 125]{_sep}  [127, 129, 131, ... 141, 143, 145]{_sep}  [147, 149, 151, ... 161, 163, 165]{_sep}  [167, 169, 171, ... 181, 183, 185]{_sep}  [187, 189, 191, ... 201, 203, 205]]{_sep}{_sep} [[207, 209, 211, ... 221, 223, 225]{_sep}  [227, 229, 231, ... 241, 243, 245]{_sep}  [247, 249, 251, ... 261, 263, 265]{_sep}  [267, 269, 271, ... 281, 283, 285]{_sep}  [287, 289, 291, ... 301, 303, 305]]{_sep}{_sep} [[307, 309, 311, ... 321, 323, 325]{_sep}  [327, 329, 331, ... 341, 343, 345]{_sep}  [347, 349, 351, ... 361, 363, 365]{_sep}  [367, 369, 371, ... 381, 383, 385]{_sep}  [387, 389, 391, ... 401, 403, 405]]{_sep}{_sep} [[407, 409, 411, ... 421, 423, 425]{_sep}  [427, 429, 431, ... 441, 443, 445]{_sep}  [447, 449, 451, ... 461, 463, 465]{_sep}  [467, 469, 471, ... 481, 483, 485]{_sep}  [487, 489, 491, ... 501, 503, 505]]]",
                    torch.tensor(Enumerable.Range(1, 250).Select(x => x * 2 + 5).ToList(), new long[] { 5, 5, 10 }, ScalarType.Float32).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test4DToNumpyString()
        {
            Assert.Equal($"[[[[0, 3.141, 6.2834, 3.1415]{_sep}   [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}  [[0.01, 0, 0, 0]{_sep}   [0, 0, 0, 0]]]{_sep}{_sep}{_sep} [[[0, 3.141, 6.2834, 3.1415]{_sep}   [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}  [[0.01, 0, 0, 0]{_sep}   [0, 0, 0, 0]]]]", torch.tensor(new float[] {
                0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            }, 2, 2, 2, 4).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test5DToNumpyString()
        {
            Assert.Equal(
                $"[[[[[0, 3.141, 6.2834, 3.1415]{_sep}    [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}   [[0.01, 0, 0, 0]{_sep}    [0, 0, 0, 0]]]{_sep}{_sep}{_sep}  [[[0, 3.141, 6.2834, 3.1415]{_sep}    [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}   [[0.01, 0, 0, 0]{_sep}    [0, 0, 0, 0]]]]{_sep}{_sep}{_sep}{_sep} [[[[0, 3.141, 6.2834, 3.1415]{_sep}    [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}   [[0.01, 0, 0, 0]{_sep}    [0, 0, 0, 0]]]{_sep}{_sep}{_sep}  [[[0, 3.141, 6.2834, 3.1415]{_sep}    [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}   [[0.01, 0, 0, 0]{_sep}    [0, 0, 0, 0]]]]]",
                torch.tensor(
                    new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f,
                        4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f,
                        6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f,
                    }, new long[] { 2, 2, 2, 2, 4 }).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test6DToNumpyString()
        {
            Assert.Equal(
                $"[[[[[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]{_sep}{_sep}{_sep}   [[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]]{_sep}{_sep}{_sep}{_sep}  [[[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]{_sep}{_sep}{_sep}   [[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]]]{_sep}{_sep}{_sep}{_sep}{_sep} [[[[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]{_sep}{_sep}{_sep}   [[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]]{_sep}{_sep}{_sep}{_sep}  [[[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]{_sep}{_sep}{_sep}   [[[0, 3.141, 6.2834, 3.1415]{_sep}     [6.28e-06, -13.142, 0.01, 4713.1]]{_sep}{_sep}    [[0.01, 0, 0, 0]{_sep}     [0, 0, 0, 0]]]]]]",
                torch.tensor(
                    new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f,
                        4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f,
                        6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f,
                        4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f,
                        6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f,
                        4713.14f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, new long[] { 2, 2, 2, 2, 2, 4 }).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
        }

        [Fact]
        [TestOf(nameof(Tensor.alias))]
        public void Alias()
        {
            var t = torch.randn(5);
            var t1 = t.alias();

            Assert.NotEqual(t.Handle, t1.Handle);
            t[0] = torch.tensor(3.14f);
            Assert.Equal(3.14f, t1[0].item<float>());
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void AliasDispose()
        {
            var t = torch.randn(5);
            var t1 = t.alias();

            Assert.NotEqual(t.Handle, t1.Handle);
            t.Dispose();
            Assert.Throws<InvalidOperationException>(() => t.Handle);
            Assert.NotEqual(IntPtr.Zero, t1.Handle);
            t1.Dispose();
            Assert.Throws<InvalidOperationException>(() => t1.Handle);
        }

        [Fact(Skip = "Sensitive to concurrency in xUnit test driver.")]
        [TestOf(nameof(torch.randn))]
        public void Usings()
        {
            var tCount = Tensor.TotalCount;

            using (var t = torch.randn(5)) { }

            Assert.Equal(tCount, Tensor.TotalCount);
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataBool()
        {
            var x = torch.ones(5, torch.@bool);
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<bool>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataByte()
        {
            var x = torch.ones(5, torch.uint8);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<byte>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataInt8()
        {
            var x = torch.ones(5, torch.int8);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<sbyte>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataInt16()
        {
            var x = torch.ones(5, torch.int16);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<short>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataInt32()
        {
            var x = torch.ones(5, torch.int32);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<int>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataInt64()
        {
            var x = torch.ones(5, torch.int64);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<long>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataFloat16()
        {
            var x = torch.ones(5, torch.float16);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
#if NET6_0_OR_GREATER
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            x.data<System.Half>();
#else
            x.data<float>();
#endif
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataBFloat16()
        {
            var x = torch.ones(5, torch.bfloat16);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            var accessor = x.data<BFloat16>();
            Assert.Equal(5, accessor.Count);
            Assert.Equal((BFloat16)1.0f, accessor[0]);
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void DataBFloat16Item()
        {
            // item<BFloat16> on scalar tensor
            var t = torch.tensor(3.14f, torch.bfloat16);
            var val = t.item<BFloat16>();
            Assert.Equal(3.140625f, val.ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void DataBFloat16RoundTrip()
        {
            // Create tensor from BFloat16 array, read back
            var input = new BFloat16[] { (BFloat16)1.0f, (BFloat16)2.0f, (BFloat16)3.0f, (BFloat16)0.5f };
            var t = torch.tensor(input);
            Assert.Equal(ScalarType.BFloat16, t.dtype);
            Assert.Equal(4, t.NumberOfElements);
            var output = t.data<BFloat16>().ToArray();
            Assert.Equal(input, output);
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryBFloat16()
        {
            {
                var array = new BFloat16[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.BFloat16, t.dtype);
            }

            {
                var array = new BFloat16[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.BFloat16, t.dtype);
            }

            {
                var array = new BFloat16[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.BFloat16, t.dtype);
            }

            {
                var array = new BFloat16[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.BFloat16, t.dtype);
            }

            {
                var array = new BFloat16[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.BFloat16, t.dtype);
            }

            {
                var array = new BFloat16[,,] { { { (BFloat16)1f, (BFloat16)2f }, { (BFloat16)3f, (BFloat16)4f } }, { { (BFloat16)5f, (BFloat16)6f }, { (BFloat16)7f, (BFloat16)8f } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.BFloat16, t.dtype);
                Assert.Equal(array.Cast<BFloat16>().ToArray(), t.data<BFloat16>().ToArray());
            }
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.ToBFloat16))]
        public void TensorToBFloat16Extension()
        {
            var t = torch.tensor(3.14f, torch.bfloat16);
            var bf = t.ToBFloat16();
            Assert.Equal(3.140625f, bf.ToSingle());
        }

        [Fact]
        [TestOf(nameof(BFloat16))]
        public void BFloat16StructBasics()
        {
            // Size must be 2 bytes
            Assert.Equal(2, Marshal.SizeOf<BFloat16>());

            // Precision loss: 333.0f -> BFloat16 -> float (matching PyTorch)
            var bf = (BFloat16)333.0f;
            Assert.Equal(332.0f, bf.ToSingle());

            // Round-to-nearest-even: 3.14f → BFloat16 matches PyTorch native conversion
            var bf_rne = (BFloat16)3.14f;
            Assert.Equal(3.140625f, bf_rne.ToSingle());

            // Round-trip for a value that fits exactly
            var bf2 = (BFloat16)1.0f;
            Assert.Equal(1.0f, bf2.ToSingle());

            // Special values
            Assert.True(BFloat16.IsNaN(BFloat16.NaN));
            Assert.True(BFloat16.IsPositiveInfinity(BFloat16.PositiveInfinity));
            Assert.True(BFloat16.IsNegativeInfinity(BFloat16.NegativeInfinity));
            Assert.True(BFloat16.IsFinite(BFloat16.One));
            Assert.False(BFloat16.IsFinite(BFloat16.NaN));

            // Arithmetic
            var a = (BFloat16)2.0f;
            var b = (BFloat16)3.0f;
            Assert.Equal(5.0f, (a + b).ToSingle());
            Assert.Equal(-1.0f, (a - b).ToSingle());
            Assert.Equal(6.0f, (a * b).ToSingle());

            // Comparison
            Assert.True(a < b);
            Assert.True(b > a);
            Assert.True(a == (BFloat16)2.0f);
            Assert.True(a != b);

            // Equality
            Assert.Equal((BFloat16)1.5f, (BFloat16)1.5f);
            Assert.NotEqual((BFloat16)1.5f, (BFloat16)2.0f);
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataFloat32()
        {
            var x = torch.ones(5, torch.float32);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<double>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<float>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataFloat64()
        {
            var x = torch.ones(5, torch.float64);
            Assert.Throws<System.ArgumentException>(() => x.data<bool>());
            Assert.Throws<System.ArgumentException>(() => x.data<byte>());
            Assert.Throws<System.ArgumentException>(() => x.data<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.data<short>());
            Assert.Throws<System.ArgumentException>(() => x.data<int>());
            Assert.Throws<System.ArgumentException>(() => x.data<long>());
            Assert.Throws<System.ArgumentException>(() => x.data<float>());
            Assert.Throws<System.ArgumentException>(() => x.data<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.data<System.Numerics.Complex>());
            x.data<double>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemBool()
        {
            var x = torch.ones(1, torch.@bool);
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<bool>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemByte()
        {
            var x = torch.ones(1, torch.uint8);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<byte>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemInt8()
        {
            var x = torch.ones(1, torch.int8);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<sbyte>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemInt16()
        {
            var x = torch.ones(1, torch.int16);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<short>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemInt32()
        {
            var x = torch.ones(1, torch.int32);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<int>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemInt64()
        {
            var x = torch.ones(1, torch.int64);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<long>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemFloat32()
        {
            var x = torch.ones(1, torch.float32);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<double>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<float>();
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void DataItemFloat64()
        {
            var x = torch.ones(1, torch.float64);
            Assert.Throws<System.ArgumentException>(() => x.item<bool>());
            Assert.Throws<System.ArgumentException>(() => x.item<byte>());
            Assert.Throws<System.ArgumentException>(() => x.item<sbyte>());
            Assert.Throws<System.ArgumentException>(() => x.item<short>());
            Assert.Throws<System.ArgumentException>(() => x.item<int>());
            Assert.Throws<System.ArgumentException>(() => x.item<long>());
            Assert.Throws<System.ArgumentException>(() => x.item<float>());
            Assert.Throws<System.ArgumentException>(() => x.item<(float, float)>());
            Assert.Throws<System.ArgumentException>(() => x.item<System.Numerics.Complex>());
            x.item<double>();
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void FromArrayFactory()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                {
                    var array = new bool[8];
                    using var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Bool, t.dtype));
                }

                {
                    var array = new Memory<byte>(new byte[8]);
                    using var t = torch.tensor(array, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Byte, t.dtype));
                }

                {
                    var array = new Memory<long>(new long[8]);
                    using var t = torch.tensor(array, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Int64, t.dtype));
                }

                {
                    var array = new long[18];
                    array[5] = 17;
                    var mem = new Memory<long>(array, 4, 10);
                    using var t = torch.tensor(mem, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(8, t.numel()),
                        () => Assert.Equal(17, t[1].item<long>()),
                        () => Assert.Equal(ScalarType.Int64, t.dtype));
                }

                {
                    var array = new bool[8];
                    using var t = torch.tensor(array, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Bool, t.dtype));
                }

                {
                    var array = new bool[18]; // Too long, on purpose
                    using var t = torch.tensor(array, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(8, t.NumberOfElements),
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Bool, t.dtype));
                }

                {
                    var array = new int[8];
                    using var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Int32, t.dtype));
                }

                {
                    var array = new float[8];
                    using var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Float32, t.dtype));
                }

                {
                    var array = new float[18]; // Too long, on purpose
                    using var t = torch.tensor(array, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(8, t.NumberOfElements),
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Float32, t.dtype));
                }

                {
                    var array = new double[1, 2];
                    using var t = torch.from_array(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(2, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2 }, t.shape),
                        () => Assert.Equal(ScalarType.Float64, t.dtype));
                }

                {
                    var array = new long[1, 2, 3];
                    using var t = torch.from_array(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(3, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2, 3 }, t.shape),
                        () => Assert.Equal(ScalarType.Int64, t.dtype));
                }

                {
                    var array = new int[1, 2, 3, 4];
                    using var t = torch.from_array(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(4, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape),
                        () => Assert.Equal(ScalarType.Int32, t.dtype));
                }

                {
                    var array = new System.Numerics.Complex[1, 2, 3, 4];
                    using var t = torch.from_array(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(4, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape),
                        () => Assert.Equal(ScalarType.ComplexFloat64, t.dtype));
                }

                {
                    var array = new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                    using var t = torch.from_array(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(3, t.ndim),
                        () => Assert.Equal(new long[] { 2, 2, 2 }, t.shape),
                        () => Assert.Equal(ScalarType.Float64, t.dtype),
                        () => Assert.Equal(array.Cast<double>().ToArray(), t.data<double>().ToArray()));
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.frombuffer))]
        public void FromBufferFactory()
        {
            // Note: frombuffer cannot create tensors on other devices than CPU,
            // since the memory cannot be shared.
            {
                var array = new double[8];
                var t = torch.frombuffer(array, ScalarType.Bool);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(8, t.shape[0]),
                    () => Assert.Equal(ScalarType.Bool, t.dtype));
            }
            {
                var array = new double[8];
                var t = torch.frombuffer(array, ScalarType.Bool, 5, 2);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(5, t.shape[0]),
                    () => Assert.Equal(ScalarType.Bool, t.dtype));
            }
            {
                // The number of input dimensions shouldn't matter.
                var array = new double[2, 4];
                var t = torch.frombuffer(array, ScalarType.Bool, 5, 2);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(5, t.shape[0]),
                    () => Assert.Equal(ScalarType.Bool, t.dtype));
            }
            {
                var array = new double[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                var t = torch.frombuffer(array, ScalarType.Float64, 5, 2);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(5, t.shape[0]),
                    () => Assert.Equal(3, t[1].item<double>()),
                    () => Assert.Equal(6, t[4].item<double>()),
                    () => Assert.Equal(ScalarType.Float64, t.dtype));
            }
            {
                var array = new double[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                var t = torch.frombuffer(array, ScalarType.Float64, 5, 2);
                t[4] = 15.0;
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(5, t.shape[0]),
                    () => Assert.Equal(3, t[1].item<double>()),
                    () => Assert.Equal(15, array[6]),
                    () => Assert.Equal(ScalarType.Float64, t.dtype));
            }
            {
                var array = new System.Numerics.Complex[8];
                for (var i = 0; i < array.Length; i++) { array[i] = new System.Numerics.Complex(i, -i); }
                var t = torch.frombuffer(array, ScalarType.ComplexFloat64, 5, 2);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(5, t.shape[0]),
                    () => Assert.Equal(3, t.real[1].item<double>()),
                    () => Assert.Equal(6, t.real[4].item<double>()),
                    () => Assert.Equal(-3, t.imag[1].item<double>()),
                    () => Assert.Equal(-6, t.imag[4].item<double>()),
                    () => Assert.Equal(ScalarType.ComplexFloat64, t.dtype));
            }
            {
                var array = new System.Numerics.Complex[8];
                for (var i = 0; i < array.Length; i++) { array[i] = new System.Numerics.Complex(i, -i); }
                var t = torch.frombuffer(array, ScalarType.ComplexFloat64, 5, 2);
                t.real[4] = 15.0;
                t.imag[4] = 25.0;
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(5, t.shape[0]),
                    () => Assert.Equal(3, t.real[1].item<double>()),
                    () => Assert.Equal(15, array[6].Real),
                    () => Assert.Equal(-3, t.imag[1].item<double>()),
                    () => Assert.Equal(25.0, array[6].Imaginary),
                    () => Assert.Equal(ScalarType.ComplexFloat64, t.dtype));
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactorySByte()
        {
            foreach (var device in TestUtils.AvailableDevices()) {
                {
                    var array = new sbyte[8];
                    var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Int8, t.dtype));
                }

                {
                    var array = new sbyte[8];
                    var t = torch.tensor(array, new long[] { 8 }, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(1, t.ndim),
                        () => Assert.Equal(ScalarType.Int8, t.dtype));
                }

                {
                    var array = new sbyte[1, 2];
                    var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(2, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2 }, t.shape),
                        () => Assert.Equal(ScalarType.Int8, t.dtype));
                }

                {
                    var array = new sbyte[1, 2, 3];
                    var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(3, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2, 3 }, t.shape),
                        () => Assert.Equal(ScalarType.Int8, t.dtype));
                }

                {
                    var array = new sbyte[1, 2, 3, 4];
                    var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(4, t.ndim),
                        () => Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape),
                        () => Assert.Equal(ScalarType.Int8, t.dtype));
                }

                {
                    var array = new sbyte[100, 100, 100];
                    var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(3, t.ndim),
                        () => Assert.Equal(new long[] { 100, 100, 100 }, t.shape),
                        () => Assert.Equal(ScalarType.Int8, t.dtype));
                }

                {
                    var array = new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                    var t = torch.tensor(array, device: device);
                    Assert.Multiple(
                        () => Assert.Equal(device.type, t.device_type),
                        () => Assert.Equal(3, t.ndim),
                        () => Assert.Equal(new long[] { 2, 2, 2 }, t.shape),
                        () => Assert.Equal(ScalarType.Int8, t.dtype),
                        () => Assert.Equal(array.Cast<sbyte>().ToArray(), t.data<sbyte>().ToArray()));
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryInt16()
        {
            {
                var array = new short[8];
                using var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Int16, t.dtype);
            }

            {
                var array = new short[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Int16, t.dtype);
            }

            {
                var array = new short[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.Int16, t.dtype);
            }

            {
                var array = new short[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.Int16, t.dtype);
            }

            {
                var array = new short[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.Int16, t.dtype);
            }

            {
                var array = new short[100, 100, 500];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 500 }, t.shape);
                Assert.Equal(ScalarType.Int16, t.dtype);
            }
            {
                var array = new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.Int16, t.dtype);
                Assert.Equal(array.Cast<short>().ToArray(), t.data<short>().ToArray());
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryInt32()
        {
            {
                var array = new int[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Int32, t.dtype);
            }

            {
                var array = new int[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Int32, t.dtype);
            }

            {
                var array = new int[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.Int32, t.dtype);
            }

            {
                var array = new int[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.Int32, t.dtype);
            }

            {
                var array = new int[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.Int32, t.dtype);
            }

            {
                var array = new int[100, 100, 250];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 250 }, t.shape);
                Assert.Equal(ScalarType.Int32, t.dtype);
            }

            {
                var array = new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.Int32, t.dtype);
                Assert.Equal(array.Cast<int>().ToArray(), t.data<int>().ToArray());
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryInt64()
        {
            {
                var array = new long[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[8];
                var t = torch.tensor(array, new long[] { 2, 4 });
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 2, 4 }, t.shape);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[8];
                Assert.Throws<ArgumentException>(() => torch.tensor(array, new long[] { 3, 4 }));
            }

            {
                var array = new long[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[100, 100, 125];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 125 }, t.shape);
                Assert.Equal(ScalarType.Int64, t.dtype);
            }

            {
                var array = new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.Int64, t.dtype);
                Assert.Equal(array.Cast<long>().ToArray(), t.data<long>().ToArray());
            }
        }

#if NET6_0_OR_GREATER
        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryFloat16()
        {
            {
                var array = new Half[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Float16, t.dtype);
            }

            {
                var array = new Half[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Float16, t.dtype);
            }

            {
                var array = new Half[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.Float16, t.dtype);
            }

            {
                var array = new Half[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.Float16, t.dtype);
            }

            {
                var array = new Half[100, 100, 250];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 250 }, t.shape);
                Assert.Equal(ScalarType.Float16, t.dtype);
            }

            {
                var array = new Half[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.Float16, t.dtype);
            }

            {
                var array = new Half[,,] { { { (Half)1, (Half)2 }, { (Half)3, (Half)4 } }, { { (Half)5, (Half)6 }, { (Half)7, (Half)8 } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.Float16, t.dtype);
                Assert.Equal(array.Cast<Half>().ToArray(), t.data<Half>().ToArray());
            }
        }
#endif

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryFloat32()
        {
            {
                var array = new float[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Float32, t.dtype);
            }

            {
                var array = new float[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Float32, t.dtype);
            }

            {
                var array = new float[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.Float32, t.dtype);
            }

            {
                var array = new float[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.Float32, t.dtype);
            }

            {
                var array = new float[100, 100, 250];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 250 }, t.shape);
                Assert.Equal(ScalarType.Float32, t.dtype);
            }

            {
                var array = new float[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.Float32, t.dtype);
            }

            {
                var array = new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.Float32, t.dtype);
                Assert.Equal(array.Cast<float>().ToArray(), t.data<float>().ToArray());
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryFloat64()
        {
            {
                var array = new double[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Float64, t.dtype);
            }

            {
                var array = new double[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.Float64, t.dtype);
            }

            {
                var array = new double[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.Float64, t.dtype);
            }

            {
                var array = new double[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.Float64, t.dtype);
            }

            {
                var array = new double[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.Float64, t.dtype);
            }

            {
                var array = new double[100, 100, 125];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 125 }, t.shape);
                Assert.Equal(ScalarType.Float64, t.dtype);
            }

            {
                var array = new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 2, 2, 2 }, t.shape);
                Assert.Equal(ScalarType.Float64, t.dtype);
                Assert.Equal(array.Cast<double>().ToArray(), t.data<double>().ToArray());
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryComplexFloat32()
        {
            {
                var array = new (float, float)[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.ComplexFloat32, t.dtype);
            }

            {
                var array = new (float, float)[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.ComplexFloat32, t.dtype);
            }

            {
                var array = new (float, float)[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat32, t.dtype);
            }

            {
                var array = new (float, float)[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat32, t.dtype);
            }

            {
                var array = new (float, float)[100, 100, 250];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 100, 250 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat32, t.dtype);
            }

            {
                var array = new (float, float)[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat32, t.dtype);
            }
            {
                var array = new (float Real, float Imaginary)[8];
                for (var i = 0; i < array.Length; i++) { array[i] = (i, -i); }
                var t = torch.tensor(array);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(8, t.shape[0]),
                    () => Assert.Equal(3, t.real[3].item<float>()),
                    () => Assert.Equal(6, t.real[6].item<float>()),
                    () => Assert.Equal(-3, t.imag[3].item<float>()),
                    () => Assert.Equal(-6, t.imag[6].item<float>()),
                    () => Assert.Equal(ScalarType.ComplexFloat32, t.dtype));
            }
            {
                var array = new (float Real, float Imaginary)[8];
                for (var i = 0; i < array.Length; i++) { array[i] = (i, -i); }
                var t = torch.tensor(array);
                t.real[6] = 17;
                t.imag[6] = 15;
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(8, t.shape[0]),
                    () => Assert.Equal(3, t.real[3].item<float>()),
                    () => Assert.Equal(6, array[6].Real),
                    () => Assert.Equal(-3, t.imag[3].item<float>()),
                    () => Assert.Equal(-6, array[6].Imaginary),
                    () => Assert.Equal(ScalarType.ComplexFloat32, t.dtype));
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void MDTensorFactoryComplexFloat64()
        {
            {
                var array = new System.Numerics.Complex[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);
            }

            {
                var array = new System.Numerics.Complex[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);
            }
            {
                var array = new System.Numerics.Complex[8];
                var t = torch.tensor(array);
                Assert.Equal(1, t.ndim);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);

                var s = t.reshape(2, 4);
                Assert.Multiple(
                    () => Assert.Equal(2, s.ndim),
                    () => Assert.Equal(2, s.shape[0]),
                    () => Assert.Equal(4, s.shape[1]));
            }

            {
                var array = new System.Numerics.Complex[1, 2];
                var t = torch.tensor(array);
                Assert.Equal(2, t.ndim);
                Assert.Equal(new long[] { 1, 2 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);
            }

            {
                var array = new System.Numerics.Complex[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);
            }

            {
                var array = new System.Numerics.Complex[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Equal(4, t.ndim);
                Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);
            }

            {
                var array = new System.Numerics.Complex[100, 500, 125];
                var t = torch.tensor(array);
                Assert.Equal(3, t.ndim);
                Assert.Equal(new long[] { 100, 500, 125 }, t.shape);
                Assert.Equal(ScalarType.ComplexFloat64, t.dtype);
            }
            {
                var array = new System.Numerics.Complex[8];
                for (var i = 0; i < array.Length; i++) { array[i] = new System.Numerics.Complex(i, -i); }
                var t = torch.tensor(array);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(8, t.shape[0]),
                    () => Assert.Equal(3, t.real[3].item<double>()),
                    () => Assert.Equal(6, t.real[6].item<double>()),
                    () => Assert.Equal(-3, t.imag[3].item<double>()),
                    () => Assert.Equal(-6, t.imag[6].item<double>()),
                    () => Assert.Equal(ScalarType.ComplexFloat64, t.dtype));
            }
            {
                var array = new System.Numerics.Complex[8];
                for (var i = 0; i < array.Length; i++) { array[i] = new System.Numerics.Complex(i, -i); }
                var t = torch.tensor(array);
                t.real[6] = 17;
                t.imag[6] = 15;
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(8, t.shape[0]),
                    () => Assert.Equal(3, t.real[3].item<double>()),
                    () => Assert.Equal(6, array[6].Real),
                    () => Assert.Equal(-6, array[6].Imaginary),
                    () => Assert.Equal(ScalarType.ComplexFloat64, t.dtype));
            }
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat32TensorZeros()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0.0f, t[0, 0].ToSingle());
            Assert.Equal(0.0f, t[1, 1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat32TensorZeros_()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0.0f, t[0, 0].ToSingle());
            Assert.Equal(0.0f, t[1, 1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateByteTensorZeros()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, torch.uint8);
            Assert.Equal(shape, t.shape);
            Assert.Equal((byte)0, t[0, 0].ToByte());
            Assert.Equal((byte)0, t[1, 1].ToByte());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateByteTensorZeros_()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, torch.uint8);
            Assert.Equal(shape, t.shape);
            Assert.Equal((byte)0, t[0, 0].ToByte());
            Assert.Equal((byte)0, t[1, 1].ToByte());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateInt32TensorZeros()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, torch.int32);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0, t[0, 0].ToInt32());
            Assert.Equal(0, t[1, 1].ToInt32());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateInt32TensorZeros_()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, torch.int32);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0, t[0, 0].ToInt32());
            Assert.Equal(0, t[1, 1].ToInt32());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateInt64TensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.zeros(shape, torch.int64);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0L, t[0, 0].ToInt64());
            Assert.Equal(0L, t[1, 1].ToInt64());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateBoolTensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.zeros(shape, torch.@bool);
            Assert.Equal(shape, t.shape);
            Assert.Equal((object)false, t[0, 0].ToBoolean());
            Assert.Equal((object)false, t[1, 1].ToBoolean());
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat16TensorZeros()
        {
            if (torch.cuda.is_available()) {
                var shape = new long[] { 2, 2 };

                Tensor t = torch.zeros(shape, device: torch.CUDA, dtype: torch.float16);
                Assert.Equal(shape, t.shape);
                Assert.Equal(0.0f, t[0, 0].ToSingle());
                Assert.Equal(0.0f, t[1, 1].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateBFloat16TensorZeros()
        {
            if (torch.cuda.is_available()) {
                var shape = new long[] { 2, 2 };

                Tensor t = torch.zeros(shape, device: torch.CUDA, dtype: torch.bfloat16);
                Assert.Equal(shape, t.shape);
                Assert.Equal(0.0f, t[0, 0].ToSingle());
                Assert.Equal(0.0f, t[1, 1].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateFloat32TensorEmpty()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.empty(shape);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat32TensorZerosWithNames()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, names: new[] { "N", "C" });
            Assert.Equal(shape, t.shape);
            Assert.Equal(0.0f, t[0, 0].ToSingle());
            Assert.Equal(0.0f, t[1, 1].ToSingle());

            Assert.True(t.has_names());

            var names = t.names.ToArray();
            Assert.Equal(new[] { "N", "C" }, names);

            var s = t + 1;
            Assert.True(s.has_names());

            names = s.names.ToArray();
            Assert.Equal(new[] { "N", "C" }, names);

            t = t.rename_(null);
            //Assert.False(t.has_names());  FAILS ON RELEASE BUILDS

            t = t.rename_(new[] { "Batch", "Channels" });
            Assert.True(t.has_names());
            Assert.Equal(new[] { "Batch", "Channels" }, t.names.ToArray());

            Assert.Throws<ArgumentException>(() => t.rename_(new[] { "N" }));
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat32TensorOnesWithNames()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.ones(shape, names: new[] { "N", "C" });
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 1].ToSingle());

            Assert.True(t.has_names());

            var names = t.names.ToArray();
            Assert.Equal(new[] { "N", "C" }, names);

            var s = t + 1;
            Assert.True(s.has_names());

            names = s.names.ToArray();
            Assert.Equal(new[] { "N", "C" }, names);

            t = t.rename_(null);
            //Assert.False(t.has_names());  FAILS ON RELEASE BUILDS

            t.rename_(new[] { "Batch", "Channels" });
            Assert.True(t.has_names());
            Assert.Equal(new[] { "Batch", "Channels" }, t.names.ToArray());

            Assert.Throws<ArgumentException>(() => t.rename_(new[] { "N" }));
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat32TensorFullWithNames()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.full(shape, 2.0f, names: new[] { "N", "C" });
            Assert.Equal(shape, t.shape);
            Assert.Equal(2.0f, t[0, 0].ToSingle());
            Assert.Equal(2.0f, t[1, 1].ToSingle());

            Assert.True(t.has_names());

            var names = t.names.ToArray();
            Assert.Equal(new[] { "N", "C" }, names);

            var s = t + 1;
            Assert.True(s.has_names());

            names = s.names.ToArray();
            Assert.Equal(new[] { "N", "C" }, names);

            var z = t.rename_(null);
            Assert.Same(t, z);
            //Assert.False(t.has_names());  FAILS ON RELEASE BUILDS

            z = t.rename_(new[] { "Batch", "Channels" });
            Assert.Same(t, z);
            Assert.True(t.has_names());
            Assert.Equal(new[] { "Batch", "Channels" }, t.names.ToArray());
        }

        [Fact]
        [TestOf(nameof(torch.full))]
        public void CreateFloat32TensorFull()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.full(2, 2, 3.14f);
            Assert.Multiple(
                () => Assert.Equal(shape, t.shape),
                () => Assert.Equal(torch.float32, t.dtype),
                () => Assert.Equal(3.14f, t[0, 0].ToSingle()),
                () => Assert.Equal(3.14f, t[1, 1].ToSingle())
            );

            var u = t.new_full(shape, 3.14f);
            Assert.Multiple(
                () => Assert.Equal(shape, u.shape),
                () => Assert.Equal(torch.float32, u.dtype),
                () => Assert.Equal(3.14f, u[0, 0].ToSingle()),
                () => Assert.Equal(3.14f, u[1, 1].ToSingle())
            );
        }

        [Fact]
        [TestOf(nameof(torch.full))]
        public void CreateFloat64TensorFull()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.full(2, 2, 3.14f, dtype: float64);
            Assert.Multiple(
                () => Assert.Equal(shape, t.shape),
                () => Assert.Equal(torch.float64, t.dtype),
                () => Assert.Equal(3.14f, t[0, 0].ToSingle()),
                () => Assert.Equal(3.14f, t[1, 1].ToSingle())
            );

            var u = t.new_full(shape, 3.14f);
            Assert.Multiple(
                () => Assert.Equal(shape, u.shape),
                () => Assert.Equal(torch.float64, u.dtype),
                () => Assert.Equal(3.14f, u[0, 0].ToSingle()),
                () => Assert.Equal(3.14f, u[1, 1].ToSingle())
            );
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateByteTensorEmpty()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.empty(shape, torch.uint8);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateInt32TensorEmpty()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.empty(shape, torch.int32);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        [TestOf(nameof(torch.full))]
        public void CreateInt32TensorFull()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.full(2, 2, 17, torch.int32);
            Assert.Equal(shape, t.shape);
            Assert.Equal(17, t[0, 0].ToInt32());
            Assert.Equal(17, t[1, 1].ToInt32());
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateInt64TensorEmpty()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.empty(shape, torch.int64);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateBoolTensorEmpty()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.empty(shape, @bool);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        [TestOf(nameof(torch.empty_strided))]
        public void CreateTensorEmptyStrided()
        {
            var shape = new long[] { 2, 3 };
            var strides = new long[] { 1, 2 };

            Tensor t = torch.empty_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
            t = torch.empty_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
        }

        [Fact]
        [TestOf(nameof(Tensor.as_strided))]
        public void CreateTensorEmptyAsStrided()
        {
            var shape = new long[] { 2, 3 };
            var strides = new long[] { 1, 2 };

            Tensor t = torch.empty(shape).as_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
            t = torch.empty(shape).as_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateFloat16TensorEmpty()
        {
            if (torch.cuda.is_available()) {
                var shape = new long[] { 2, 2 };

                Tensor t = torch.empty(shape, float16, device: torch.CUDA);
                Assert.Equal(shape, t.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateBFloat16TensorEmpty()
        {
            if (torch.cuda.is_available()) {
                var shape = new long[] { 2, 2 };

                Tensor t = torch.empty(shape, bfloat16, device: torch.CUDA);
                Assert.Equal(shape, t.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torch.linspace))]
        public void CreateFloat32Linspace()
        {
            Tensor t = torch.linspace(0.0f, 10.0f, 101);
            Assert.Equal(101, t.shape[0]);
            Assert.Equal(0.0f, t[0].ToSingle());
            Assert.Equal(0.1f, t[1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.logspace))]
        public void CreateFloat32Logspace()
        {
            Tensor t = torch.logspace(0.0f, 10.0f, 101);
            Assert.Equal(101, t.shape[0]);
            Assert.Equal(1.0f, t[0].ToSingle());
            Assert.Equal(10.0f, t[10].ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateFloat32TensorOnes()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateByteTensorOnes()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.ones(shape, uint8);
            Assert.Equal(shape, t.shape);
            Assert.Equal((byte)1, t[0, 0].ToByte());
            Assert.Equal((byte)1, t[1, 1].ToByte());
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateInt32TensorOnes()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.ones(shape, int32);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t[0, 0].ToInt32());
            Assert.Equal(1, t[1, 1].ToInt32());
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateInt64TensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.ones(shape, int64);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1L, t[0, 0].ToInt64());
            Assert.Equal(1L, t[1, 1].ToInt64());
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateBoolTensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.ones(shape, @bool);
            Assert.Equal(shape, t.shape);
            Assert.Equal((object)true, t[0, 0].ToBoolean());
            Assert.Equal((object)true, t[1, 1].ToBoolean());
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateFloat16TensorOnes()
        {
            if (torch.cuda.is_available()) {
                var shape = new long[] { 2, 2 };

                Tensor t = torch.ones(shape, float16, device: torch.CUDA);
                Assert.Equal(shape, t.shape);
                Assert.Equal(1.0f, t[0, 0].ToSingle());
                Assert.Equal(1.0f, t[1, 1].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateBFloat16TensorOnes()
        {
            if (torch.cuda.is_available()) {
                var shape = new long[] { 2, 2 };

                Tensor t = torch.ones(shape, bfloat16, device: torch.CUDA);
                Assert.Equal(shape, t.shape);
                Assert.Equal(1.0f, t[0, 0].ToSingle());
                Assert.Equal(1.0f, t[1, 1].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateComplexFloat32TensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.zeros(shape, torch.complex64);
            Assert.Equal(shape, t.shape);
            var v3 = t.data<(float Real, float Imaginary)>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(0.0f, v3[i].Real);
                Assert.Equal(0.0f, v3[i].Imaginary);
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateComplexFloat32TensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.ones(shape, torch.complex64);
            Assert.Equal(shape, t.shape);
            var v3 = t.data<(float Real, float Imaginary)>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(1.0f, v3[i].Real);
                Assert.Equal(0.0f, v3[i].Imaginary);
            }
        }

        [Fact]
        [TestOf(nameof(torch.complex))]
        public void CreateComplex32()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = torch.randn(shape);
            Tensor i = torch.randn(shape);

            Tensor x = torch.complex(r, i);

            Assert.Equal(shape, x.shape);
            Assert.Equal(ScalarType.ComplexFloat32, x.dtype);
            Assert.True(r.allclose(x.real));
            Assert.True(i.allclose(x.imag));
        }

        [Fact]
        [TestOf(nameof(torch.complex))]
        public void CreateComplex64()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = torch.randn(shape, torch.float64);
            Tensor i = torch.randn(shape, torch.float64);

            Tensor x = torch.complex(r, i);

            Assert.Equal(shape, x.shape);
            Assert.Equal(ScalarType.ComplexFloat64, x.dtype);
            Assert.True(r.allclose(x.real));
            Assert.True(i.allclose(x.imag));
        }

        [Fact]
        [TestOf(nameof(torch.polar))]
        public void CreatePolar32()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = torch.randn(shape);
            Tensor i = torch.randn(shape);

            Tensor x = torch.complex(r, i);

            Tensor p = torch.polar(x.abs(), x.angle());

            Assert.Equal(x.shape, p.shape);
            Assert.Equal(ScalarType.ComplexFloat32, p.dtype);
            Assert.True(r.allclose(p.real, rtol: 1e-04, atol: 1e-07));
            Assert.True(i.allclose(p.imag, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(torch.polar))]
        public void CreatePolar64()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = torch.randn(shape, torch.float64);
            Tensor i = torch.randn(shape, torch.float64);

            Tensor x = torch.complex(r, i);

            Tensor p = torch.polar(x.abs(), x.angle());

            Assert.Equal(x.shape, p.shape);
            Assert.Equal(ScalarType.ComplexFloat64, p.dtype);
            Assert.True(r.allclose(p.real, rtol: 1e-04, atol: 1e-07));
            Assert.True(i.allclose(p.imag, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(torch.rand))]
        public void CreateComplexFloat32TensorRand()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.rand(shape, torch.complex64);
            Assert.Equal(shape, t.shape);
            var v3 = t.data<(float Real, float Imaginary)>().ToArray();
            Assert.All(v3, t => Assert.True(t.Real >= 0.0f && t.Real < 1.0f && t.Imaginary >= 0.0f && t.Imaginary < 1.0f));
        }

        [Fact]
        [TestOf(nameof(torch.randn))]
        public void CreateComplexFloat32TensorRandn()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.randn(shape, torch.complex64);
            Assert.Equal(shape, t.shape);
            var v3 = t.data<(float Real, float Imaginary)>().ToArray();
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateComplexFloat64TensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.zeros(shape, complex128);
            Assert.Equal(shape, t.shape);
            var v3 = t.data<System.Numerics.Complex>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(0.0, v3[i].Real);
                Assert.Equal(0.0, v3[i].Imaginary);
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateComplexFloat64TensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.ones(shape, complex128);
            Assert.Equal(shape, t.shape);
            var v3 = t.data<System.Numerics.Complex>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(1.0, v3[i].Real);
                Assert.Equal(0.0, v3[i].Imaginary);
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateFloat32TensorCheckMemory()
        {
            for (int i = 0; i < 10; i++) {
                using var tmp = torch.ones(new long[] { 100, 100, 100 });
                Assert.NotNull(tmp);
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateFloat32TensorOnesCheckData()
        {
            var ones = torch.ones(new long[] { 2, 2 });
            var data = ones.data<float>();

            for (int i = 0; i < 4; i++) {
                Assert.Equal(1.0, data[i]);
            }
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateFloat32TensorZerosCheckData()
        {
            var zeros = torch.zeros(new long[] { 2, 2 });
            var data = zeros.data<float>();

            for (int i = 0; i < 4; i++) {
                Assert.Equal(0, data[i]);
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateInt32TensorOnesCheckData()
        {
            var ones = torch.ones(new long[] { 2, 2 }, int32);
            var data = ones.data<int>();

            for (int i = 0; i < 4; i++) {
                Assert.Equal(1, data[i]);
            }
        }

        [Fact]
        [TestOf(nameof(torch.eye))]
        public void CreateInt32TensorEyeCheckData1()
        {
            var ones = torch.eye(4, 4, int32);
            Assert.Equal(ones.shape[0], ones.shape[1]);

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j)
                        Assert.Equal(1, ones[i, j].ToInt32());
                    else
                        Assert.Equal(0, ones[i, j].ToInt32());
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.eye))]
        public void CreateInt32TensorEyeCheckData2()
        {
            var ones = torch.eye(4, dtype: torch.int32);
            Assert.Equal(ones.shape[0], ones.shape[1]);

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j)
                        Assert.Equal(1, ones[i, j].ToInt32());
                    else
                        Assert.Equal(0, ones[i, j].ToInt32());
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.eye))]
        public void CreateComplexFloat32TensorEyeCheckData()
        {
            var ones = torch.eye(4, 4, torch.complex64);
            Assert.Equal(ones.shape[0], ones.shape[1]);

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j) {
                        var scalar = ones[i, j].ToComplex32();
                        Assert.Equal(1.0f, scalar.Real);
                        Assert.Equal(0.0f, scalar.Imaginary);
                    } else {
                        var scalar = ones[i, j].ToComplex32();
                        Assert.Equal(0.0f, scalar.Real);
                        Assert.Equal(0.0f, scalar.Imaginary);
                    }
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.eye))]
        public void CreateComplexFloat64TensorEyeCheckData()
        {
            var ones = torch.eye(4, 4, torch.complex128);
            Assert.Equal(ones.shape[0], ones.shape[1]);

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j) {
                        var scalar = ones[i, j].ToComplex64();
                        Assert.Equal(1.0, scalar.Real);
                        Assert.Equal(0.0, scalar.Imaginary);
                    } else {
                        var scalar = ones[i, j].ToComplex64();
                        Assert.Equal(0.0, scalar.Real);
                        Assert.Equal(0.0, scalar.Imaginary);
                    }
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateFloat32TensorCheckDevice()
        {
            var ones = torch.ones(new long[] { 2, 2 });
            var device = ones.device;

            Assert.Equal("cpu", ones.device.ToString());
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateFloat32TensorFromData()
        {
            var data = new float[1000];
            data[100] = 1;

            using var tensor = torch.tensor(data, new long[] { 100, 10 });
            Assert.Equal(1, tensor.data<float>()[100]);
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateFloat32TensorFromDataCheckDispose()
        {
            var data = new float[1000];
            data[100] = 1;

            using var tensor = torch.tensor(data, new long[] { 100, 10 });
            Assert.Equal(1, tensor.data<float>()[100]);
            Assert.Equal(1, data[100]);
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.ToTensor))]
        public void CreateFloat32TensorFromData2()
        {
            var data = new float[1000];

            using var tensor = data.ToTensor(new long[] { 10, 100 });
            Assert.Equal(default(float), tensor.data<float>()[100]);
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.ToTensor))]
        public void CreateFloat32TensorFromDataCheckStrides()
        {
            var data = new double[] { 0.2663158, 0.1144736, 0.1147367, 0.1249998, 0.1957895, 0.1231576, 0.1944732, 0.111842, 0.1065789, 0.667881, 0.5682123, 0.5824502, 0.4824504, 0.4844371, 0.6463582, 0.5334439, 0.5079474, 0.2281452 };
            var dataTensor = data.ToTensor(new long[] { 2, 9 });

            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < 9; i++) {
                    var fromData = data[(r * 9) + i];
                    var fromTensor = dataTensor[r, i].ToDouble();
                    Assert.True(Math.Abs(fromData - fromTensor) < 0.0001);
                }
            }

            var firstPart = dataTensor[0];

            for (int i = 0; i < 9; i++) {
                var fromData = data[i];
                var fromChunk = firstPart[i].ToDouble();
                Assert.True(Math.Abs(fromData - fromChunk) < 0.0001);
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateFloat16TensorFromDataCheckStrides()
        {
            var data = new float[] { 0.2663158f, 0.1144736f, 0.1147367f, 0.1249998f, 0.1957895f, 0.1231576f, 0.1944732f, 0.111842f, 0.1065789f, 0.667881f, 0.5682123f, 0.5824502f, 0.4824504f, 0.4844371f, 0.6463582f, 0.5334439f, 0.5079474f, 0.2281452f };
            var dataTensor = torch.tensor(data, new long[] { 2, 9 }, float16);

            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < 9; i++) {
                    var fromData = data[(r * 9) + i];
                    var fromTensor = dataTensor[r, i].ToSingle();
                    Assert.True(Math.Abs(fromData - fromTensor) < 0.01);
                }
            }

            var firstPart = dataTensor[0];

            for (int i = 0; i < 9; i++) {
                var fromData = data[i];
                var fromChunk = firstPart[i].ToSingle();
                Assert.True(Math.Abs(fromData - fromChunk) < 0.01);
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateBFloat16TensorFromDataCheckStrides()
        {
            var data = new float[] { 0.2663158f, 0.1144736f, 0.1147367f, 0.1249998f, 0.1957895f, 0.1231576f, 0.1944732f, 0.111842f, 0.1065789f, 0.667881f, 0.5682123f, 0.5824502f, 0.4824504f, 0.4844371f, 0.6463582f, 0.5334439f, 0.5079474f, 0.2281452f };
            var dataTensor = torch.tensor(data, new long[] { 2, 9 }, bfloat16);

            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < 9; i++) {
                    var fromData = data[(r * 9) + i];
                    var fromTensor = dataTensor[r, i].ToSingle();
                    Assert.True(Math.Abs(fromData - fromTensor) < 0.1);
                }
            }

            var firstPart = dataTensor[0];

            for (int i = 0; i < 9; i++) {
                var fromData = data[i];
                var fromChunk = firstPart[i].ToSingle();
                Assert.True(Math.Abs(fromData - fromChunk) < 0.1);
            }
        }

        [Fact]
        [TestOf(nameof(torch.bartlett_window))]
        public void CreateFloat32BartlettWindow()
        {
            Tensor t = torch.bartlett_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.blackman_window))]
        public void CreateFloat32BlackmanWindow()
        {
            Tensor t = torch.blackman_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.hamming_window))]
        public void CreateFloat32HammingWindow()
        {
            Tensor t = torch.hamming_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.hann_window))]
        public void CreateFloat32HannWindow()
        {
            Tensor t = torch.hann_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.kaiser_window))]
        public void CreateFloat32KaiserWindow()
        {
            Tensor t = torch.kaiser_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateFloat32TensorFromScalar()
        {
            float scalar = 333.0f;

            using var tensor = torch.tensor(scalar);
            Assert.Equal(333.0f, tensor.ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateFloat16TensorFromScalar()
        {
            float scalar = 333.0f;

            using var tensor = torch.tensor(scalar, float16);
            Assert.Equal(333.0f, tensor.ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void CreateBFloat16TensorFromScalar()
        {
            float scalar = 333.0f;
            using var tensor = torch.tensor(scalar, bfloat16);
            Assert.Equal(332.0f, tensor.ToSingle()); // NOTE: bfloat16 loses precision, this really is 332.0f
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.ToTensor))]
        public void CreateFloat32TensorFromScalar2()
        {
            float scalar = 333.0f;
            using var tensor = scalar.ToTensor();
            Assert.Equal(333, tensor.ToSingle());
        }

        [Fact]
        [TestOf(nameof(torch.ones_like))]
        public void CreateFloat32TensorOnesLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.ones_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.data<float>().ToArray(), t => Assert.True(t == 1.0f));
        }

        [Fact]
        [TestOf(nameof(torch.ones_like))]
        public void CreateFloat32TensorOnesLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.ones_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.data<double>().ToArray(), t => Assert.True(t == 1.0));
        }

        [Fact]
        [TestOf(nameof(torch.zeros_like))]
        public void CreateFloat32TensorZerosLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.zeros_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.data<float>().ToArray(), t => Assert.True(t == 0.0f));
        }

        [Fact]
        [TestOf(nameof(torch.zeros_like))]
        public void CreateFloat32TensorZerosLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.zeros_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.data<double>().ToArray(), t => Assert.True(t == 0.0));
        }

        [Fact]
        [TestOf(nameof(torch.empty_like))]
        public void CreateFloat32TensorEmptyLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.empty_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.empty_like))]
        public void CreateFloat32TensorEmptyLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.empty_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.full_like))]
        public void CreateFloat32TensorFullLike()
        {
            var shape = new long[] { 10, 20, 30 };
            Scalar value = 3.14f;

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.full_like(value);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.data<float>().ToArray(), t => Assert.True(t == 3.14f));
        }

        [Fact]
        [TestOf(nameof(torch.full_like))]
        public void CreateFloat32TensorFullLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };
            Scalar value = 3.14;

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.full_like(value, dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.data<double>().ToArray(), t => Assert.True(t == 3.14));
        }

        [Fact]
        [TestOf(nameof(torch.rand_like))]
        public void CreateFloat32TensorRandLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.rand_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.data<float>().ToArray(), t => Assert.True(t >= 0.0f && t < 1.0f));
        }

        [Fact]
        [TestOf(nameof(torch.rand_like))]
        public void CreateFloat32TensorRandLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.rand_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.data<double>().ToArray(), t => Assert.True(t >= 0.0 && t < 1.0));
        }

        [Fact]
        [TestOf(nameof(torch.rand_like))]
        public void CreateComplexFloat32TensorRandLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape, complex64);
            Tensor t2 = t1.rand_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.rand_like))]
        public void CreateComplexFloat32TensorRandLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.rand_like(dtype: ScalarType.ComplexFloat32);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.randn_like))]
        public void CreateFloat32TensorRandnLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.randn_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.randn_like))]
        public void CreateFloat32TensorRandnLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.randn_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.randn_like))]
        public void CreateComplexFloat32TensorRandnLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape, complex64);
            Tensor t2 = t1.randn_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.randn_like))]
        public void CreateComplexFloat32TensorRandnLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.randn_like(dtype: ScalarType.ComplexFloat32);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        [TestOf(nameof(torch.randint_like))]
        public void CreateFloat32TensorRandintLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.randint_like(5, 15);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.data<float>().ToArray(), t => Assert.True(t >= 5.0f && t < 15.0f));
        }

        [Fact]
        [TestOf(nameof(torch.randint_like))]
        public void CreateFloat32TensorRandintLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = torch.empty(shape);
            Tensor t2 = t1.randint_like(5, 15, dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.data<double>().ToArray(), t => Assert.True(t >= 5.0 && t < 15.0));
        }

        [Fact]
        [TestOf(nameof(torch.rand))]
        [TestOf(nameof(torch.randn))]
        public void CreateRandomTensorWithNames()
        {
            var names = new[] { "A", "B" };
            var t = torch.rand(3, 4, names: names);
            Assert.True(t.has_names());
            Assert.NotNull(t.names);
            Assert.Equal(names, t.names);

            t = torch.randn(3, 4, names: names);
            Assert.True(t.has_names());
            Assert.NotNull(t.names);
            Assert.Equal(names, t.names);
        }

        [Fact]
        [TestOf(nameof(Tensor.mean))]
        public void Float32Mean()
        {
            using var tensor = torch.arange(1, 100, float32);
            var mean = tensor.mean().item<float>();
            Assert.Equal(50.0f, mean);
        }


        [Fact]
        [TestOf(nameof(Tensor.mode))]
        public void Int32Mode()
        {
            using var tensor = torch.tensor(new int[] { 1, 5, 4, 5, 3, 3, 5, 5 });
            var mode = tensor.mode();
            Assert.Equal(new int[] { 5 }, mode.values.data<int>().ToArray());
            Assert.Equal(new long[] { 7 }, mode.indices.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItem2()
        {
            var shape = new long[] { 2, 3 };
            Tensor t = torch.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2].ToSingle());
            t[1, 2] = torch.tensor(2.0f);
            Assert.Equal(2.0f, t[1, 2].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItemComplexFloat2()
        {
            var shape = new long[] { 2, 3 };
            Tensor t = torch.ones(shape, complex64);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToComplex32().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex32().Real);
            Assert.Equal(1.0f, t[1, 2].ToComplex32().Real);

            t[1, 1] = torch.tensor(0.5f, 1.0f, complex64);
            Assert.Equal(0.5f, t[1, 1].ToComplex32().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex32().Imaginary);

            t[1, 2] = torch.tensor((2.0f, 3.0f), complex64);
            Assert.Equal(2.0f, t[1, 2].ToComplex32().Real);
            Assert.Equal(3.0f, t[1, 2].ToComplex32().Imaginary);
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItemComplexDouble2()
        {
            var shape = new long[] { 2, 3 };
            Tensor t = torch.ones(shape, complex128);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToComplex64().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex64().Real);
            Assert.Equal(1.0f, t[1, 2].ToComplex64().Real);

            t[0, 1] = torch.tensor((0.5, 1.0), complex128);
            Assert.Equal(0.5f, t[0, 1].ToComplex64().Real);
            Assert.Equal(1.0f, t[0, 1].ToComplex64().Imaginary);

            t[1, 1] = torch.tensor(0.5, 1.0, complex128);
            Assert.Equal(0.5f, t[1, 1].ToComplex64().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex64().Imaginary);

            t[1, 2] = torch.tensor(new System.Numerics.Complex(2.0, 3.0f), complex128);
            Assert.Equal(2.0f, t[1, 2].ToComplex64().Real);
            Assert.Equal(3.0f, t[1, 2].ToComplex64().Imaginary);
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItem3()
        {
            var shape = new long[] { 2, 3, 4 };
            Tensor t = torch.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3].ToSingle());
            t[1, 2, 3] = torch.tensor(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItem4()
        {
            var shape = new long[] { 2, 3, 4, 5 };
            Tensor t = torch.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3, 4].ToSingle());
            t[1, 2, 3, 4] = torch.tensor(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3, 4].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItem5()
        {
            var shape = new long[] { 2, 3, 4, 5, 6 };
            Tensor t = torch.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3, 4, 5].ToSingle());
            t[1, 2, 3, 4, 5] = torch.tensor(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3, 4, 5].ToSingle());
        }


        [Fact]
        [TestOf(nameof(Tensor))]
        public void GetSetItem6()
        {
            var shape = new long[] { 2, 3, 4, 5, 6, 7 };
            Tensor t = torch.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3, 4, 5, 6].ToSingle());
            t[1, 2, 3, 4, 5, 6] = torch.tensor(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3, 4, 5, 6].ToSingle());
        }

        // regression for https://github.com/dotnet/TorchSharp/issues/521
        [Fact]
        [TestOf(nameof(ones))]
        public void SetItemSlice()
        {
            var shape = new long[] { 2, 2, 2, 2, 2, 2, 2 };

            using var t = zeros(size: shape);
            t[1] = 1 * ones(size: new long[] { 2, 2, 2, 2, 2, 2 });
            t[1, 1] = 2 * ones(size: new long[] { 2, 2, 2, 2, 2 });
            t[1, 1, 1] = 3 * ones(size: new long[] { 2, 2, 2, 2 });
            t[1, 1, 1, 1] = 4 * ones(size: new long[] { 2, 2, 2 });
            t[1, 1, 1, 1, 1] = 5 * ones(size: new long[] { 2, 2 });
            t[1, 1, 1, 1, 1, 1] = 6 * ones(size: new long[] { 2 });

            Assert.Equal(0, t[0, 0, 0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(1, t[1, 0, 0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(2, t[1, 1, 0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(3, t[1, 1, 1, 0, 0, 0, 0].ToSingle());
            Assert.Equal(4, t[1, 1, 1, 1, 0, 0, 0].ToSingle());
            Assert.Equal(5, t[1, 1, 1, 1, 1, 0, 0].ToSingle());
            Assert.Equal(6, t[1, 1, 1, 1, 1, 1, 0].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.index_add))]
        public void IndexAdd1()
        {
            using var _ = NewDisposeScope();

            var y = torch.ones(3, 3);
            var t = torch.tensor(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var index = torch.tensor(new long[] { 2, 0 });
            var x = y.index_add(0, index, t, 2.0);

            Assert.Multiple(
                () => Assert.Equal(9.0, x[0, 0].ToSingle()),
                () => Assert.Equal(1.0, x[1, 0].ToSingle()),
                () => Assert.Equal(3.0, x[2, 0].ToSingle()),
                () => Assert.Equal(11.0, x[0, 1].ToSingle()),
                () => Assert.Equal(1.0, x[1, 1].ToSingle()),
                () => Assert.Equal(5.0, x[2, 1].ToSingle()),
                () => Assert.Equal(13.0, x[0, 2].ToSingle()),
                () => Assert.Equal(1.0, x[1, 2].ToSingle()),
                () => Assert.Equal(7.0, x[2, 2].ToSingle()));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_add_))]
        public void IndexAdd2()
        {
            using var _ = NewDisposeScope();

            var x = torch.ones(3, 3);
            var t = torch.tensor(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var index = torch.tensor(new long[] { 2, 0 });
            var y = x.index_add_(0, index, t, 2.0);

            Assert.Multiple(
                () => Assert.Same(x, y),
                () => Assert.Equal(9.0, x[0, 0].ToSingle()),
                () => Assert.Equal(1.0, x[1, 0].ToSingle()),
                () => Assert.Equal(3.0, x[2, 0].ToSingle()),
                () => Assert.Equal(11.0, x[0, 1].ToSingle()),
                () => Assert.Equal(1.0, x[1, 1].ToSingle()),
                () => Assert.Equal(5.0, x[2, 1].ToSingle()),
                () => Assert.Equal(13.0, x[0, 2].ToSingle()),
                () => Assert.Equal(1.0, x[1, 2].ToSingle()),
                () => Assert.Equal(7.0, x[2, 2].ToSingle()));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_copy))]
        public void IndexCopy1()
        {
            using var _ = NewDisposeScope();

            var y = torch.zeros(3, 3);
            var t = torch.tensor(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var index = torch.tensor(new long[] { 2, 0 });
            var x = y.index_copy(0, index, t);

            Assert.Multiple(
                () => Assert.Equal(4.0, x[0, 0].ToSingle()),
                () => Assert.Equal(0.0, x[1, 0].ToSingle()),
                () => Assert.Equal(1.0, x[2, 0].ToSingle()),
                () => Assert.Equal(5.0, x[0, 1].ToSingle()),
                () => Assert.Equal(0.0, x[1, 1].ToSingle()),
                () => Assert.Equal(2.0, x[2, 1].ToSingle()),
                () => Assert.Equal(6.0, x[0, 2].ToSingle()),
                () => Assert.Equal(0.0, x[1, 2].ToSingle()),
                () => Assert.Equal(3.0, x[2, 2].ToSingle()));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_copy_))]
        public void IndexCopy2()
        {
            using var _ = NewDisposeScope();

            var x = torch.zeros(3, 3);
            var t = torch.tensor(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var index = torch.tensor(new long[] { 2, 0 });
            var y = x.index_copy_(0, index, t);

            Assert.Multiple(
                () => Assert.Same(x, y),
                () => Assert.Equal(4.0, x[0, 0].ToSingle()),
                () => Assert.Equal(0.0, x[1, 0].ToSingle()),
                () => Assert.Equal(1.0, x[2, 0].ToSingle()),
                () => Assert.Equal(5.0, x[0, 1].ToSingle()),
                () => Assert.Equal(0.0, x[1, 1].ToSingle()),
                () => Assert.Equal(2.0, x[2, 1].ToSingle()),
                () => Assert.Equal(6.0, x[0, 2].ToSingle()),
                () => Assert.Equal(0.0, x[1, 2].ToSingle()),
                () => Assert.Equal(3.0, x[2, 2].ToSingle()));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_fill))]
        public void IndexFill1()
        {
            using var _ = NewDisposeScope();

            var y = torch.zeros(3, 3);
            var index = torch.tensor(new long[] { 0, 2 });
            var x = y.index_fill(1, index, 1.0);

            Assert.Multiple(
                () => Assert.Equal(1.0, x[0, 0].ToSingle()),
                () => Assert.Equal(1.0, x[1, 0].ToSingle()),
                () => Assert.Equal(1.0, x[2, 0].ToSingle()),
                () => Assert.Equal(0.0, x[0, 1].ToSingle()),
                () => Assert.Equal(0.0, x[1, 1].ToSingle()),
                () => Assert.Equal(0.0, x[2, 1].ToSingle()),
                () => Assert.Equal(1.0, x[0, 2].ToSingle()),
                () => Assert.Equal(1.0, x[1, 2].ToSingle()),
                () => Assert.Equal(1.0, x[2, 2].ToSingle()));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_fill))]
        public void IndexFill2()
        {
            using var _ = NewDisposeScope();

            var x = torch.zeros(3, 3);
            var index = torch.tensor(new long[] { 0, 2 });
            var y = x.index_fill_(1, index, 1.0);

            Assert.Multiple(
                () => Assert.Same(x, y),
                () => Assert.Equal(1.0, x[0, 0].ToSingle()),
                () => Assert.Equal(1.0, x[1, 0].ToSingle()),
                () => Assert.Equal(1.0, x[2, 0].ToSingle()),
                () => Assert.Equal(0.0, x[0, 1].ToSingle()),
                () => Assert.Equal(0.0, x[1, 1].ToSingle()),
                () => Assert.Equal(0.0, x[2, 1].ToSingle()),
                () => Assert.Equal(1.0, x[0, 2].ToSingle()),
                () => Assert.Equal(1.0, x[1, 2].ToSingle()),
                () => Assert.Equal(1.0, x[2, 2].ToSingle()));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_put_))]
        public void IndexPutOneValueOneIndex()
        {
            using var _ = NewDisposeScope();

            var tensor = ones(5);
            var indices = new TensorIndex[] { TensorIndex.Tensor(1) };
            var values = torch.tensor(5.0f);

            // default accumulate value is false, should only replace value at index 1 with 5 
            tensor.index_put_(values, indices);
            Assert.True(tensor.Equals(torch.tensor(new float[] { 1.0f, 5.0f, 1.0f, 1.0f, 1.0f })));

            tensor = ones(5);
            // accumulate value is false, explicitly set, should only replace value at index 1 with 5
            tensor.index_put_(values, indices, accumulate: false);
            Assert.True(tensor.Equals(torch.tensor(new float[] { 1.0f, 5.0f, 1.0f, 1.0f, 1.0f })));

            tensor = ones(5);
            // accumulate value is true, should add value to index 1, 1 + 5 = 6
            tensor.index_put_(values, indices, accumulate: true);
            Assert.True(tensor.Equals(torch.tensor(new float[] { 1.0f, 6.0f, 1.0f, 1.0f, 1.0f })));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_put_))]
        public void IndexPutOneValueMultipleIndexes()
        {
            using var _ = NewDisposeScope();

            var tensor = ones(5);
            var indices = new TensorIndex[] { TensorIndex.Tensor(new long[] {1, 2}) };
            var values = torch.tensor(10.0f);

            // default accumulate value is false, should only replace value at given indexes
            tensor.index_put_(values, indices);
            Assert.True(tensor.Equals(torch.tensor(new float[] { 1.0f, 10.0f, 10.0f, 1.0f, 1.0f })));

            tensor = ones(5);
            // accumulate value is true, should add value to given indexes
            tensor.index_put_(values, indices, true);
            Assert.True(tensor.Equals(torch.tensor(new float[] { 1.0f, 11.0f, 11.0f, 1.0f, 1.0f })));

            // accumulate value is false, explicitly set, should replace value at given indexes
            tensor.index_put_(values, indices, false);
            Assert.True(tensor.Equals(torch.tensor(new float[] { 1.0f, 10.0f, 10.0f, 1.0f, 1.0f })));
        }

        [Fact]
        [TestOf(nameof(Tensor.index_put_))]
        public void IndexPutMultipleValuesMultipleIndexes()
        {
            using var _ = NewDisposeScope();

            var tensor = ones(5, 2);
            var indices = new TensorIndex[]
            {
                TensorIndex.Tensor(new long[] { 1, 2, 0, 3 }), // for first tensor dimension (row)
                TensorIndex.Tensor(new long[] { 0, 1, 0, 0 })  // for second tensor dimension (column)
            };
            var values = torch.tensor(new float[] { 3.0f, 4.0f, 5.0f, 10f });

            // default accumulate value is false, should only replace values at given indices with 3, 4, 5, 10 
            // Indexes to be replaced: (1, 0) -> 3.0,  (2, 1) -> 4.0, (0, 0) -> 5.0, (3, 0) -> 10.0
            tensor.index_put_(values, indices);
            Assert.True(tensor.Equals(torch.tensor(new float[,] { { 5.0f, 1.0f }, { 3.0f, 1.0f }, { 1.0f, 4.0f }, { 10.0f, 1.0f }, { 1.0f, 1.0f } })));

            tensor = ones(5, 2);
            // accumulate value is true, should perform addition at given indices, 1 + 3 = 4, 1 + 4 = 5, 1 + 5 = 6, 1 + 10 = 11
            // Indexes to be replaced: (1, 0) -> 4.0,  (2, 1) -> 5.0, (0, 0) -> 6.0, (3, 0) -> 11.0
            tensor.index_put_(values, indices, true);
            Assert.True(tensor.Equals(torch.tensor(new float[,] { { 6.0f, 1.0f }, { 4.0f, 1.0f }, { 1.0f, 5.0f }, { 11.0f, 1.0f }, { 1.0f, 1.0f } })));

            // accumulate value is false, explicitly set, should only replace values at given indices with 3, 4, 5, 10
            // Indexes to be replaced: (1, 0) -> 3.0,  (2, 1) -> 4.0, (0, 0) -> 5.0, (3, 0) -> 10.0
            tensor.index_put_(values, indices, false);
            Assert.True(tensor.Equals(torch.tensor(new float[,] { { 5.0f, 1.0f }, { 3.0f, 1.0f }, { 1.0f, 4.0f }, { 10.0f, 1.0f }, { 1.0f, 1.0f } })));
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.ToTensor))]
        public void ScalarToTensor()
        {
            Assert.Throws<ArgumentException>(() => 1.ToTensor(requires_grad: true));
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void ScalarToTensor2()
        {
            using (Tensor tensor = 1) {
                Assert.Equal(ScalarType.Int32, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(1, tensor.ToInt32());
            }
            using (Tensor tensor = ((byte)1).ToTensor()) {
                Assert.Equal(ScalarType.Byte, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(1, tensor.ToByte());
            }
            using (Tensor tensor = ((sbyte)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int8, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(-1, tensor.ToSByte());
            }
            using (Tensor tensor = ((short)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int16, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(-1, tensor.ToInt16());
            }
            using (Tensor tensor = ((long)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int64, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(-1L, tensor.ToInt64());
            }
            using (Tensor tensor = ((float)-1).ToTensor()) {
                Assert.Equal(ScalarType.Float32, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(-1.0f, tensor.ToSingle());
            }
            using (Tensor tensor = ((double)-1).ToTensor()) {
                Assert.Equal(ScalarType.Float64, tensor.dtype);
                Assert.Equal(0, tensor.ndim);
                Assert.Empty(tensor.shape);
                Assert.Equal(-1.0, tensor.ToDouble());
            }
        }

        [Fact]
        [TestOf(nameof(TensorExtensionMethods.ToTensor))]
        public void ScalarToTensor3()
        {
            using (Tensor tensor = 1.ToTensor()) {
                Assert.Equal(ScalarType.Int32, tensor.dtype);
                Assert.Equal(1, (int)tensor);
            }
            using (Tensor tensor = (byte)1) {
                Assert.Equal(ScalarType.Byte, tensor.dtype);
                Assert.Equal(1, (byte)tensor);
            }
            using (Tensor tensor = (sbyte)-1) {
                Assert.Equal(ScalarType.Int8, tensor.dtype);
                Assert.Equal(-1, (sbyte)tensor);
            }
            using (Tensor tensor = (short)-1) {
                Assert.Equal(ScalarType.Int16, tensor.dtype);
                Assert.Equal(-1, (short)tensor);
            }
            using (Tensor tensor = (long)-1) {
                Assert.Equal(ScalarType.Int64, tensor.dtype);
                Assert.Equal(-1L, (long)tensor);
            }
            using (Tensor tensor = (float)-1) {
                Assert.Equal(ScalarType.Float32, tensor.dtype);
                Assert.Equal(-1.0f, (float)tensor);
            }
            using (Tensor tensor = (double)-1) {
                Assert.Equal(ScalarType.Float64, tensor.dtype);
                Assert.Equal(-1.0, (double)tensor);
            }
        }
        [Fact]
        [TestOf(nameof(Tensor))]
        public void ScalarToTensorDoesNotLeakMemory()
        {
            AssertTensorDoesNotLeak(() => {
                Tensor tensor = 1;
                return tensor;
            });
            AssertTensorDoesNotLeak(() => ((byte)1).ToTensor());
            AssertTensorDoesNotLeak(() => ((sbyte)-1).ToTensor());
            AssertTensorDoesNotLeak(() => ((short)-1).ToTensor());
            AssertTensorDoesNotLeak(() => ((long)-1).ToTensor());
            AssertTensorDoesNotLeak(() => ((float)-1).ToTensor());
            AssertTensorDoesNotLeak(() => ((double)-1).ToTensor());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void ScalarArrayToTensorDoesNotLeakMemory()
        {
            AssertTensorDoesNotLeak(() => (new byte[] { 1 }).ToTensor(new long[] { 1 }));
            AssertTensorDoesNotLeak(() => (new sbyte[] { -1 }).ToTensor(new long[] { 1 }));
            AssertTensorDoesNotLeak(() => (new short[] { -1 }).ToTensor(new long[] { 1 }));
            AssertTensorDoesNotLeak(() => (new long[] { -1 }).ToTensor(new long[] { 1 }));
            AssertTensorDoesNotLeak(() => (new float[] { -1 }).ToTensor(new long[] { 1 }));
            AssertTensorDoesNotLeak(() => (new double[] { -1 }).ToTensor(new long[] { 1 }));
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void ComplexNumberOfDoubleDoesNotLeakMemory()
        {
            AssertTensorDoesNotLeak(() => (torch.tensor((double)-1, (double)-2)));
            AssertTensorDoesNotLeak(() => (torch.tensor(((double)-1, (double)-2))));
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void ComplexNumberOfFloatDoesNotLeakMemory()
        {
            AssertTensorDoesNotLeak(() => (torch.tensor((float)-1, (float)-2)));
            AssertTensorDoesNotLeak(() => (torch.tensor(((float)-1, (float)-2))));
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void DotNetComplexNumberDoesNotLeakMemory()
        {
            AssertTensorDoesNotLeak(() => (torch.tensor(new Complex(1, 2))));
        }

        private void AssertTensorDoesNotLeak(Func<Tensor> createTensorFunc)
        {
            var stats = DisposeScopeManager.Statistics.TensorStatistics;
            stats.Reset();
            using (Tensor tensor = createTensorFunc()) {
                Assert.Equal(1, stats.ThreadTotalLiveCount);
            }
            Assert.Equal(0, stats.ThreadTotalLiveCount);
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void NegativeScalarToTensor()
        {
            Scalar s = 10;
            TensorIndex ti = 10;
            Tensor t;

            Assert.Throws<InvalidOperationException>(() => { t = s; });
            Assert.Throws<InvalidOperationException>(() => { t = ti; });
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.calculate_gain))]
        public void CalculateGain()
        {
            Assert.Equal(1, torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.Linear));
            Assert.Equal(1, torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.Conv1D));
            Assert.Equal(1, torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.Conv2D));
            Assert.Equal(1, torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.Conv3D));
            Assert.Equal(1, torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.Sigmoid));
            Assert.Equal(5.0 / 3.0, torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.Tanh));
            Assert.Equal(Math.Sqrt(2.0), torch.nn.init.calculate_gain(torch.nn.init.NonlinearityType.ReLU));
        }

        [Fact]
        [TestOf(nameof(Tensor.isposinf))]
        [TestOf(nameof(Tensor.isinf))]
        [TestOf(nameof(Tensor.isneginf))]
        [TestOf(nameof(Tensor.isfinite))]
        [TestOf(nameof(Tensor.isnan))]
        public void InfinityTest()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            using var t1 = tensor.fill_(float.PositiveInfinity);
            Assert.Same(tensor, t1);

            Assert.True(tensor.isposinf().data<bool>().ToArray().All(b => b));
            Assert.True(tensor.isinf().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isneginf().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isfinite().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isnan().data<bool>().ToArray().All(b => b));

            using var t2 = tensor.fill_(float.NegativeInfinity);
            Assert.Same(tensor, t2);

            Assert.True(tensor.isneginf().data<bool>().ToArray().All(b => b));
            Assert.True(tensor.isinf().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isposinf().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isfinite().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isnan().data<bool>().ToArray().All(b => b));
        }

        [Fact]
        [TestOf(nameof(torch.isnan))]
        public void IsNaNTest()
        {
            var array = new float[] { float.NaN, float.PositiveInfinity, 1f, 0f, -1f, float.NegativeInfinity };
            var expected = new bool[] { true, false, false, false, false, false };
            using Tensor tensor = torch.from_array(array);
            var result = torch.isnan(tensor).data<bool>().ToArray();
            Assert.Equal(expected, result);
        }

        [Fact]
        [TestOf(nameof(torch.bernoulli))]
        public void TorchBernoulli()
        {
            using var tensor = torch.bernoulli(torch.rand(5));
            Assert.Equal(5, tensor.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.multinomial))]
        public void TorchMultinomial()
        {
            using var tensor = torch.multinomial(torch.rand(5), 17, true);
            Assert.Equal(17, tensor.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.poisson))]
        public void TorchPoisson()
        {
            using var tensor = torch.poisson(torch.rand(5));
            Assert.Equal(5, tensor.shape[0]);
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.zeros_))]
        public void InitZeros()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.zeros_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.ones_))]
        public void InitOnes()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.ones_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.dirac_))]
        public void InitDirac()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.dirac_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.eye_))]
        public void InitEye()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.eye_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.constant_))]
        public void InitConstant()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.constant_(tensor, Math.PI));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.uniform_))]
        public void InitUniform()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.uniform_(tensor));
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, torch.nn.init.uniform_(tensor, generator: gen));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.normal_))]
        public void InitNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.normal_(tensor));
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, torch.nn.init.normal_(tensor, generator: gen));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.trunc_normal_))]
        public void InitTruncNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.trunc_normal_(tensor));
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, torch.nn.init.trunc_normal_(tensor, generator: gen));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.orthogonal_))]
        public void InitOrthogonal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.orthogonal_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.sparse_))]
        public void InitSparse()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.sparse_(tensor, 0.25));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.kaiming_uniform_))]
        public void InitKaimingUniform()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.kaiming_uniform_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.kaiming_normal_))]
        public void InitKaimingNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.kaiming_normal_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.xavier_uniform_))]
        public void InitXavierUniform()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.xavier_uniform_(tensor));
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.xavier_normal_))]
        public void InitXavierNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, torch.nn.init.xavier_normal_(tensor));
        }

        [Fact]
        [TestOf(nameof(Tensor.bernoulli_))]
        public void InplaceBernoulli()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.bernoulli_());
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.bernoulli_(generator: gen));
        }

        [Fact]
        [TestOf(nameof(Tensor.cauchy_))]
        public void InplaceCauchy()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.cauchy_());
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.cauchy_(generator: gen));
        }

        [Fact]
        [TestOf(nameof(Tensor.exponential_))]
        public void InplaceExponential()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.exponential_());
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.exponential_(generator: gen));
        }

        [Fact]
        [TestOf(nameof(Tensor.geometric_))]
        public void InplaceGeometric()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.geometric_(0.25));
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.geometric_(0.25, generator: gen));
            using (var res = tensor.geometric_(0.25, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.log_normal_))]
        public void InplaceLogNormal()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.log_normal_());
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.log_normal_(generator: gen));
        }

        [Fact]
        [TestOf(nameof(torch.normal))]
        public void Normal()
        {
            long[] size = new long[] { 2, 3 };
            var mean = torch.randn(size: size);
            var std = torch.randn(size: size);
            var a = torch.normal(mean, std);
            Assert.Equal(size, a.shape);
            var b = torch.normal(1.0, 2.0, size: size);
            Assert.Equal(size, a.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.normal_))]
        public void InplaceNormal()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.normal_());
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.normal_(generator: gen));
        }

        [Fact]
        [TestOf(nameof(Tensor.random_))]
        public void InplaceRandom()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.random_(0.0, 1.0));
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.random_(0.0, 1.0, generator: gen));
        }

        [Fact]
        [TestOf(nameof(Tensor.uniform_))]
        public void InplaceUniform()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor are the same.
            Assert.Same(tensor, tensor.uniform_(0.0, 1.0));
            using var gen = new torch.Generator(4711L);
            Assert.Same(tensor, tensor.uniform_(0.0, 1.0, generator: gen));
        }

        [Fact]
        [TestOf(nameof(torch.sparse))]
        public void SparseCOOTensor()
        {
            using var i = torch.tensor(new long[] { 0, 1, 1, 2, 0, 2 }, new long[] { 2, 3 });
            using Tensor v = new float[] { 3, 4, 5 };
            var sparse = torch.sparse_coo_tensor(i, v, new long[] { 2, 3 });

            Assert.True(sparse.is_sparse);
            Assert.False(i.is_sparse);
            Assert.False(v.is_sparse);
            Assert.Equal(sparse.SparseIndices.data<long>().ToArray(), new long[] { 0, 1, 1, 2, 0, 2 });
            Assert.Equal(sparse.SparseValues.data<float>().ToArray(), new float[] { 3, 4, 5 });
        }

        [Fact]
        public void IndexerAndImplicitConversion()
        {
            var points = torch.zeros(6, torch.ScalarType.Float16);
            points[0] = 4.0f;
            points[1] = 1.0d;
            points[2] = 7;
        }

        [Fact]
        public void IndexSingle()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            Assert.Equal(0, i.index(new TensorIndex[] { TensorIndex.Single(0), TensorIndex.Single(0) }).ToInt32());
            Assert.Equal(1, i.index(new TensorIndex[] { TensorIndex.Single(0), TensorIndex.Single(1) }).ToInt32());
            Assert.Equal(2, i.index(new TensorIndex[] { TensorIndex.Single(0), TensorIndex.Single(2) }).ToInt32());
            Assert.Equal(6, i.index(new TensorIndex[] { TensorIndex.Single(1), TensorIndex.Single(0) }).ToInt32());
            Assert.Equal(5, i.index(new TensorIndex[] { TensorIndex.Single(1), TensorIndex.Single(1) }).ToInt32());
            Assert.Equal(4, i.index(new TensorIndex[] { TensorIndex.Single(1), TensorIndex.Single(2) }).ToInt32());
        }

        [Fact]
        public void IndexEllipsis()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Single(0) });
            Assert.Equal(0, t1[0].ToInt32());
            Assert.Equal(6, t1[1].ToInt32());
        }

        [Fact]
        public void IndexNull()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.Null, TensorIndex.Single(0) });
            Assert.Equal(0, t1[0, 0].ToInt32());
            Assert.Equal(1, t1[0, 1].ToInt32());
            Assert.Equal(2, t1[0, 2].ToInt32());
        }

        [Fact]
        public void IndexNone()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.None, TensorIndex.Single(0) });
            Assert.Equal(0, t1[0, 0].ToInt32());
            Assert.Equal(1, t1[0, 1].ToInt32());
            Assert.Equal(2, t1[0, 2].ToInt32());
        }

        [Fact]
        public void IndexSlice1()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.Slice(0, 2), TensorIndex.Single(0) });
            Assert.Equal(0, t1[0].ToInt32());
            Assert.Equal(6, t1[1].ToInt32());

            // one slice
            var t2 = i.index(new TensorIndex[] { TensorIndex.Slice(1, 2), TensorIndex.Single(0) });
            Assert.Equal(6, t2[0].ToInt32());

            // two slice
            var t3 = i.index(new TensorIndex[] { TensorIndex.Slice(1, 2), TensorIndex.Slice(1, 3) });
            Assert.Equal(5, t3[0, 0].ToInt32());
            Assert.Equal(4, t3[0, 1].ToInt32());

            // slice with step
            var t4 = i.index(new TensorIndex[] { TensorIndex.Slice(1, 2), TensorIndex.Slice(0, 3, step: 2) });
            Assert.Equal(6, t4[0, 0].ToInt32());
            Assert.Equal(4, t4[0, 1].ToInt32());

            // end absent
            var t5 = i.index(new TensorIndex[] { TensorIndex.Slice(start: 1), TensorIndex.Slice(start: 1) });
            Assert.Equal(5, t5[0, 0].ToInt32());
            Assert.Equal(4, t5[0, 1].ToInt32());

            // start absent
            var t6 = i.index(new TensorIndex[] { TensorIndex.Slice(start: 1), TensorIndex.Slice(stop: 2) });
            Assert.Equal(6, t6[0, 0].ToInt32());
            Assert.Equal(5, t6[0, 1].ToInt32());

            // start and end absent
            var t7 = i.index(new TensorIndex[] { TensorIndex.Slice(start: 1), TensorIndex.Slice(step: 2) });
            Assert.Equal(6, t7[0, 0].ToInt32());
            Assert.Equal(4, t7[0, 1].ToInt32());
        }


        [Fact]
        [TestOf(nameof(Tensor))]
        public void IndexSlice2()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
#if NET472_OR_GREATER
            var t1 = i[(0, 2), 0];
            Assert.Equal(0, t1[0].ToInt32());
            Assert.Equal(6, t1[1].ToInt32());

            // one slice
            var t2 = i[(1, 2), 0];
            Assert.Equal(6, t2[0].ToInt32());

            t2 = i[1, -1];
            Assert.Equal(4, t2.item<long>());

            t2 = i[-1, -2];
            Assert.Equal(5, t2.item<long>());

            // two slice
            var t3 = i[(1, 2), (1, 3)];
            Assert.Equal(5, t3[0, 0].ToInt32());
            Assert.Equal(4, t3[0, 1].ToInt32());

            t3 = i[(1, 2), (-2, null)];
            Assert.Equal(5, t3[0, 0].ToInt32());
            Assert.Equal(4, t3[0, 1].ToInt32());

            // both absent
            var t4 = i[(0, null), (0, null)];
            Assert.Equal(0, t4[0, 0].ToInt32());
            Assert.Equal(1, t4[0, 1].ToInt32());

            // end absent
            var t5 = i[(1, null), (1, null)];
            Assert.Equal(5, t5[0, 0].ToInt32());
            Assert.Equal(4, t5[0, 1].ToInt32());

            // start absent
            var t6 = i[(1, null), (0, 2)];
            Assert.Equal(6, t6[0, 0].ToInt32());
            Assert.Equal(5, t6[0, 1].ToInt32());
#else
            var t1 = i[0..2, 0];
            Assert.Equal(0, t1[0].ToInt32());
            Assert.Equal(6, t1[1].ToInt32());

            // one slice
            var t2 = i[1..2, 0];
            Assert.Equal(6, t2[0].ToInt32());

            t2 = i[1, ^1];
            Assert.Equal(4, t2.item<long>());

            t2 = i[-1, -2];
            Assert.Equal(5, t2.item<long>());

            // two slice
            var t3 = i[1..2, 1..3];
            Assert.Equal(5, t3[0, 0].ToInt32());
            Assert.Equal(4, t3[0, 1].ToInt32());

            t3 = i[1..2, ^2..];
            Assert.Equal(5, t3[0, 0].ToInt32());
            Assert.Equal(4, t3[0, 1].ToInt32());

            // both absent
            var t4 = i[.., ..];
            Assert.Equal(0, t4[0, 0].ToInt32());
            Assert.Equal(1, t4[0, 1].ToInt32());

            // end absent
            var t5 = i[1.., 1..];
            Assert.Equal(5, t5[0, 0].ToInt32());
            Assert.Equal(4, t5[0, 1].ToInt32());

            // start absent
            var t6 = i[1.., ..2];
            Assert.Equal(6, t6[0, 0].ToInt32());
            Assert.Equal(5, t6[0, 1].ToInt32());
#endif // NET472_OR_GREATER
        }

        [Fact]
        [TestOf(nameof(Tensor.cuda))]
        public void CopyCpuToCuda()
        {
            Tensor cpu = torch.ones(new long[] { 2, 2 });
            Assert.Equal("cpu", cpu.device.ToString());

            if (torch.cuda.is_available()) {
                var cuda = cpu.cuda();
                Assert.NotSame(cuda, cpu);
                Assert.Equal("cuda:0", cuda.device.ToString());

                // Copy back to CPU to inspect the elements
                var cpu2 = cuda.cpu();
                Assert.NotSame(cuda, cpu2);
                Assert.Equal("cpu", cpu2.device.ToString());
                Assert.Equal("cuda:0", cuda.device.ToString());

                var data = cpu.data<float>();
                for (int i = 0; i < 4; i++) {
                    Assert.Equal(1, data[i]);
                }
            } else {
                Assert.Throws<InvalidOperationException>(() => cpu.cuda());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.cpu))]
        public void CopyCudaToCpu()
        {
            if (torch.cuda.is_available()) {
                var cuda = torch.ones(new long[] { 2, 2 }, device: torch.CUDA);
                Assert.Equal("cuda:0", cuda.device.ToString());

                var cpu = cuda.cpu();
                Assert.NotSame(cuda, cpu);
                Assert.Equal("cpu", cpu.device.ToString());
                Assert.Equal("cuda:0", cuda.device.ToString());

                var data = cpu.data<float>();
                for (int i = 0; i < 4; i++) {
                    Assert.Equal(1, data[i]);
                }
            } else {
                Assert.Throws<InvalidOperationException>((Action)(() => { torch.ones(new long[] { 2, 2 }, device: torch.CUDA); }));
            }
        }

        [Fact]
        public void SquareEuclideanDistance()
        {
            Tensor input = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.1f };

            var zeros = torch.zeros(new long[] { 1, 9 });
            var ones = torch.ones(new long[] { 1, 9 });
            var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);

            var distanceFromZero = input.reshape(new long[] { -1, 1, 9 }).sub(zeros).pow(2.ToScalar()).sum(new long[] { 2 });
            var distanceFromOne = input.reshape(new long[] { -1, 1, 9 }).sub(ones).pow(2.ToScalar()).sum(new long[] { 2 });
            var distanceFromCentroids = input.reshape(new long[] { -1, 1, 9 }).sub(centroids).pow(2.ToScalar()).sum(new long[] { 2 });

            Assert.True(true);
        }

        [Fact]
        public void Cat()
        {
            var zeros = torch.zeros(new long[] { 1, 9 });
            var ones = torch.ones(new long[] { 1, 9 });
            var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);

            var shape = centroids.shape;
            Assert.Equal(new long[] { 2, 9 }, shape);
        }

        [Fact]
        public void CatCuda()
        {
            if (torch.cuda.is_available()) {
                var zeros = torch.zeros(new long[] { 1, 9 }).cuda();
                var ones = torch.ones(new long[] { 1, 9 }).cuda();
                var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);
                var shape = centroids.shape;
                Assert.Equal(new long[] { 2, 9 }, shape);
                Assert.Equal(DeviceType.CUDA, centroids.device_type);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.copy_))]
        public void Copy()
        {
            var first = torch.rand(new long[] { 1, 9 });
            var second = torch.zeros(new long[] { 1, 9 });

            second.copy_(first);

            Assert.Equal(first, second);
        }

        [Fact]
        [TestOf(nameof(Tensor.copy_))]
        public void CopyCuda()
        {
            if (torch.cuda.is_available()) {
                var first = torch.rand(new long[] { 1, 9 }).cuda();
                var second = torch.zeros(new long[] { 1, 9 }).cuda();

                second.copy_(first);

                Assert.Equal(first, second);
            }
        }

        void StackGen(Device device)
        {
            {
                var t1 = torch.zeros(new long[] { }, device: device);
                var t2 = torch.ones(new long[] { }, device: device);
                var t3 = torch.ones(new long[] { }, device: device);
                var res = torch.stack(new Tensor[] { t1, t2, t3 }, 0);

                var shape = res.shape;
                Assert.Equal(new long[] { 3 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = torch.zeros(new long[] { }, device: device);
                var t2 = torch.ones(new long[] { }, device: device);
                var t3 = torch.ones(new long[] { }, device: device);
                var res = torch.hstack(new Tensor[] { t1, t2, t3 });

                var shape = res.shape;
                Assert.Equal(new long[] { 3 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = torch.zeros(new long[] { 2, 9 }, device: device);
                var t2 = torch.ones(new long[] { 2, 9 }, device: device);
                var res = torch.stack(new Tensor[] { t1, t2 }, 0);

                var shape = res.shape;
                Assert.Equal(new long[] { 2, 2, 9 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = torch.zeros(new long[] { 2, 9 }, device: device);
                var t2 = torch.ones(new long[] { 2, 9 }, device: device);
                var res = torch.hstack(new Tensor[] { t1, t2 });

                var shape = res.shape;
                Assert.Equal(new long[] { 2, 18 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = torch.zeros(new long[] { 2, 9 }, device: device);
                var t2 = torch.ones(new long[] { 2, 9 }, device: device);
                var res = torch.vstack(new Tensor[] { t1, t2 });

                var shape = res.shape;
                Assert.Equal(new long[] { 4, 9 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = torch.zeros(new long[] { 2, 9 }, device: device);
                var t2 = torch.ones(new long[] { 2, 9 }, device: device);
                var res = torch.dstack(new Tensor[] { t1, t2 });

                var shape = res.shape;
                Assert.Equal(new long[] { 2, 9, 2 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.to))]
        public void Cast()
        {
            using var input = torch.rand(new long[] { 128 }, float64, torch.CPU);

            {
                using var moved = input.to(ScalarType.Float32);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
            {
                using var moved = input.type(ScalarType.Float32);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
            {
                using var moved = input.type(torch.FloatTensor);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
            {
                using var moved = input.to(ScalarType.Int32);
                Assert.Equal(ScalarType.Int32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
            {
                using var moved = input.type(ScalarType.Int32);
                Assert.Equal(ScalarType.Int32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
            {
                using var moved = input.type(torch.IntTensor);
                Assert.Equal(ScalarType.Int32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.to))]
        public void MoveAndCast()
        {
            var input = torch.rand(new long[] { 128 }, float64, torch.CPU);

            if (torch.cuda.is_available()) {
                var moved = input.to(ScalarType.Float32, torch.CUDA, non_blocking: true);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CUDA, moved.device_type);
            } else {
                var moved = input.to(ScalarType.Float32, non_blocking: true);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.to))]
        public void Meta()
        {
            var input = torch.rand(new long[] { 128 }, float64, torch.CPU);
            var x = input.to(DeviceType.META);

            var y = x + 10;

            Assert.Equal(DeviceType.META, y.device_type);
            Assert.Equal(x.shape, y.shape);

            var z = input.to(ScalarType.Float32, torch.META);

            Assert.Equal(ScalarType.Float32, z.dtype);
            Assert.Equal(DeviceType.META, z.device_type);
            Assert.Equal(x.shape, z.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.to))]
        public void CastMoveAndDisposeAfter()
        {
            {
                // Cast the input on the same device
                using var input = torch.ones(10, float32, torch.CPU);
                using var cast = input.to(int32, disposeAfter: true);
                Assert.True(input.IsInvalid);
                Assert.False(cast.IsInvalid);
                // make sure we can access the values
                Assert.Equal(1, cast[0].ToInt32());
            }
            if (torch.cuda.is_available()) {
                {
                    // Move the input to a different device
                    using var input = torch.ones(10, float32, torch.CPU);
                    using var moved = input.to(torch.CUDA, disposeAfter: true);
                    Assert.True(input.IsInvalid);
                    Assert.False(moved.IsInvalid);
                    // make sure we can access the values
                    Assert.Equal(1, moved[0].ToSingle());
                }
                {
                    // Cast and move the input to a different device
                    using var input = torch.ones(10, float32, torch.CPU);
                    using var moved = input.to(int32, torch.CUDA, disposeAfter: true);
                    Assert.True(input.IsInvalid);
                    Assert.False(moved.IsInvalid);
                    // make sure we can access the values
                    Assert.Equal(1, moved[0].ToInt32());
                }
            }
            {
                // Sanity: If we cast to the same type, values should still be accessible
                using var input = torch.ones(10, float32, torch.CPU);
                using var cast = input.to(float32, disposeAfter: true);
                Assert.True(input.IsInvalid);
                Assert.False(cast.IsInvalid);
                // make sure we can access the values
                Assert.Equal(1, cast[0].ToSingle());
            }
            {
                // Sanity: If we move to the same device, values should still be accessible
                using var input = torch.ones(10, float32, torch.CPU);
                using var moved = input.to(torch.CPU, disposeAfter: true);
                Assert.True(input.IsInvalid);
                Assert.False(moved.IsInvalid);
                // make sure we can access the values
                Assert.Equal(1, moved[0].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.masked_scatter))]
        [TestOf(nameof(Tensor.masked_scatter_))]
        public void MaskedScatter()
        {
            var input = torch.zeros(new long[] { 4, 4 });
            var mask = torch.zeros(new long[] { 4, 4 }, torch.@bool);
            var tTrue = torch.tensor(true);
            mask[0, 1] = true;
            mask[2, 3] = true;

            var res = input.masked_scatter(mask, torch.tensor(new float[] { 3.14f, 2 * 3.14f }));
            Assert.Equal(3.14f, res[0, 1].item<float>());
            Assert.Equal(2 * 3.14f, res[2, 3].item<float>());

            var z = input.masked_scatter_(mask, torch.tensor(new float[] { 3.14f, 2 * 3.14f }));
            Assert.Same(input, z);
            Assert.Equal(3.14f, input[0, 1].item<float>());
            Assert.Equal(2 * 3.14f, input[2, 3].item<float>());
        }

        [Fact]
        [TestOf(nameof(Tensor.masked_fill))]
        [TestOf(nameof(Tensor.masked_fill_))]
        public void MaskedFill()
        {
            var input = torch.zeros(new long[] { 4, 4 });
            var mask = torch.zeros(new long[] { 4, 4 }, torch.@bool);
            mask[0, 1] = true;
            mask[2, 3] = true;

            var res = input.masked_fill(mask, 3.14f);
            Assert.Equal(3.14f, res[0, 1].item<float>());
            Assert.Equal(3.14f, res[2, 3].item<float>());

            var z = input.masked_fill_(mask, 3.14f);
            Assert.Same(input, z);
            Assert.Equal(3.14f, input[0, 1].item<float>());
            Assert.Equal(3.14f, input[2, 3].item<float>());
        }

        [Fact]
        [TestOf(nameof(Tensor.masked_select))]
        public void MaskedSelect()
        {
            var input = torch.zeros(new long[] { 4, 4 });
            var mask = torch.eye(4, 4, torch.@bool);

            var res = input.masked_select(mask);
            Assert.Equal(4, res.numel());
        }

        [Fact]
        [TestOf(nameof(Tensor.diagonal_scatter))]
        public void DiagonalScatter()
        {
            var a = torch.zeros(3, 3);

            var res = a.diagonal_scatter(torch.ones(3), 0);

            Assert.Equal(0, res[0, 1].item<float>());

            Assert.Equal(1, res[0, 0].item<float>());
            Assert.Equal(1, res[1, 1].item<float>());
            Assert.Equal(1, res[2, 2].item<float>());
        }

        [Fact]
        [TestOf(nameof(Tensor.slice_scatter))]
        public void SliceScatter()
        {
            var a = torch.zeros(8, 8);

            var res = a.slice_scatter(torch.ones(2, 8), start: 6);

            Assert.Equal(0, res[0, 0].item<float>());
            Assert.Equal(1, res[6, 0].item<float>());
            Assert.Equal(1, res[7, 0].item<float>());

            res = a.slice_scatter(torch.ones(2, 8), start: 5, step: 2);

            Assert.Equal(0, res[0, 0].item<float>());
            Assert.Equal(1, res[5, 0].item<float>());
            Assert.Equal(1, res[7, 0].item<float>());

            res = a.slice_scatter(torch.ones(8, 2), dim: 1, start: 6);

            Assert.Equal(0, res[0, 0].item<float>());
            Assert.Equal(1, res[0, 6].item<float>());
            Assert.Equal(1, res[0, 7].item<float>());
        }

        [Fact]
        [TestOf(nameof(torch.CPU))]
        public void StackCpu()
        {
            StackGen(torch.CPU);
        }

        [Fact]
        [TestOf(nameof(torch.CUDA))]
        public void StackCuda()
        {
            if (torch.cuda.is_available()) {
                StackGen(torch.CUDA);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.block_diag))]
        public void BlockDiag()
        {
            // Example from PyTorch documentation
            var A = torch.tensor(new long[] { 0, 1, 1, 0 }, 2, 2, int64);
            var B = torch.tensor(new long[] { 3, 4, 5, 6, 7, 8 }, 2, 3, int64);
            var C = torch.tensor(7, int64);
            var D = torch.tensor(new long[] { 1, 2, 3 }, int64);
            var E = torch.tensor(new long[] { 4, 5, 6 }, 3, 1, int64);

            var expected = torch.tensor(new long[] {
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 3, 4, 5, 0, 0, 0, 0, 0,
                0, 0, 6, 7, 8, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 6 }, 9, 10);

            var res = Tensor.block_diag(A, B, C, D, E);
            Assert.Equal(expected, res);
        }

        [Fact]
        [TestOf(nameof(Tensor.diag))]
        public void Diag1D()
        {
            var input = torch.ones(new long[] { 3 }, int64);
            var expected = new long[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            var res = input.diag();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.diag))]
        public void Diag2D()
        {
            var input = torch.zeros(new long[] { 5, 5 }, int64);
            for (int i = 0; i < 5; i++) {
                input[i, i] = torch.tensor(1, int64);
            }

            var expected = new long[] { 1, 1, 1, 1, 1 };

            var res = input.diag();
            Assert.Equal(1, res.Dimensions);
            Assert.Equal(expected, res.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.diagflat))]
        public void DiagFlat1D()
        {
            var input = torch.ones(new long[] { 3 }, int64);
            var expected = new long[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            var res = input.diagflat();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.diagflat))]
        public void DiagFlat2D()
        {
            var input = torch.ones(new long[] { 2, 2 }, int64);

            var expected = new long[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

            var res = input.diagflat();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.dim))]
        [TestOf(nameof(Tensor.Dimensions))]
        public void Dimensions()
        {
            {
                var res = torch.rand(new long[] { 5 });
                Assert.Equal(1, res.Dimensions);
                Assert.Equal(1, res.dim());
            }
            {
                var res = torch.rand(new long[] { 5, 5 });
                Assert.Equal(2, res.Dimensions);
                Assert.Equal(2, res.dim());
            }
            {
                var res = torch.rand(new long[] { 5, 5, 5 });
                Assert.Equal(3, res.Dimensions);
                Assert.Equal(3, res.dim());
            }
            {
                var res = torch.rand(new long[] { 5, 5, 5, 5 });
                Assert.Equal(4, res.Dimensions);
                Assert.Equal(4, res.dim());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.numel))]
        [TestOf(nameof(Tensor.NumberOfElements))]
        public void NumberofElements()
        {
            {
                var res = torch.rand(new long[] { 5 });
                Assert.Equal(5, res.NumberOfElements);
                Assert.Equal(5, res.numel());
            }
            {
                var res = torch.rand(new long[] { 5, 5 });
                Assert.Equal(25, res.NumberOfElements);
                Assert.Equal(25, res.numel());
            }
            {
                var res = torch.rand(new long[] { 5, 5, 5 });
                Assert.Equal(125, res.NumberOfElements);
                Assert.Equal(125, res.numel());
            }
            {
                var res = torch.rand(new long[] { 5, 5, 5, 5 });
                Assert.Equal(625, res.NumberOfElements);
                Assert.Equal(625, res.numel());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.element_size))]
        [TestOf(nameof(Tensor.ElementSize))]
        public void ElementSize()
        {
            {
                var res = torch.randint(100, new long[] { 5 }, int8);
                Assert.Equal(1, res.ElementSize);
                Assert.Equal(1, res.element_size());
            }
            {
                var res = torch.randint(250, new long[] { 5 }, int16);
                Assert.Equal(2, res.ElementSize);
                Assert.Equal(2, res.element_size());
            }
            {
                var res = torch.randint(250, new long[] { 5 }, int32);
                Assert.Equal(4, res.ElementSize);
                Assert.Equal(4, res.element_size());
            }
            {
                var res = torch.randint(250, new long[] { 5 }, int64);
                Assert.Equal(8, res.ElementSize);
                Assert.Equal(8, res.element_size());
            }
            {
                var res = torch.rand(new long[] { 5 });
                Assert.Equal(4, res.ElementSize);
                Assert.Equal(4, res.element_size());
            }
            {
                var res = torch.rand(new long[] { 5 }, float64);
                Assert.Equal(8, res.ElementSize);
                Assert.Equal(8, res.element_size());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.atleast_1d))]
        public void Atleast1d()
        {
            {
                var input = torch.tensor(1.0f);
                var res = input.atleast_1d();
                Assert.Equal(1, res.Dimensions);
                Assert.Equal(1, res.dim());
                Assert.Equal(1, res.NumberOfElements);
                Assert.Equal(1, res.numel());
            }
            {
                var input = torch.rand(new long[] { 5 });
                var res = input.atleast_1d();
                Assert.Equal(1, res.Dimensions);
                Assert.Equal(1, res.dim());
                Assert.Equal(5, res.NumberOfElements);
                Assert.Equal(5, res.numel());
            }
            {
                var input = torch.rand(new long[] { 5, 5 });
                var res = input.atleast_1d();
                Assert.Equal(2, res.Dimensions);
                Assert.Equal(2, res.dim());
                Assert.Equal(25, res.NumberOfElements);
                Assert.Equal(25, res.numel());
            }
            {
                var input = torch.rand(new long[] { 5, 5, 5 });
                var res = input.atleast_1d();
                Assert.Equal(3, res.Dimensions);
                Assert.Equal(3, res.dim());
                Assert.Equal(125, res.NumberOfElements);
                Assert.Equal(125, res.numel());
            }
            {
                var input = torch.rand(new long[] { 5, 5, 5, 5 });
                var res = input.atleast_1d();
                Assert.Equal(4, res.Dimensions);
                Assert.Equal(4, res.dim());
                Assert.Equal(625, res.NumberOfElements);
                Assert.Equal(625, res.numel());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.atleast_2d))]
        public void Atleast2d()
        {
            {
                var input = torch.tensor(1.0f);
                var res = input.atleast_2d();
                Assert.Equal(2, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5 });
                var res = input.atleast_2d();
                Assert.Equal(2, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5, 5 });
                var res = input.atleast_2d();
                Assert.Equal(2, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5, 5, 5 });
                var res = input.atleast_2d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5, 5, 5, 5 });
                var res = input.atleast_2d();
                Assert.Equal(4, res.Dimensions);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.atleast_3d))]
        public void Atleast3d()
        {
            {
                var input = torch.tensor(1.0f);
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5 });
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5, 5 });
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5, 5, 5 });
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = torch.rand(new long[] { 5, 5, 5, 5 });
                var res = input.atleast_3d();
                Assert.Equal(4, res.Dimensions);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.requires_grad))]
        public void SetGrad()
        {
            var x = torch.rand(new long[] { 10, 10 });
            Assert.False(x.requires_grad);

            x.requires_grad = true;
            Assert.True(x.requires_grad);
            x.requires_grad = false;
            Assert.False(x.requires_grad);
        }

        [Fact]
        [TestOf(nameof(Tensor.grad))]
        public void AutoGradMode()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var x = torch.randn(new long[] { 2, 3 }, requires_grad: true);
                using (torch.no_grad()) {
                    Assert.False(torch.is_grad_enabled());
                    var sum = x.sum();
                    Assert.Throws<ExternalException>(() => sum.backward());
                    //var grad = x.Grad();
                    //Assert.True(grad.Handle == IntPtr.Zero);
                }
                using (torch.enable_grad()) {
                    Assert.True(torch.is_grad_enabled());
                    var sum = x.sum();
                    sum.backward();
                    var grad = x.grad;
                    Assert.False(grad is null || grad.Handle == IntPtr.Zero);
                    var data = grad is not null ? grad.data<float>().ToArray() : new float[] { };
                    for (int i = 0; i < 2 * 3; i++) {
                        Assert.Equal(1.0, data[i]);
                    }
                }
                x = torch.randn(new long[] { 2, 3 }, requires_grad: true);
                using (torch.set_grad_enabled(false)) {
                    Assert.False(torch.is_grad_enabled());
                    var sum = x.sum();
                    Assert.Throws<ExternalException>(() => sum.backward());
                    //var grad = x.Grad();
                    //Assert.True(grad.Handle == IntPtr.Zero);
                }
                using (torch.set_grad_enabled(true)) {
                    Assert.True(torch.is_grad_enabled());
                    var sum = x.sum();
                    sum.backward();
                    var grad = x.grad;
                    Assert.False(grad is not null && grad.Handle == IntPtr.Zero);
                    var data = grad is not null ? grad.data<float>().ToArray() : new float[] { };
                    for (int i = 0; i < 2 * 3; i++) {
                        Assert.Equal(1.0, data[i]);
                    }
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.sub_))]
        public void SubInPlace()
        {
            var x = torch.ones(new long[] { 100, 100 }, int32);
            var y = torch.ones(new long[] { 100, 100 }, int32);

            x.sub_(y);

            var xdata = x.data<int>();

            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    Assert.Equal(0, xdata[i + j]);
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void MemoryDisposalZeros()
        {
            for (int i = 0; i < 1024; i++) {
                var x = torch.zeros(new long[] { 1024, 1024 }, float64);
                x.Dispose();
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void MemoryDisposalOnes()
        {
            for (int i = 0; i < 1024; i++) {
                var x = torch.ones(new long[] { 1024, 1024 }, float64);
                x.Dispose();
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void MemoryDisposalScalarTensors()
        {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 1000 * 100; j++) {
                    var x = torch.tensor(i * j * 3.1415);
                    x.Dispose();
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void MemoryDisposalScalars()
        {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 1000 * 100; j++) {
                    var x = (i * j * 3.1415).ToScalar();
                    x.Dispose();
                }
            }
        }


        [Fact]
        [TestOf(nameof(Tensor.save))]
        [TestOf(nameof(Tensor.load))]
        public void SaveLoadTensorDouble()
        {
            var file = ".saveload.double.ts";
            if (File.Exists(file)) File.Delete(file);
            var tensor = torch.ones(new long[] { 5, 6 }, float64);
            tensor.save(file);
            var tensorLoaded = Tensor.load(file);
            File.Delete(file);
            Assert.NotNull(tensorLoaded);
            Assert.Equal(tensorLoaded.dtype, tensor.dtype);
            Assert.Equal(tensorLoaded, tensor);
        }

        [Fact]
        [TestOf(nameof(Tensor.save))]
        [TestOf(nameof(Tensor.load))]
        public void SaveLoadTensorFloat()
        {
            var file = ".saveload.float.ts";
            if (File.Exists(file)) File.Delete(file);
            var tensor = torch.ones(new long[] { 5, 6 });
            tensor.save(file);
            var tensorLoaded = Tensor.load(file);
            File.Delete(file);
            Assert.NotNull(tensorLoaded);
            Assert.Equal(tensorLoaded.dtype, tensor.dtype);
            Assert.Equal(tensorLoaded, tensor);
        }

        [Fact]
        [TestOf(nameof(Tensor.positive))]
        public void Positive()
        {
            var a = torch.randn(25, 25);
            var b = a.positive();

            Assert.Equal(a.data<float>().ToArray(), b.data<float>().ToArray());

            var c = torch.ones(25, 25, @bool);
            Assert.Throws<ArgumentException>(() => c.positive());
        }

        [Fact]
        [TestOf(nameof(Tensor.where))]
        public void WhereTest()
        {
            var bits = 3;
            var mask = -(1 << (8 - bits));
            var condition = torch.rand(25) > 0.5;
            var ones = torch.ones(25, int32);
            var zeros = torch.zeros(25, int32);

            var cond1 = ones.where(condition, zeros);
            var cond2 = condition.to_type(ScalarType.Int32);
            Assert.Equal(cond1, cond2);
        }

        [Fact]
        [TestOf(nameof(torch.where))]
        public void WhereTest1()
        {
            var input = new bool[] { true, true, true, false, true };
            var expected = new Tensor[] { torch.tensor(new long[] { 0, 1, 2, 4 }) };

            var res = torch.where(torch.tensor(input));
            Assert.Equal(expected, res);

            var input1 = new bool[,] { { true, true, false, false },
                                       { false, true, false, false },
                                       { false, false, false, true },
                                       { false, false, true, false }};
            var expected1 = new Tensor[] { torch.tensor(new long[] { 0, 0, 1, 2, 3 }),
                                           torch.tensor(new long[] { 0, 1, 1, 3, 2 })};
            var res1 = torch.where(torch.tensor(input1));
            Assert.Equal(expected1, res1);
        }

        [Fact]
        [TestOf(nameof(Tensor.heaviside))]
        public void HeavisideTest()
        {
            var input = new float[] { -1.0f, 0.0f, 3.0f };
            var values = new float[] { 1.0f, 2.0f, 1.0f };
            var expected = new float[] { 0.0f, 2.0f, 1.0f };
            var res = torch.tensor(input).heaviside(torch.tensor(values));
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.heaviside_))]
        public void HeavisideInPlaceTest()
        {
            var input = new float[] { -1.0f, 0.0f, 3.0f };
            var values = new float[] { 1.0f, 2.0f, 1.0f };
            var expected = new float[] { 0.0f, 2.0f, 1.0f };
            {
                var t = torch.tensor(input);
                var res = t.heaviside_(torch.tensor(values));
                Assert.Same(t, res);
                Assert.True(res.allclose(torch.tensor(expected)));
            }
            {
                var t = torch.tensor(input);
                var res = torch.heaviside_(t, torch.tensor(values));
                Assert.Same(t, res);
                Assert.True(res.allclose(torch.tensor(expected)));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.maximum))]
        public void MaximumTest()
        {
            var a = torch.tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var b = a.neg();
            var expected = a;
            var res = a.maximum(b);
            Assert.Equal(expected, res);
        }

        [Fact]
        [TestOf(nameof(Tensor.minimum))]
        public void MinimumTest()
        {
            var a = torch.tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var b = a.neg();
            var expected = b;
            var res = a.minimum(b);
            Assert.Equal(expected, res);
        }

        [Fact]
        [TestOf(nameof(Tensor.argmax))]
        public void ArgMaxTest()
        {
            var a = torch.randn(new long[] { 15, 5 });
            var b = a.argmax();
            Assert.Equal(1, b.NumberOfElements);
            var c = a.argmax(0, keepdim: true);
            Assert.Equal(new long[] { 1, 5 }, c.shape);
            var d = a.argmax(0, keepdim: false);
            Assert.Equal(new long[] { 5 }, d.shape);
        }

        [Fact]
        [TestOf(nameof(torch.argmax))]
        public void ArgMaxFuncTest()
        {
            var a = torch.arange(3, 15).reshape(3, 4);
            var b = torch.argmax(a);
            Assert.Equal(11, b.item<long>());
            var c = torch.argmax(a, dim: 0, keepdim: true);
            Assert.Equal(new long[] { 1, 4 }, c.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.argmin))]
        public void ArgMinTest()
        {
            var a = torch.randn(new long[] { 15, 5 });
            var b = a.argmin();
            Assert.Equal(1, b.NumberOfElements);
            var c = a.argmin(0, keepdim: true);
            Assert.Equal(new long[] { 1, 5 }, c.shape);
            var d = a.argmin(0, keepdim: false);
            Assert.Equal(new long[] { 5 }, d.shape);
        }

        [Fact]
        [TestOf(nameof(torch.argmin))]
        public void ArgMinFuncTest()
        {
            var a = torch.arange(3, 15).reshape(3, 4);
            var b = torch.argmin(a);
            Assert.Equal(0, b.item<long>());
            var c = torch.argmin(a, dim: 1, keepdim: true);
            Assert.Equal(new long[] { 3, 1 }, c.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.amax))]
        public void AMaxTest()
        {
            var a = torch.randn(new long[] { 15, 5, 4, 3 });
            var b = a.amax(0, 1);
            Assert.Equal(new long[] { 4, 3 }, b.shape);
            var c = a.amax(new long[] { 0, 1 }, keepdim: true);
            Assert.Equal(new long[] { 1, 1, 4, 3 }, c.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.amin))]
        public void AMinTest()
        {
            var a = torch.randn(new long[] { 15, 5, 4, 3 });
            var b = a.amin(0, 1);
            Assert.Equal(new long[] { 4, 3 }, b.shape);
            var c = a.amin(new long[] { 0, 1 }, keepdim: true);
            Assert.Equal(new long[] { 1, 1, 4, 3 }, c.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.aminmax))]
        public void AMinMaxTest()
        {
            var a = torch.randn(new long[] { 15, 5, 4, 3 });
            var b = a.aminmax(0);
            Assert.Equal(new long[] { 5, 4, 3 }, b.min.shape);
            Assert.Equal(new long[] { 5, 4, 3 }, b.max.shape);
            var c = a.aminmax(0, keepdim: true);
            Assert.Equal(new long[] { 1, 5, 4, 3 }, c.min.shape);
            Assert.Equal(new long[] { 1, 5, 4, 3 }, c.max.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.cov))]
        public void CovarianceTest()
        {
            var data = new float[] { 0, 2, 1, 1, 2, 0 };
            var expected = new float[] { 1, -1, -1, 1 };
            var res = torch.tensor(data).reshape(3, 2).T;
            {
                var cov1 = res.cov();
                Assert.True(cov1.allclose(torch.tensor(expected).reshape(2, 2)));
            }
            {
                var cov1 = torch.cov(res);
                Assert.True(cov1.allclose(torch.tensor(expected).reshape(2, 2)));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.logit))]
        public void LogitTest()
        {
            // From the PyTorch reference docs.
            var data = new float[] { 0.2796f, 0.9331f, 0.6486f, 0.1523f, 0.6516f };
            var expected = new float[] { -0.946446538f, 2.635313f, 0.6128909f, -1.71667457f, 0.6260796f };
            var res = torch.tensor(data).logit(eps: 1f - 6);
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.logit_))]
        public void LogitInPlaceTest()
        {
            var data = new float[] { 0.2796f, 0.9331f, 0.6486f, 0.1523f, 0.6516f };
            var expected = new float[] { -0.946446538f, 2.635313f, 0.6128909f, -1.71667457f, 0.6260796f };
            {
                var input = torch.tensor(data);
                var res = input.logit_(eps: 1f - 6);
                Assert.Same(input, res);
                Assert.True(res.allclose(torch.tensor(expected)));
            }
            {
                var input = torch.tensor(data);
                var res = torch.logit_(input, eps: 1f - 6);
                Assert.Same(input, res);
                Assert.True(res.allclose(torch.tensor(expected)));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.logcumsumexp))]
        public void LogCumSumExpTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };
            var expected = new float[data.Length];
            for (int i = 0; i < data.Length; i++) {
                for (int j = 0; j <= i; j++) {
                    expected[i] += MathF.Exp(data[j]);
                }
                expected[i] = MathF.Log(expected[i]);
            }
            var res = torch.tensor(data).logcumsumexp(dim: 0);
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.outer))]
        public void OuterTest()
        {
            var x = torch.arange(1, 5, 1, float32);
            var y = torch.arange(1, 4, 1, float32);
            var expected = new float[] { 1, 2, 3, 2, 4, 6, 3, 6, 9, 4, 8, 12 };

            var res = x.outer(y);
            Assert.Equal(torch.tensor(expected, 4, 3), res);
        }

        [Fact]
        [TestOf(nameof(Tensor.conj))]
        public void ConjTest()
        {
            var input = torch.randn(10, dtype: complex64);
            Assert.False(input.is_conj());

            var res = input.conj();
            Assert.Equal(10, res.shape[0]);
            Assert.True(torch.is_conj(res));

            var resolved = torch.resolve_conj(res);
            Assert.Equal(10, res.shape[0]);
            Assert.False(resolved.is_conj());

            var physical = torch.conj_physical(input);
            Assert.Equal(10, res.shape[0]);
            Assert.False(physical.is_conj());
        }

        [Fact]
        [TestOf(nameof(Tensor.bucketize))]
        public void BucketizeTest()
        {
            var boundaries = torch.tensor(new int[] { 1, 3, 5, 7, 9 });
            var tensor = torch.tensor(new int[] { 3, 6, 9, 3, 6, 9 }, 2, 3);
            {
                var res = tensor.bucketize(boundaries, true, false);
                var expected = torch.tensor(new int[] { 1, 3, 4, 1, 3, 4 }, 2, 3);
                Assert.True(res.allclose(expected));
            }
            {
                var res = tensor.bucketize(boundaries, true, true);
                var expected = torch.tensor(new int[] { 2, 3, 5, 2, 3, 5 }, 2, 3);
                Assert.True(res.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.channel_shuffle))]
        public void ChannelShuffleTest()
        {
            var tensor = torch.tensor(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, 1, 4, 2, 2);
            {
                var res = tensor.channel_shuffle(2);
                var expected = torch.tensor(new int[] { 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16 }, 1, 4, 2, 2);
                Assert.True(res.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.expand))]
        public void ExpandTest()
        {
            Tensor ones = torch.ones(new long[] { 2 });
            Tensor onesExpanded = ones.expand(new long[] { 3, 2 });

            Assert.Equal(onesExpanded.shape, new long[] { 3, 2 });
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    Assert.Equal(1.0, onesExpanded[i, j].ToSingle());
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.movedim))]
        public void MovedimTest()
        {
            Tensor input = torch.randn(new long[] { 3, 2, 1 });
            {
                var res = input.movedim(new long[] { 1 }, new long[] { 0 });
                Assert.Equal(new long[] { 2, 3, 1 }, res.shape);
            }
            {
                var res = input.movedim(new long[] { 1, 2 }, new long[] { 0, 1 });
                Assert.Equal(new long[] { 2, 1, 3 }, res.shape);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.count_nonzero))]
        public void CountNZTest()
        {
            Tensor input = torch.tensor(new float[] { 0, 1, 1, 0, 0, 0, 0, 0, 1 }, 3, 3);
            {
                var res = input.count_nonzero();
                Assert.Equal(torch.tensor(3, int64), res);
            }
            {
                var res = input.count_nonzero(new long[] { 0 });
                Assert.Equal(torch.tensor(new long[] { 0, 1, 2 }), res);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.sort))]
        public void SortTest1()
        {
            var input = torch.tensor(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expectedValues = torch.tensor(new double[] {
                -1.2631, -1.1289, -0.1321, 0.4370,
                -2.0527, -1.1250,  0.2275, 0.3077,
                -0.5495, -0.1259, -0.0881, 1.0284
            }, 3, 4);

            var expectedIndices = torch.tensor(new long[] {
                2, 3, 0, 1,
                0, 1, 2, 3,
                2, 1, 0, 3
            }, 3, 4);

            var res = input.sort();
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        [TestOf(nameof(Tensor.sort))]
        public void SortTest2()
        {
            var input = torch.tensor(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expectedValues = torch.tensor(new double[] {
                -2.0527, -1.1250, -1.2631, -1.1289,
                -0.1321, -0.1259, -0.5495,  0.3077,
                -0.0881,  0.4370,  0.2275,  1.0284
            }, 3, 4);

            var expectedIndices = torch.tensor(new long[] {
                1, 1, 0, 0,
                0, 2, 2, 1,
                2, 0, 1, 2
            }, 3, 4);

            var res = input.sort(dim: 0);
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        [TestOf(nameof(Tensor.sort))]
        public void SortTest3()
        {
            var input = torch.tensor(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284,
            }, 3, 4);

            var expectedValues = torch.tensor(new double[] {
                0.4370, -0.1321, -1.1289, -1.2631,
                0.3077,  0.2275, -1.1250, -2.0527,
                1.0284, -0.0881, -0.1259, -0.5495,
            }, 3, 4);

            var expectedIndices = torch.tensor(new long[] {
                1, 0, 3, 2,
                3, 2, 1, 0,
                3, 0, 1, 2,
            }, 3, 4);

            var res = input.sort(descending: true);
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        [TestOf(nameof(Tensor.sort))]
        public void SortTest4()
        {
            var input = torch.tensor(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expectedValues = torch.tensor(new double[] {
                -0.0881,  0.4370,  0.2275,  1.0284,
                -0.1321, -0.1259, -0.5495,  0.3077,
                -2.0527, -1.1250, -1.2631, -1.1289,
            }, 3, 4);

            var expectedIndices = torch.tensor(new long[] {
                2, 0, 1, 2,
                0, 2, 2, 1,
                1, 1, 0, 0,
            }, 3, 4);

            var res = input.sort(dim: 0, descending: true);
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        [TestOf(nameof(Tensor.msort))]
        public void MSortTest()
        {
            var input = torch.tensor(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expected = torch.tensor(new double[] {
                -2.0527, -1.1250, -1.2631, -1.1289,
                -0.1321, -0.1259, -0.5495,  0.3077,
                -0.0881,  0.4370,  0.2275,  1.0284
            }, 3, 4);

            var res = input.msort();
            Assert.True(res.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(Tensor.repeat))]
        public void RepeatTest()
        {
            var input = torch.tensor(new int[] { 1, 2, 3 });
            var expected = torch.tensor(new int[] {
                1,  2,  3,  1,  2,  3,
                1,  2,  3,  1,  2,  3,
                1,  2,  3,  1,  2,  3,
                1,  2,  3,  1,  2,  3
            }).reshape(4, 6);

            var res = input.repeat(new long[] { 4, 2 });
            Assert.Equal(res, expected);
        }

        [Fact]
        [TestOf(nameof(Tensor.repeat_interleave))]
        public void RepeatInterleaveTest()
        {
            {
                var input = torch.tensor(new int[] { 1, 2, 3 });
                var expected = torch.tensor(new int[] {
                    1, 1, 1, 2, 2, 2, 3, 3, 3
                });

                var res = input.repeat_interleave(torch.tensor(new long[] { 3 }));
                Assert.Equal(res, expected);
            }

            {
                var input = torch.tensor(new int[] { 1, 2, 3 });
                var expected = torch.tensor(new int[] {
                    1, 1, 1, 2, 2, 2, 3, 3, 3
                });

                var res = input.repeat_interleave(3, output_size: 9);
                Assert.Equal(res, expected);
            }

            {
                var input = torch.tensor(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
                var expected = torch.tensor(new int[] {
                    1, 2, 3,
                    1, 2, 3,
                    4, 5, 6,
                    4, 5, 6,
                    4, 5, 6
                }).reshape(5, 3);

                var res = input.repeat_interleave(torch.tensor(new long[] { 2, 3 }), dim: 0);
                Assert.Equal(res, expected);
            }

            {
                var input = torch.tensor(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
                var expected = torch.tensor(new int[] {
                    1, 1, 1, 2, 2, 2, 3, 3, 3,
                    4, 4, 4, 5, 5, 5, 6, 6, 6
                }).reshape(2, 9);

                var res = input.repeat_interleave(3, dim: 1);
                Assert.Equal(res, expected);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.fliplr))]
        public void FlipLRTest()
        {
            var input = torch.tensor(new int[] {
                1,  2,  3,
                1,  2,  3,
                1,  2,  3,
            }).reshape(3, 3);

            var expected = torch.tensor(new int[] {
                3,  2,  1,
                3,  2,  1,
                3,  2,  1,
            }).reshape(3, 3);

            var res = input.fliplr();
            Assert.Equal(res, expected);
        }

        [Fact]
        [TestOf(nameof(Tensor.flipud))]
        public void FlipUDTest()
        {
            var input = torch.tensor(new int[] {
                1,  1,  1,
                2,  2,  2,
                3,  3,  3,
            }).reshape(3, 3);

            var expected = torch.tensor(new int[] {
                3,  3,  3,
                2,  2,  2,
                1,  1,  1,
            }).reshape(3, 3);

            var res = input.flipud();
            Assert.Equal(res, expected);
        }

        [Fact]
        [TestOf(nameof(Tensor.topk))]
        public void TopKTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res1 = torch.tensor(data).topk(1);
            var res1_0 = res1.values[0].ToSingle();
            var index1_0 = res1.indexes[0].ToInt64();
            Assert.Equal(3.1f, res1_0);
            Assert.Equal(2L, index1_0);

            var res2 = torch.tensor(data).topk(2, sorted: true);
            var res2_0 = res2.values[0].ToSingle();
            var index2_0 = res2.indexes[0].ToInt64();
            var res2_1 = res2.values[1].ToSingle();
            var index2_1 = res2.indexes[1].ToInt64();
            Assert.Equal(3.1f, res2_0);
            Assert.Equal(2L, index2_0);
            Assert.Equal(2.0f, res2_1);
            Assert.Equal(1L, index2_1);
        }

        [Fact]
        [TestOf(nameof(Tensor.sum))]
        public void SumTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };

            var res1 = torch.tensor(data).sum();
            var res1_0 = res1.ToSingle();
            Assert.Equal(6.0f, res1_0);

            var res2 = torch.tensor(data).sum(type: ScalarType.Float64);
            var res2_0 = res2.ToDouble();
            Assert.Equal(6.0, res2_0);

            // summing integers gives long unless type is explicitly specified
            var dataInt32 = new int[] { 1, 2, 3 };
            var res3 = torch.tensor(dataInt32).sum();
            Assert.Equal(ScalarType.Int64, res3.dtype);
            var res3_0 = res3.ToInt64();
            Assert.Equal(6L, res3_0);

            // summing integers gives long unless type is explicitly specified
            var res4 = torch.tensor(dataInt32).sum(type: ScalarType.Int32);
            Assert.Equal(ScalarType.Int32, res4.dtype);
            var res4_0 = res4.ToInt32();
            Assert.Equal(6L, res4_0);

        }

        [Fact]
        [TestOf(nameof(torch.sum))]
        public void SumFuncTest()
        {
            var a = torch.arange(3, 15).reshape(3, 4);
            var b = torch.sum(a);
            Assert.Equal(102L, b.item<long>());
            var c = torch.sum(a, dim: 1);
            Assert.Equal(new long[] { 18, 34, 50 }, c.data<long>());
        }

        [Fact]
        [TestOf(nameof(torch.prod))]
        public void ProdFuncTest()
        {
            var a = torch.arange(1, 11).reshape(2, 5);
            var b = torch.prod(a);
            Assert.Equal(3628800, b.item<long>());
            var c = torch.prod(a, dim: 1);
            Assert.Equal(new long[] { 120, 30240 }, c.data<long>());
        }

        [Fact]
        [TestOf(nameof(Tensor.std))]
        public void StdTest()
        {
            var data = torch.rand(10, 10, 10);

            {
                var std = data.std();
                Assert.NotNull(std);
                Assert.Empty(std.shape);
            }
            {
                var std = torch.std(data, unbiased: false);
                Assert.NotNull(std);
                Assert.Empty(std.shape);
            }
            {
                var std = data.std(1);
                Assert.NotNull(std);
                Assert.Equal(new long[] { 10, 10 }, std.shape);
            }
            {
                var std = torch.std(data, 1, keepdim: true);
                Assert.NotNull(std);
                Assert.Equal(new long[] { 10, 1, 10 }, std.shape);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.std_mean))]
        public void StdMeanTest()
        {
            var data = torch.rand(10, 10, 10);

            {
                var (std, mean) = data.std_mean();
                Assert.NotNull(std);
                Assert.NotNull(mean);
                Assert.Empty(std.shape);
                Assert.Empty(mean.shape);
            }
            {
                var (std, mean) = torch.std_mean(data, unbiased: false);
                Assert.NotNull(std);
                Assert.NotNull(mean);
                Assert.Empty(std.shape);
                Assert.Empty(mean.shape);
            }
            {
                var (std, mean) = data.std_mean(1);
                Assert.NotNull(std);
                Assert.NotNull(mean);
                Assert.Equal(new long[] { 10, 10 }, std.shape);
                Assert.Equal(new long[] { 10, 10 }, mean.shape);
            }
            {
                var (std, mean) = torch.std_mean(data, 1, keepdim: true);
                Assert.NotNull(std);
                Assert.NotNull(mean);
                Assert.Equal(new long[] { 10, 1, 10 }, std.shape);
                Assert.Equal(new long[] { 10, 1, 10 }, mean.shape);
            }
            {
                var t = torch.from_array(new float[,] { { 1f, 2f }, { 3f, 4f } });
                var stdExpected = torch.std(t);
                var meanExpected = torch.mean(t);
                var (std, mean) = torch.std_mean(t);
                Assert.NotNull(std);
                Assert.NotNull(mean);
                Assert.True(stdExpected.allclose(std));
                Assert.True(meanExpected.allclose(mean));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.var_mean))]
        public void VarMeanTest()
        {
            var data = torch.rand(10, 10, 10);

            {
                var (@var, mean) = data.var_mean();
                Assert.NotNull(@var);
                Assert.NotNull(mean);
                Assert.Empty(@var.shape);
                Assert.Empty(mean.shape);
            }
            {
                var (@var, mean) = torch.var_mean(data, unbiased: false);
                Assert.NotNull(@var);
                Assert.NotNull(mean);
                Assert.Empty(@var.shape);
                Assert.Empty(mean.shape);
            }
            {
                var (@var, mean) = data.var_mean(1);
                Assert.NotNull(@var);
                Assert.NotNull(mean);
                Assert.Equal(new long[] { 10, 10 }, @var.shape);
                Assert.Equal(new long[] { 10, 10 }, mean.shape);
            }
            {
                var (@var, mean) = torch.var_mean(data, 1, keepdim: true);
                Assert.NotNull(@var);
                Assert.NotNull(mean);
                Assert.Equal(new long[] { 10, 1, 10 }, @var.shape);
                Assert.Equal(new long[] { 10, 1, 10 }, mean.shape);
            }
            //{
            //    var t = torch.from_array(new float[,]{ { 1f, 2f }, { 3f, 4f } });
            //    var varExpected = torch.var(t);
            //    var meanExpected = torch.mean(t);
            //    var (@var, mean) = torch.var_mean(t);
            //    Assert.NotNull(@var);
            //    Assert.NotNull(mean);
            //    Assert.True(varExpected.allclose(@var));
            //    Assert.True(meanExpected.allclose(mean));
            //}
        }

        [Fact]
        [TestOf(nameof(Tensor.unbind))]
        public void UnbindTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = torch.tensor(data).unbind();
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { }, res[0].shape);
            Assert.Equal(new long[] { }, res[1].shape);
            Assert.Equal(new long[] { }, res[2].shape);
            Assert.Equal(1.1f, res[0].ToSingle());
            Assert.Equal(2.0f, res[1].ToSingle());
            Assert.Equal(3.1f, res[2].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.unfold))]
        public void UnfoldTest()
        {
            var data = torch.arange(1, 8);

            var res = data.unfold(0, 2, 1);
            Assert.Equal(new long[] { 6, 2 }, res.shape);

            res = data.unfold(0, 2, 2);
            Assert.Equal(new long[] { 3, 2 }, res.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.split))]
        public void SplitWithSizeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = torch.tensor(data).split(2);
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { 2 }, res[0].shape);
            Assert.Equal(new long[] { 2 }, res[1].shape);
            Assert.Equal(new long[] { 1 }, res[2].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[0][1].ToSingle());
            Assert.Equal(3.1f, res[1][0].ToSingle());
            Assert.Equal(4.2f, res[1][1].ToSingle());
            Assert.Equal(5.3f, res[2][0].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.split))]
        public void SplitWithSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = torch.tensor(data).split(new long[] { 2, 1 });
            Assert.Equal(2, res.Length);
            Assert.Equal(new long[] { 2 }, res[0].shape);
            Assert.Equal(new long[] { 1 }, res[1].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[0][1].ToSingle());
            Assert.Equal(3.1f, res[1][0].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.tensor_split))]
        public void TensorSplitWithSizeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = torch.tensor(data).tensor_split(2);
            Assert.Equal(2, res.Length);
            Assert.Equal(new long[] { 3 }, res[0].shape);
            Assert.Equal(new long[] { 2 }, res[1].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[0][1].ToSingle());
            Assert.Equal(3.1f, res[0][2].ToSingle());
            Assert.Equal(4.2f, res[1][0].ToSingle());
            Assert.Equal(5.3f, res[1][1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.tensor_split))]
        public void TensorSplitWithSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = torch.tensor(data).tensor_split(new long[] { 1, 4 });
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { 1 }, res[0].shape);
            Assert.Equal(new long[] { 3 }, res[1].shape);
            Assert.Equal(new long[] { 1 }, res[2].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[1][0].ToSingle());
            Assert.Equal(3.1f, res[1][1].ToSingle());
            Assert.Equal(4.2f, res[1][2].ToSingle());
            Assert.Equal(5.3f, res[2][0].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.vsplit))]
        public void VSplitWithSizeTest()
        {
            var a = torch.arange(64, int32).reshape(4, 4, 4);

            var b = a.vsplit(2);
            Assert.Equal(new long[] { 2, 4, 4 }, b[0].shape);
            Assert.Equal(new long[] { 2, 4, 4 }, b[1].shape);

            Assert.Throws<ArgumentException>(() => a.vsplit(3));
        }

        [Fact]
        [TestOf(nameof(Tensor.vsplit))]
        public void VSplitWithSizesTest()
        {
            var a = torch.arange(80, int32).reshape(5, 4, 4);

            var b = a.vsplit(new long[] { 2, 3 });
            Assert.Equal(new long[] { 2, 4, 4 }, b[0].shape);
            Assert.Equal(new long[] { 1, 4, 4 }, b[1].shape);
            Assert.Equal(new long[] { 2, 4, 4 }, b[2].shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.hsplit))]
        public void HSplitWithSizeTest()
        {
            var a = torch.arange(64, int32).reshape(4, 4, 4);

            var b = a.hsplit(2);
            Assert.Equal(new long[] { 4, 2, 4 }, b[0].shape);
            Assert.Equal(new long[] { 4, 2, 4 }, b[1].shape);

            Assert.Throws<ArgumentException>(() => a.hsplit(3));
        }

        [Fact]
        [TestOf(nameof(Tensor.hsplit))]
        public void HSplitWithSizesTest()
        {
            var a = torch.arange(80, int32).reshape(4, 5, 4);

            var b = a.hsplit(new long[] { 2, 3 });
            Assert.Equal(new long[] { 4, 2, 4 }, b[0].shape);
            Assert.Equal(new long[] { 4, 1, 4 }, b[1].shape);
            Assert.Equal(new long[] { 4, 2, 4 }, b[2].shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.dsplit))]
        public void DSplitWithSizeTest()
        {
            var a = torch.arange(64, int32).reshape(4, 4, 4);

            var b = a.dsplit(2);
            Assert.Equal(new long[] { 4, 4, 2 }, b[0].shape);
            Assert.Equal(new long[] { 4, 4, 2 }, b[1].shape);

            Assert.Throws<ArgumentException>(() => a.hsplit(3));
        }

        [Fact]
        [TestOf(nameof(Tensor.dsplit))]
        public void DSplitWithSizesTest()
        {
            var a = torch.arange(80, int32).reshape(4, 4, 5);

            var b = a.dsplit(new long[] { 2, 3 });
            Assert.Equal(new long[] { 4, 4, 2 }, b[0].shape);
            Assert.Equal(new long[] { 4, 4, 1 }, b[1].shape);
            Assert.Equal(new long[] { 4, 4, 2 }, b[2].shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.tensor_split))]
        public void TensorSplitWithTensorSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = torch.tensor(data).tensor_split(torch.tensor(new long[] { 1, 4 }));
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { 1 }, res[0].shape);
            Assert.Equal(new long[] { 3 }, res[1].shape);
            Assert.Equal(new long[] { 1 }, res[2].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[1][0].ToSingle());
            Assert.Equal(3.1f, res[1][1].ToSingle());
            Assert.Equal(4.2f, res[1][2].ToSingle());
            Assert.Equal(5.3f, res[2][0].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.chunk))]
        public void TensorChunkTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = torch.tensor(data).chunk(2);
            Assert.Equal(2, res.Length);
            Assert.Equal(new long[] { 3 }, res[0].shape);
            Assert.Equal(new long[] { 2 }, res[1].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[0][1].ToSingle());
            Assert.Equal(3.1f, res[0][2].ToSingle());
            Assert.Equal(4.2f, res[1][0].ToSingle());
            Assert.Equal(5.3f, res[1][1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.nonzero))]
        [TestOf(nameof(Tensor.nonzero_as_list))]
        public void TensorNonZeroTest()
        {
            var data = new double[] { 0.6, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 1.2, 0.5 };
            using var t = torch.tensor(data, 3, 4);
            {
                var res = t.nonzero();
                Assert.Equal(new long[] { 4, 2 }, res.shape);
            }
            {
                var res = t.nonzero_as_list();
                Assert.Equal(2, res.Count);
                Assert.Equal(4, res[0].shape[0]);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.take_along_dim))]
        public void TakeAlongDimTest()
        {
            var t = torch.tensor(new int[] { 10, 30, 20, 60, 40, 50 }).reshape(2, 3);
            var max_idx = t.argmax();
            var sort_idx = t.argsort(dim: 1);

            var x = t.take_along_dim(max_idx);
            var y = t.take_along_dim(sort_idx, dim: 1);

            Assert.Equal(60, x.item<int>());
            Assert.Equal(new int[] { 10, 20, 30, 40, 50, 60 }, y.data<int>().ToArray());
        }

        [Fact]
        [TestOf(nameof(torch.rand))]
        [TestOf(nameof(torch.randint))]
        public void RandomTest()
        {
            var res = torch.rand(new long[] { 2 });
            Assert.Equal(new long[] { 2 }, res.shape);

            var res1 = torch.randint(10, new long[] { 200 }, int16);
            Assert.Equal(new long[] { 200 }, res1.shape);

            var res2 = torch.randint(10, new long[] { 200 }, int32);
            Assert.Equal(new long[] { 200 }, res2.shape);

            var res3 = torch.randint(10, new long[] { 200 }, int64);
            Assert.Equal(new long[] { 200 }, res3.shape);

            var res4 = torch.randint(10, new long[] { 200 }, uint8);
            Assert.Equal(new long[] { 200 }, res4.shape);

            var res5 = torch.randint(10, new long[] { 200 }, int8);
            Assert.Equal(new long[] { 200 }, res5.shape);

            var res6 = torch.randint(10, new long[] { 200 }, float16);
            Assert.Equal(new long[] { 200 }, res6.shape);

            var res7 = torch.randint(10, new long[] { 200 }, bfloat16);
            Assert.Equal(new long[] { 200 }, res7.shape);

            var res8 = torch.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res8.shape);

            var res9 = torch.randint(10, 200, float64);
            Assert.Equal(new long[] { 200 }, res9.shape);

            //var res7 = torch.randint(100, new long[] { 20, 10 }, complex32);
            //Assert.Equal(new long[] { 200 }, res7.Shape);

            var res10 = torch.randint(100, (20, 10), complex64);
            Assert.Equal(new long[] { 20, 10 }, res10.shape);

            var res11 = torch.randint(10, (20, 10), complex128);
            Assert.Equal(new long[] { 20, 10 }, res11.shape);
        }


        [Fact]
        [TestOf(nameof(torch.randint_bool))]
        [TestOf(nameof(torch.randint_int))]
        [TestOf(nameof(torch.randint_long))]
        [TestOf(nameof(torch.rand_float))]
        [TestOf(nameof(torch.randn_float))]
        public void RandomScalarTest()
        {
            torch.randint_bool();   // Just check it doesn't throw.

            var res1 = torch.randint_int(15); // Exclusive upper bound
            Assert.InRange(res1, 0, 14);      // Inclusive upper bound

            var res2 = torch.randint_long(155); // Exclusive upper bound
            Assert.InRange(res1, 0, 154);       // Inclusive upper bound

            var res3 = torch.randint_int(5, 15); // Exclusive upper bound
            Assert.InRange(res3, 5, 14);         // Inclusive upper bound

            var res4 = torch.randint_long(5, 155); // Exclusive upper bound
            Assert.InRange(res4, 5, 154);          // Inclusive upper bound

            for (var i = 0; i < 1000; i++) {
                Assert.InRange(torch.rand_float(), 0, 1.0);
            }

            torch.randn_float(); // Check that it doesn't blow up.
        }

        [Fact]
        [TestOf(nameof(torch.randint_bool))]
        [TestOf(nameof(torch.randint_int))]
        [TestOf(nameof(torch.randint_long))]
        [TestOf(nameof(torch.rand_float))]
        [TestOf(nameof(torch.randn_float))]
        public void RandomScalarTest_Gen()
        {
            var gen = new Generator(4711);

            torch.randint_bool(gen);   // Just check it doesn't throw.

            var res1 = torch.randint_int(15, gen); // Exclusive upper bound
            Assert.InRange(res1, 0, 14);      // Inclusive upper bound

            var res2 = torch.randint_long(155, gen); // Exclusive upper bound
            Assert.InRange(res1, 0, 154);       // Inclusive upper bound

            var res3 = torch.randint_int(5, 15, gen); // Exclusive upper bound
            Assert.InRange(res3, 5, 14);         // Inclusive upper bound

            var res4 = torch.randint_long(5, 155, gen); // Exclusive upper bound
            Assert.InRange(res4, 5, 154);          // Inclusive upper bound

            for (var i = 0; i < 1000; i++) {
                Assert.InRange(torch.rand_float(gen), 0, 1.0);
            }

            torch.randn_float(gen); // Check that it doesn't blow up.
        }

        [Fact]
        [TestOf(nameof(torch.squeeze))]
        public void SqueezeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            using (var res = torch.tensor(data).expand(new long[] { 1, 1, 3 }).squeeze(0).squeeze(0)) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
            // Test negative dims, too.
            using (var res = torch.tensor(data).expand(new long[] { 1, 1, 3 }).squeeze(-3).squeeze(0)) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
            // And all dims.
            using (var res = torch.tensor(data).expand(new long[] { 1, 1, 3 }).squeeze()) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(torch.squeeze_))]
        public void SqueezeTest1()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            using (var x = torch.tensor(data).expand(new long[] { 1, 1, 3 })) {
                var z = x.squeeze(0);
                Assert.NotSame(x, z);
                var y = x.squeeze_(0);
                Assert.Same(x, y);
            }

            using (var x = torch.tensor(data).expand(new long[] { 1, 1, 3 })) {
                var z = x.squeeze();
                Assert.NotSame(x, z);
                var y = x.squeeze_();
                Assert.Same(x, y);
            }

            using (var res = torch.tensor(data).expand(new long[] { 1, 1, 3 }).squeeze_(0).squeeze_(0)) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
            // Test negative dims, too.
            using (var res = torch.tensor(data).expand(new long[] { 1, 1, 3 }).squeeze_(-3).squeeze_(0)) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
            // And all dims.
            using (var res = torch.tensor(data).expand(new long[] { 1, 1, 3 }).squeeze_()) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(torch.unsqueeze))]
        [TestOf(nameof(torch.unsqueeze_))]
        public void UnsqueezeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.1f };

            var input = torch.tensor(data);

            using (var res = input.unsqueeze(0)) {
                Assert.Equal(new long[] { 1, 4 }, res.shape);
                Assert.Equal(1.1f, res[0, 0].ToSingle());
                Assert.Equal(2.0f, res[0, 1].ToSingle());
                Assert.Equal(3.1f, res[0, 2].ToSingle());
                Assert.Equal(4.1f, res[0, 3].ToSingle());
            }
            using (var res = input.unsqueeze(1)) {
                Assert.Equal(new long[] { 4, 1 }, res.shape);
                Assert.Equal(1.1f, res[0, 0].ToSingle());
                Assert.Equal(2.0f, res[1, 0].ToSingle());
                Assert.Equal(3.1f, res[2, 0].ToSingle());
                Assert.Equal(4.1f, res[3, 0].ToSingle());
            }
            using (var res = input.unsqueeze_(0)) {
                Assert.Same(input, res);
                Assert.Equal(new long[] { 1, 4 }, res.shape);
                Assert.Equal(1.1f, res[0, 0].ToSingle());
                Assert.Equal(2.0f, res[0, 1].ToSingle());
                Assert.Equal(3.1f, res[0, 2].ToSingle());
                Assert.Equal(4.1f, res[0, 3].ToSingle());
            }
            input = torch.tensor(data);
            using (var res = input.unsqueeze_(1)) {
                Assert.Same(input, res);
                Assert.Equal(new long[] { 4, 1 }, res.shape);
                Assert.Equal(1.1f, res[0, 0].ToSingle());
                Assert.Equal(2.0f, res[1, 0].ToSingle());
                Assert.Equal(3.1f, res[2, 0].ToSingle());
                Assert.Equal(4.1f, res[3, 0].ToSingle());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.narrow))]
        public void NarrowTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = torch.tensor(data).narrow(0, 1, 2);
            Assert.Equal(new long[] { 2 }, res.shape);
            Assert.Equal(2.0f, res[0].ToSingle());
            Assert.Equal(3.1f, res[1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.slice))]
        public void SliceTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.0f };

            var res = torch.tensor(data).slice(0, 1, 1, 1);
            Assert.Equal(new long[] { 0 }, res.shape);

            var res2 = torch.tensor(data).slice(0, 1, 2, 1);
            Assert.Equal(new long[] { 1 }, res2.shape);
            Assert.Equal(2.0f, res2[0].ToSingle());

            var res3 = torch.tensor(data).slice(0, 1, 4, 2);
            Assert.Equal(new long[] { 2 }, res3.shape);
            Assert.Equal(2.0f, res3[0].ToSingle());
            Assert.Equal(4.0f, res3[1].ToSingle());
        }

        [Fact]
        [TestOf(nameof(Tensor.roll))]
        public void RollTest()
        {
            using var x = torch.tensor(new long[] { 1, 2, 3, 4, 5, 6, 7, 8 }).view(4, 2);
            using var expected_1 = torch.tensor(new long[] { 7, 8, 1, 2, 3, 4, 5, 6 }).view(4, 2);
            using var expected_2 = torch.tensor(new long[] { 5, 6, 7, 8, 1, 2, 3, 4 }).view(4, 2);
            using var expected_m1 = torch.tensor(new long[] { 3, 4, 5, 6, 7, 8, 1, 2 }).view(4, 2);

            using var expected_tuple = torch.tensor(new long[] { 6, 5, 8, 7, 2, 1, 4, 3 }).view(4, 2);

            Assert.Equal(expected_1, x.roll(1, 0));
            Assert.Equal(expected_2, x.roll(2, 0));
            Assert.Equal(expected_m1, x.roll(-1, 0));
            Assert.Equal(expected_tuple, x.roll((2, 1), (0, 1)));
            Assert.Equal(expected_tuple, x.roll(new long[] { 2, 1 }, new long[] { 0, 1 }));
        }

        [Fact]
        [TestOf(nameof(torch.nn.functional.conv1d))]
        public void Conv1DTest()
        {
            var t1 =
                new float[3, 4, 5]
                   {{{0.3460f, 0.4414f, 0.2384f, 0.7905f, 0.2267f},
                                     {0.5161f, 0.9032f, 0.6741f, 0.6492f, 0.8576f},
                                     {0.3373f, 0.0863f, 0.8137f, 0.2649f, 0.7125f},
                                     {0.7144f, 0.1020f, 0.0437f, 0.5316f, 0.7366f}},

                                    {{0.9871f, 0.7569f, 0.4329f, 0.1443f, 0.1515f},
                                     {0.5950f, 0.7549f, 0.8619f, 0.0196f, 0.8741f},
                                     {0.4595f, 0.7844f, 0.3580f, 0.6469f, 0.7782f},
                                     {0.0130f, 0.8869f, 0.8532f, 0.2119f, 0.8120f}},

                                    {{0.5163f, 0.5590f, 0.5155f, 0.1905f, 0.4255f},
                                     {0.0823f, 0.7887f, 0.8918f, 0.9243f, 0.1068f},
                                     {0.0337f, 0.2771f, 0.9744f, 0.0459f, 0.4082f},
                                     {0.9154f, 0.2569f, 0.9235f, 0.9234f, 0.3148f}}};
            var t2 =
                new float[2, 4, 3]
                   {{{0.4941f, 0.8710f, 0.0606f},
                     {0.2831f, 0.7930f, 0.5602f},
                     {0.0024f, 0.1236f, 0.4394f},
                     {0.9086f, 0.1277f, 0.2450f}},

                    {{0.5196f, 0.1349f, 0.0282f},
                     {0.1749f, 0.6234f, 0.5502f},
                     {0.7678f, 0.0733f, 0.3396f},
                     {0.6023f, 0.6546f, 0.3439f}}};

            var t1raw = new float[3 * 4 * 5];
            var t2raw = new float[2 * 4 * 3];
            { for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 5; k++) { t1raw[i * 4 * 5 + j * 5 + k] = t1[i, j, k]; } }
            { for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 3; k++) { t2raw[i * 4 * 3 + j * 3 + k] = t2[i, j, k]; } }
            var t1t = torch.tensor(t1raw, new long[] { 3, 4, 5 });
            var t2t = torch.tensor(t2raw, new long[] { 2, 4, 3 });

            if (torch.cuda.is_available()) {
                t1t = t1t.cuda();
                t2t = t2t.cuda();
            }
            var t3t = torch.nn.functional.conv1d(t1t, t2t, stride: 1, padding: 0, dilation: 1);
            if (torch.cuda.is_available()) {
                t3t = t3t.cpu();
            }

            // Check the answer
            var t3Correct =
                new float[3, 2, 3]
                    {{{2.8516f, 2.0732f, 2.6420f},
                      {2.3239f, 1.7078f, 2.7450f}},

                    {{3.0127f, 2.9651f, 2.5219f},
                     {3.0899f, 3.1496f, 2.4110f}},

                    {{3.4749f, 2.9038f, 2.7131f},
                     {2.7692f, 2.9444f, 3.2554f}}};
            {
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++) {
                            var itemCorrect = t3Correct[i, j, k];
                            var item = t3t[i, j, k].ToDouble();
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }

        }


        [Fact(Skip = "Failing")]
        [TestOf(nameof(torch.nn.functional.conv1d))]
        public void Conv1DTest2()
        {
            var t1 =
                new float[2, 3, 4]
                   {{{0.1264f, 5.3183f, 6.6905f, -10.6416f},
                     { 13.8060f, 4.5253f, 2.8568f, -3.2037f},
                     { -0.5796f, -2.7937f, -3.3662f, -1.3017f}},

                    {{ -2.8910f, 3.9349f, -4.3892f, -2.6051f},
                     {  4.2547f, 2.6049f, -9.8226f, -5.4543f},
                     { -0.9674f, 1.0070f, -4.6518f, 7.1702f}}};
            var t2 =
                new float[2, 3, 2]
                   {{{4.0332e+00f, 6.3036e+00f},
                     { 8.4410e+00f, -5.7543e+00f},
                     {-5.6937e-03f, -6.7241e+00f}},

                    {{-2.2619e+00f, 1.2082e+00f},
                     {-1.2203e-01f, -4.9373e+00f},
                     {-4.1881e+00f, -3.4198e+00f}}};

            var t1raw = new float[2 * 3 * 4];
            var t2raw = new float[2 * 3 * 2];
            { for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 4; k++) { t1raw[i * 3 * 4 + j * 4 + k] = t1[i, j, k]; } }
            { for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 2; k++) { t2raw[i * 3 * 2 + j * 2 + k] = t2[i, j, k]; } }
            var t1t = torch.tensor(t1raw, new long[] { 2, 3, 4 });
            var t2t = torch.tensor(t2raw, new long[] { 2, 3, 2 });

            if (torch.cuda.is_available()) {
                t1t = t1t.cuda();
                t2t = t2t.cuda();
            }
            var t3t = torch.nn.functional.conv1d(t1t, t2t, stride: 1, padding: 0, dilation: 1);
            if (torch.cuda.is_available()) {
                t3t = t3t.cpu();
            }

            // Check the answer
            var t3Correct =
                new float[2, 2, 3]
                    {{{143.3192f, 108.0332f, 11.2241f},
                      {  -5.9062f, 4.6091f, 6.0273f}},

                     {{  27.3032f, 97.9855f, -133.8372f},
                      {  -1.4792f, 45.6659f, 29.8705f}}};
            {
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++) {
                            var itemCorrect = t3Correct[i, j, k];
                            var item = t3t[i, j, k].ToDouble();
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }

        }

        [Fact]
        [TestOf(nameof(torch.nn.functional.conv1d))]
        public void Conv1DTestPadding2Dilation3()
        {
            var t1 =
                new float[3, 4, 5]
                   {{{0.3460f, 0.4414f, 0.2384f, 0.7905f, 0.2267f},
                                     {0.5161f, 0.9032f, 0.6741f, 0.6492f, 0.8576f},
                                     {0.3373f, 0.0863f, 0.8137f, 0.2649f, 0.7125f},
                                     {0.7144f, 0.1020f, 0.0437f, 0.5316f, 0.7366f}},

                                    {{0.9871f, 0.7569f, 0.4329f, 0.1443f, 0.1515f},
                                     {0.5950f, 0.7549f, 0.8619f, 0.0196f, 0.8741f},
                                     {0.4595f, 0.7844f, 0.3580f, 0.6469f, 0.7782f},
                                     {0.0130f, 0.8869f, 0.8532f, 0.2119f, 0.8120f}},

                                    {{0.5163f, 0.5590f, 0.5155f, 0.1905f, 0.4255f},
                                     {0.0823f, 0.7887f, 0.8918f, 0.9243f, 0.1068f},
                                     {0.0337f, 0.2771f, 0.9744f, 0.0459f, 0.4082f},
                                     {0.9154f, 0.2569f, 0.9235f, 0.9234f, 0.3148f}}};
            var t2 =
                new float[2, 4, 3]
                   {{{0.4941f, 0.8710f, 0.0606f},
                     {0.2831f, 0.7930f, 0.5602f},
                     {0.0024f, 0.1236f, 0.4394f},
                     {0.9086f, 0.1277f, 0.2450f}},

                    {{0.5196f, 0.1349f, 0.0282f},
                     {0.1749f, 0.6234f, 0.5502f},
                     {0.7678f, 0.0733f, 0.3396f},
                     {0.6023f, 0.6546f, 0.3439f}}};

            var t1raw = new float[3 * 4 * 5];
            var t2raw = new float[2 * 4 * 3];
            { for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 5; k++) { t1raw[i * 4 * 5 + j * 5 + k] = t1[i, j, k]; } }
            { for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 3; k++) { t2raw[i * 4 * 3 + j * 3 + k] = t2[i, j, k]; } }
            var t1t = torch.tensor(t1raw, new long[] { 3, 4, 5 }); //.cuda();
            var t2t = torch.tensor(t2raw, new long[] { 2, 4, 3 }); //.cuda();

            if (torch.cuda.is_available()) {
                t1t = t1t.cuda();
                t2t = t2t.cuda();
            }
            var t3p2d3 = torch.nn.functional.conv1d(t1t, t2t, padding: 2, dilation: 3);
            if (torch.cuda.is_available()) {
                t3p2d3 = t3p2d3.cpu();
            }

            // Check the answer
            var t3p2d3Correct =
                new float[3, 2, 3]
                    {{{ 2.1121f, 0.8484f, 2.2709f},
                      {1.6692f, 0.5406f, 1.8381f}},

                     {{2.5078f, 1.2137f, 0.9173f},
                      {2.2395f, 1.1805f, 1.1954f}},

                     {{1.5215f, 1.3946f, 2.1327f},
                      {1.0732f, 1.3014f, 2.0696f}}};
            {
                var data = t3p2d3.data<float>();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++) {
                            var itemCorrect = t3p2d3Correct[i, j, k];
                            var item = t3p2d3[i, j, k].ToDouble();
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }
        }

        [Fact]
        [TestOf(nameof(special.entr))]
        public void SpecialEntropy()
        {
            var a = torch.tensor(new float[] { -0.5f, 1.0f, 0.5f });
            var expected = torch.tensor(
                    new float[] { Single.NegativeInfinity, 0.0f, 0.3465736f });
            var b = torch.special.entr(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.airy_ai))]
        public void AiryAI()
        {
            var a = torch.tensor(new float[] { -0.5f, -0.25f, 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 0.47573f, 0.41872f, 0.13529f, 0.29116f, 0.23169f });
            var b = torch.special.airy_ai(a);
            Assert.True(b.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.bessel_j0))]
        public void BesselJ0()
        {
            var a = torch.tensor(new float[] { -0.5f, -0.25f, 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 0.93847f, 0.98444f, 0.7652f, 0.98444f, 0.93847f });
            var b = torch.special.bessel_j0(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.bessel_j0(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.bessel_j1))]
        public void BesselJ1()
        {
            var a = torch.tensor(new float[] { -0.5f, -0.25f, 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { -0.24227f, -0.12403f, 0.44005f, 0.12403f, 0.24227f });
            var b = torch.special.bessel_j1(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.bessel_j1(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.bessel_y0))]
        public void BesselY0()
        {
            var a = torch.tensor(new float[] { 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 0.0883f, -0.9316f, -0.4445f });
            var b = torch.special.bessel_y0(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.bessel_y0(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.bessel_y1))]
        public void BesselY1()
        {
            var a = torch.tensor(new float[] { 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { -0.7812f, -2.7041f, -1.4715f });
            var b = torch.special.bessel_y1(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.bessel_y1(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.modified_bessel_i0))]
        public void ModifiedBesselI0()
        {
            var a = torch.tensor(new float[] { -0.5f, -0.25f, 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 1.0635f, 1.0157f, 1.2661f, 1.0157f, 1.0635f });
            var b = torch.special.modified_bessel_i0(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.modified_bessel_i0(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.modified_bessel_i1))]
        public void ModifiedBesselI1()
        {
            var a = torch.tensor(new float[] { -0.5f, -0.25f, 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { -0.2579f, -0.1260f, 0.5652f, 0.1260f, 0.2579f });
            var b = torch.special.modified_bessel_i1(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.modified_bessel_i1(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.modified_bessel_k0))]
        public void ModifiedBesselK0()
        {
            var a = torch.tensor(new float[] { 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 0.4210f, 1.5415f, 0.9244f });
            var b = torch.special.modified_bessel_k0(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.modified_bessel_k0(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.modified_bessel_k1))]
        public void ModifiedBesselK1()
        {
            var a = torch.tensor(new float[] { 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 0.6019f, 3.7470f, 1.6564f });
            var b = torch.special.modified_bessel_k1(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.modified_bessel_k1(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.scaled_modified_bessel_k0))]
        public void ScaledModifiedBesselK0()
        {
            var a = torch.tensor(new float[] { 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 1.1445f, 1.9793f, 1.5241f });
            var b = torch.special.scaled_modified_bessel_k0(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.scaled_modified_bessel_k0(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.scaled_modified_bessel_k1))]
        public void ScaledModifiedBesselK1()
        {
            var a = torch.tensor(new float[] { 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 1.6362f, 4.8113f, 2.7310f });
            var b = torch.special.scaled_modified_bessel_k1(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.scaled_modified_bessel_k1(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.spherical_bessel_j0))]
        public void SphericalBesselJ0()
        {
            var a = torch.tensor(new float[] { -0.5f, -0.25f, 1.0f, 0.25f, 0.5f });
            var z = torch.zeros_like(a);
            var expected = torch.tensor(
                    new float[] { 0.9589f, 0.9896f, 0.8415f, 0.9896f, 0.9589f });
            var b = torch.special.spherical_bessel_j0(a);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.spherical_bessel_j0(a, z);
            Assert.True(z.allclose(expected, 0.001));
        }

        [Fact]
        [TestOf(nameof(special.chebyshev_polynomial_t))]
        public void ChebyshevPolynomial1()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 0.8474f, 0.1164f, -0.9931f });
            var b = torch.special.chebyshev_polynomial_t(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.chebyshev_polynomial_t(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.chebyshev_polynomial_u))]
        public void ChebyshevPolynomial2()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 0.7805f, -0.0622f, -1.0256f });
            var b = torch.special.chebyshev_polynomial_u(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.chebyshev_polynomial_u(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.chebyshev_polynomial_v))]
        public void ChebyshevPolynomial3()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 1.3157f, 0.9469f, -0.9038f });
            var b = torch.special.chebyshev_polynomial_v(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.chebyshev_polynomial_v(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.chebyshev_polynomial_w))]
        public void ChebyshevPolynomial4()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 0.2453f, -1.0714f, -1.1474f });
            var b = torch.special.chebyshev_polynomial_w(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.chebyshev_polynomial_w(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.shifted_chebyshev_polynomial_t))]
        public void ShiftedChebyshevPolynomial1()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { -0.9611f, 0.5865f, -0.9260f });
            var b = torch.special.shifted_chebyshev_polynomial_t(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.shifted_chebyshev_polynomial_t(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.shifted_chebyshev_polynomial_u))]
        public void ShiftedChebyshevPolynomial2()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { -0.6478f, -0.0990f, -0.7273f });
            var b = torch.special.shifted_chebyshev_polynomial_u(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.shifted_chebyshev_polynomial_u(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.shifted_chebyshev_polynomial_v))]
        public void ShiftedChebyshevPolynomial3()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { -0.2302f, -1.1600f, -0.3007f });
            var b = torch.special.shifted_chebyshev_polynomial_v(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.shifted_chebyshev_polynomial_v(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.shifted_chebyshev_polynomial_w))]
        public void ShiftedChebyshevPolynomial4()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { -1.0655f, 0.9621f, -1.1538f });
            var b = torch.special.shifted_chebyshev_polynomial_w(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.shifted_chebyshev_polynomial_w(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.hermite_polynomial_h))]
        public void HermitePolynomial1()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 2.0256e+09f, 2.6235e+09f, 3.0913e+09f });
            var b = torch.special.hermite_polynomial_h(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.hermite_polynomial_h(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.hermite_polynomial_he))]
        public void HermitePolynomial2()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 4129906.25f, 5600832f, 7537106f });
            var b = torch.special.hermite_polynomial_he(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.hermite_polynomial_he(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.laguerre_polynomial_l))]
        public void LaguerrePolynomial1()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { -0.2612f, -0.4182f, -0.4087f });
            var b = torch.special.laguerre_polynomial_l(x, n);
            Assert.True(b.allclose(expected, 0.001));
            torch.special.laguerre_polynomial_l(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.legendre_polynomial_p))]
        public void LegendrePolynomial1()
        {
            var x = torch.tensor(new float[] { 0.125f, 0.177f, 0.267f });
            var n = torch.ones(3, dtype: torch.int64) * 17;
            var z = torch.zeros_like(x);
            var expected = torch.tensor(
                    new float[] { 0.1554f, 0.0051f, -0.1942f });
            var b = torch.special.legendre_polynomial_p(x, n);
            Assert.True(b.allclose(expected, 0.01));
            torch.special.legendre_polynomial_p(x, n, z);
            Assert.True(z.allclose(expected, 0.01));
        }

        [Fact]
        [TestOf(nameof(special.erf))]
        public void SpecialErrorFunction()
        {
            var a = torch.tensor(new float[] { 0.0f, -1.0f, 10.0f });
            var expected = torch.tensor(
                    new float[] { 0.0f, -0.8427f, 1.0f });
            var b = torch.special.erf(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.erfc))]
        public void SpecialComplementaryErrorFunction()
        {
            var a = torch.tensor(new float[] { 0.0f, -1.0f, 10.0f });
            var expected = torch.tensor(
                    new float[] { 1.0f, 1.8427f, 0.0f });
            var b = torch.special.erfc(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.erfinv))]
        public void SpecialInverseErrorFunction()
        {
            var a = torch.tensor(new float[] { 0.0f, 0.5f, -1.0f });
            var expected = torch.tensor(
                    new float[] { 0.0f, 0.476936281f, Single.NegativeInfinity });
            var b = torch.special.erfinv(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.expit))]
        public void SpecialExpit()
        {
            var a = torch.randn(new long[] { 10 });
            var expected = torch.tensor(a.data<float>().ToArray().Select(x => 1.0f / (1.0f + MathF.Exp(-x))).ToArray());
            var b = torch.special.expit(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(special.expm1))]
        public void SpecialExpm1()
        {
            var a = torch.randn(new long[] { 10 });
            var expected = torch.tensor(a.data<float>().ToArray().Select(x => MathF.Exp(x) - 1.0f).ToArray());
            var b = torch.special.expm1(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(special.exp2))]
        public void SpecialExp2()
        {
            var a = torch.randn(new long[] { 10 });
            var expected = torch.tensor(a.data<float>().ToArray().Select(x => MathF.Pow(2.0f, x)).ToArray());
            var b = torch.special.exp2(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(special.gammaln))]
        public void SpecialGammaLN()
        {
            var a = torch.arange(0.5f, 2f, 0.5f);
            var expected = torch.tensor(new float[] { 0.5723649f, 0.0f, -0.120782226f });
            var b = torch.special.gammaln(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i0))]
        public void Speciali0()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 0.99999994f, 1.266066f, 2.27958512f, 4.88079262f, 11.3019209f });
            var b = torch.special.i0(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i0e))]
        public void Speciali0e()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 1.0f, 0.465759635f, 0.3085083f, 0.243000358f, 0.20700191f });
            var b = torch.special.i0e(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i1))]
        public void Speciali1()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 0.0000f, 0.5651591f, 1.59063685f, 3.95337057f, 9.759467f });
            var b = torch.special.i1(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i1e))]
        public void Speciali1e()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 0.0000f, 0.207910419f, 0.215269282f, 0.196826726f, 0.178750873f });
            var b = torch.special.i1e(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(Tensor.nan_to_num))]
        public void NanToNumTest()
        {
            {
                var a = torch.tensor(new float[] { Single.NaN, Single.PositiveInfinity, Single.NegativeInfinity, MathF.PI });

                {
                    var expected = torch.tensor(new float[] { 0.0f, Single.MaxValue, Single.MinValue, MathF.PI });
                    Assert.True(a.nan_to_num().allclose(expected));
                }
                {
                    var expected = torch.tensor(new float[] { 2.0f, Single.MaxValue, Single.MinValue, MathF.PI });
                    Assert.True(a.nan_to_num(nan: 2.0f).allclose(expected));
                }
                {
                    var expected = torch.tensor(new float[] { 2.0f, 3.0f, Single.MinValue, MathF.PI });
                    Assert.True(a.nan_to_num(nan: 2.0f, posinf: 3.0f).allclose(expected));
                }
                {
                    var expected = torch.tensor(new float[] { 2.0f, 3.0f, -13.0f, MathF.PI });
                    Assert.True(a.nan_to_num(nan: 2.0f, posinf: 3.0f, neginf: -13.0f).allclose(expected));
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.nan_to_num_))]
        public void NanToNumInPlaceTest()
        {
            {
                var a = torch.tensor(new float[] { Single.NaN, Single.PositiveInfinity, Single.NegativeInfinity, MathF.PI });
                var expected = torch.tensor(new float[] { 2.0f, 3.0f, -13.0f, MathF.PI });
                var res = a.nan_to_num_(nan: 2.0f, posinf: 3.0f, neginf: -13.0f);
                Assert.Same(a, res);
                Assert.True(res.allclose(expected));
            }
            {
                var a = torch.tensor(new float[] { Single.NaN, Single.PositiveInfinity, Single.NegativeInfinity, MathF.PI });
                var expected = torch.tensor(new float[] { 2.0f, 3.0f, -13.0f, MathF.PI });
                var res = torch.nan_to_num_(a, nan: 2.0f, posinf: 3.0f, neginf: -13.0f);
                Assert.Same(a, res);
                Assert.True(res.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.diff))]
        public void TensorDiffTest()
        {
            var a = torch.tensor(new float[] { 1, 3, 2 });
            Assert.True(a.diff().allclose(torch.tensor(new float[] { 2, -1 })));
            var b = torch.tensor(new float[] { 4, 5 });
            Assert.True(a.diff(append: b).allclose(torch.tensor(new float[] { 2, -1, 2, 1 })));
            var c = torch.tensor(new float[] { 1, 2, 3, 3, 4, 5 }, 2, 3);
            Assert.True(c.diff(dim: 0).allclose(torch.tensor(new float[] { 2, 2, 2 }, 1, 3)));
            Assert.True(c.diff(dim: 1).allclose(torch.tensor(new float[] { 1, 1, 1, 1 }, 2, 2)));
        }

        [Fact]
        [TestOf(nameof(Tensor.ravel))]
        public void RavelTest()
        {
            var expected = torch.tensor(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var a = expected.view(2, 2, 2);
            Assert.Equal(new long[] { 2, 2, 2 }, a.shape);
            Assert.Equal(expected, a.ravel());
        }

        [Fact]
        [TestOf(nameof(Tensor.stride))]
        public void StrideTest_1()
        {
            var x = torch.zeros(new long[] { 2, 2, 2 }, int32);
            Assert.Equal(4, x.stride(0));
            Assert.Equal(2, x.stride(1));
            Assert.Equal(1, x.stride(2));

            Assert.Equal(new long[] { 4, 2, 1 }, x.stride());
        }

        [Fact]
        [TestOf(nameof(Tensor.stride))]
        public void StrideTest_2()
        {
            var x = torch.zeros(new long[] { 2, 2, 2 }, int32);
            Assert.Equal(new long[] { 4, 2, 1 }, x.stride());
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void Complex32PartsTest()
        {
            var x = torch.zeros(new long[] { 20 }, complex64);
            var r1 = x.real;
            var i1 = x.imag;

            Assert.Equal(x.shape, r1.shape);
            Assert.Equal(x.shape, i1.shape);
            Assert.Equal(ScalarType.Float32, r1.dtype);
            Assert.Equal(ScalarType.Float32, i1.dtype);

            var vasr = x.view_as_real();

            Assert.Equal(new long[] { 20, 2 }, vasr.shape);
            Assert.Equal(ScalarType.Float32, vasr.dtype);

            var r2 = vasr[TensorIndex.Ellipsis, TensorIndex.Single(0)];
            var i2 = vasr[TensorIndex.Ellipsis, TensorIndex.Single(1)];

            Assert.Equal(x.shape, r2.shape);
            Assert.Equal(x.shape, i2.shape);
            Assert.Equal(ScalarType.Float32, r2.dtype);
            Assert.Equal(ScalarType.Float32, i2.dtype);

            Assert.Equal(r1, r2);
            Assert.Equal(i1, i2);
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void Complex64PartsTest()
        {
            var x = torch.zeros(new long[] { 20 }, complex128);
            var r1 = x.real;
            var i1 = x.imag;

            Assert.Equal(x.shape, r1.shape);
            Assert.Equal(x.shape, i1.shape);
            Assert.Equal(ScalarType.Float64, r1.dtype);
            Assert.Equal(ScalarType.Float64, i1.dtype);

            var vasr = x.view_as_real();

            Assert.Equal(new long[] { 20, 2 }, vasr.shape);
            Assert.Equal(ScalarType.Float64, vasr.dtype);

            var r2 = vasr[TensorIndex.Ellipsis, TensorIndex.Single(0)];
            var i2 = vasr[TensorIndex.Ellipsis, TensorIndex.Single(1)];

            Assert.Equal(x.shape, r2.shape);
            Assert.Equal(x.shape, i2.shape);
            Assert.Equal(ScalarType.Float64, r2.dtype);
            Assert.Equal(ScalarType.Float64, i2.dtype);

            Assert.Equal(r1, r2);
            Assert.Equal(i1, i2);
        }

        [Fact]
        [TestOf(nameof(Tensor.view_as_real))]
        [TestOf(nameof(Tensor.view_as_complex))]
        public void Complex32TensorView()
        {
            var x = torch.zeros(new long[] { 20, 20, 20 }, complex64);
            var y = torch.zeros(new long[] { 20, 20, 20, 2 });

            var vasr = x.view_as_real();
            Assert.Equal(new long[] { 20, 20, 20, 2 }, vasr.shape);
            Assert.Equal(ScalarType.Float32, vasr.dtype);

            var vasc = y.view_as_complex();
            Assert.Equal(ScalarType.ComplexFloat32, vasc.dtype);
            Assert.Equal(x.shape, vasc.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.view_as_real))]
        [TestOf(nameof(Tensor.view_as_complex))]
        public void Complex64TensorView()
        {
            var x = torch.zeros(new long[] { 20, 20, 20 }, complex128);
            var y = torch.zeros(new long[] { 20, 20, 20, 2 }, float64);

            var vasr = x.view_as_real();
            Assert.Equal(new long[] { 20, 20, 20, 2 }, vasr.shape);
            Assert.Equal(ScalarType.Float64, vasr.dtype);

            var vasc = y.view_as_complex();
            Assert.Equal(ScalarType.ComplexFloat64, vasc.dtype);
            Assert.Equal(x.shape, vasc.shape);

        }

        [Fact]
        [TestOf(nameof(fft.fft_))]
        public void Float32FFT()
        {
            var input = torch.arange(4);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fft_))]
        public void Float64FFT()
        {
            var input = torch.arange(4, float64);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft))]
        public void Float32HFFT()
        {
            var input = torch.arange(4);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float32, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft))]
        public void Float64HFFT()
        {
            var input = torch.arange(4, float64);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float64, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.rfft))]
        public void Float32RFFT()
        {
            var input = torch.arange(4);
            var output = fft.rfft(input);
            Assert.Equal(3, output.shape[0]);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.irfft(output);
            Assert.Equal(ScalarType.Float32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.rfft))]
        public void Float64RFFT()
        {
            var input = torch.arange(4, float64);
            var output = fft.rfft(input);
            Assert.Equal(3, output.shape[0]);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.irfft(output);
            Assert.Equal(ScalarType.Float64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fft_))]
        public void ComplexFloat32FFT()
        {
            var input = torch.arange(4, complex64);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }


        [Fact]
        [TestOf(nameof(fft.fft_))]
        public void ComplexFloat64FFT()
        {
            var input = torch.arange(4, complex128);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft))]
        public void ComplexFloat32HFFT()
        {
            var input = torch.arange(4, complex64);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float32, output.dtype);

            var inverted = fft.ihfft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft))]
        public void ComplexFloat64HFFT()
        {
            var input = torch.arange(4, complex128);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float64, output.dtype);

            var inverted = fft.ihfft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fft2))]
        public void Float32FFT2()
        {
            var input = torch.rand(new long[] { 5, 5 });
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fft2))]
        public void Float64FFT2()
        {
            var input = torch.rand(new long[] { 5, 5 }, float64);
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.rfft2))]
        public void Float32RFFT2()
        {
            var input = torch.rand(new long[] { 5, 5 });
            var output = fft.rfft2(input);
            Assert.Equal(new long[] { 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.irfft2(output);
            Assert.Equal(ScalarType.Float32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.rfft2))]
        public void Float64RFFT2()
        {
            var input = torch.rand(new long[] { 5, 5 }, float64);
            var output = fft.rfft2(input);
            Assert.Equal(new long[] { 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.irfft2(output);
            Assert.Equal(ScalarType.Float64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fft2))]
        public void ComplexFloat32FFT2()
        {
            var input = torch.rand(new long[] { 5, 5 }, complex64);
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fft2))]
        public void ComplexFloat64FFT2()
        {
            var input = torch.rand(new long[] { 5, 5 }, complex128);
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fftn))]
        public void Float32FFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fftn))]
        public void Float64FFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 }, float64);
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fftn))]
        public void ComplexFloat32FFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 }, complex64);
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fftn))]
        public void ComplexFloat64FFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 }, complex128);
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.fftn))]
        public void Float32RFFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.rfftn(input);
            Assert.Equal(new long[] { 5, 5, 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.irfftn(output);
            Assert.Equal(ScalarType.Float32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.rfftn))]
        public void Float64RFFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 }, float64);
            var output = fft.rfftn(input);
            Assert.Equal(new long[] { 5, 5, 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.irfftn(output);
            Assert.Equal(ScalarType.Float64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft2))]
        public void Float32HFFT2()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.hfft2(input);
            Assert.Equal(new long[] { 5, 5, 5, 8 }, output.shape);
            Assert.Equal(input.dtype, output.dtype);

            var inverted = fft.ihfft2(output);
            Assert.Equal(new long[] { 5, 5, 5, 5 }, inverted.shape);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft2))]
        public void Float64HFFT2()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 }, float64);
            var output = fft.hfft2(input);
            Assert.Equal(new long[] { 5, 5, 5, 8 }, output.shape);
            Assert.Equal(input.dtype, output.dtype);

            var inverted = fft.ihfft2(output);
            Assert.Equal(new long[] { 5, 5, 5, 5 }, inverted.shape);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        [TestOf(nameof(fft.hfft2))]
        public void Float32HFFTN()
        {
            var input = torch.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.hfft2(input);
            Assert.Equal(new long[] { 5, 5, 5, 8 }, output.shape);
            Assert.Equal(input.dtype, output.dtype);

            var inverted = fft.ihfft2(output);
            Assert.Equal(new long[] { 5, 5, 5, 5 }, inverted.shape);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact(Skip = "Fails on all Release builds.")]
        [TestOf(nameof(fft.hfftn))]
        public void Float64HFFTN()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && !RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {

                // TODO: Something in this test makes if fail on Windows / Release and MacOS / Release

                var input = torch.rand(new long[] { 5, 5, 5, 5 }, float64);
                var output = fft.hfftn(input);
                Assert.Equal(new long[] { 5, 5, 5, 8 }, output.shape);
                Assert.Equal(input.dtype, output.dtype);

                var inverted = fft.ihfftn(output);
                Assert.Equal(new long[] { 5, 5, 5, 5 }, inverted.shape);
                Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
            }
        }

        [Fact]
        [TestOf(nameof(fft.fftfreq))]
        public void Float32FFTFrequency()
        {
            var x = torch.fft.fftfreq(5);
            Assert.Equal(ScalarType.Float32, x.dtype);
            Assert.Equal(1, x.dim());
            Assert.Equal(5, x.shape[0]);
        }

        [Fact]
        [TestOf(nameof(fft.fftfreq))]
        public void Float64FFTFrequency()
        {
            var x = torch.fft.fftfreq(5, dtype: float64);
            Assert.Equal(ScalarType.Float64, x.dtype);
            Assert.Equal(1, x.dim());
            Assert.Equal(5, x.shape[0]);
        }

        [Fact]
        [TestOf(nameof(fft.fftshift))]
        public void Float32FFTShift()
        {
            var x = torch.fft.fftfreq(4, dtype: float32);
            var shifted = fft.fftshift(x);
            Assert.Equal(ScalarType.Float32, shifted.dtype);
            Assert.Equal(1, shifted.dim());
            Assert.Equal(4, shifted.shape[0]);
            var y = fft.ifftshift(x);
            Assert.Equal(ScalarType.Float32, y.dtype);
            Assert.Equal(1, y.dim());
            Assert.Equal(4, y.shape[0]);
        }

        [Fact]
        [TestOf(nameof(fft.fftshift))]
        public void Float64FFTShift()
        {
            var x = torch.fft.fftfreq(4, dtype: float64);
            var shifted = fft.fftshift(x);
            Assert.Equal(ScalarType.Float64, shifted.dtype);
            Assert.Equal(1, shifted.dim());
            Assert.Equal(4, shifted.shape[0]);
            var y = fft.ifftshift(x);
            Assert.Equal(ScalarType.Float64, y.dtype);
            Assert.Equal(1, y.dim());
            Assert.Equal(4, y.shape[0]);
        }

        [Fact]
        [TestOf(nameof(VisionExtensionMethods.crop))]
        public void CropTensor()
        {
            {
                var input = torch.rand(25, 25);
                var cropped = input.crop(5, 5, 15, 13);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 13; i++) {
                    // Test the diagonal only.
                    Assert.Equal(input[5 + i, 5 + i], cropped[i, i]);
                }
            }
            {
                var input = torch.rand(3, 25, 25);
                var cropped = input.crop(5, 5, 15, 13);

                Assert.Equal(new long[] { 3, 15, 13 }, cropped.shape);
                for (int c = 0; c < 3; c++)
                    for (int i = 0; i < 13; i++) {
                        // Test the diagonal only.
                        Assert.Equal(input[c, 5 + i, 5 + i], cropped[c, i, i]);
                    }
            }
            {
                var input = torch.rand(16, 3, 25, 25);
                var cropped = input.crop(5, 5, 15, 13);

                Assert.Equal(new long[] { 16, 3, 15, 13 }, cropped.shape);
                for (int n = 0; n < 16; n++)
                    for (int c = 0; c < 3; c++)
                        for (int i = 0; i < 13; i++) {
                            // Test the diagonal only.
                            Assert.Equal(input[n, c, 5 + i, 5 + i], cropped[n, c, i, i]);
                        }
            }
        }

        [Fact]
        [TestOf(nameof(VisionExtensionMethods.crop))]
        public void CroppedTensorWithPadding()
        {
            {
                var input = torch.rand(25, 25);
                var cropped = input.crop(-1, -1, 15, 13);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 12; i++) {
                    // Test the diagonal only, and skip the padded corner.
                    Assert.Equal(input[i, i], cropped[i + 1, i + 1]);
                }
            }
            {
                var input = torch.rand(25, 25);
                var cropped = input.crop(11, 13, 15, 13);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 12; i++) {
                    // Test the diagonal only, and skip the padded corner.
                    Assert.Equal(input[11 + i, 13 + i], cropped[i, i]);
                }
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.RandomResizedCrop))]
        public void RandomResizeCropTensor()
        {
            {
                var input = torch.rand(4, 3, 100, 100);
                var cropped = torchvision.transforms.RandomResizedCrop(100, 0.1, 0.5).call(input);

                Assert.Equal(new long[] { 4, 3, 100, 100 }, cropped.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.CenterCrop))]
        public void CenterCropTensor()
        {
            {
                var input = torch.rand(25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).call(input);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 13; i++) {
                    // Test the diagonal only.
                    Assert.Equal(input[5 + i, 6 + i], cropped[i, i]);
                }
            }
            {
                var input = torch.rand(3, 25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).call(input);

                Assert.Equal(new long[] { 3, 15, 13 }, cropped.shape);
                for (int c = 0; c < 3; c++)
                    for (int i = 0; i < 13; i++) {
                        // Test the diagonal only.
                        Assert.Equal(input[c, 5 + i, 6 + i], cropped[c, i, i]);
                    }
            }
            {
                var input = torch.rand(16, 3, 25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).call(input);

                Assert.Equal(new long[] { 16, 3, 15, 13 }, cropped.shape);
                for (int n = 0; n < 16; n++)
                    for (int c = 0; c < 3; c++)
                        for (int i = 0; i < 13; i++) {
                            // Test the diagonal only.
                            Assert.Equal(input[n, c, 5 + i, 6 + i], cropped[n, c, i, i]);
                        }
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Resize))]
        public void ResizeTensorDown()
        {
            {
                var input = torch.rand(16, 3, 25, 25);
                var resized = torchvision.transforms.Resize(15).call(input);

                Assert.Equal(new long[] { 16, 3, 15, 15 }, resized.shape);
            }
            {
                var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, int32);
                var resized = torchvision.transforms.Resize(15).call(input);

                Assert.Equal(new long[] { 16, 3, 15, 15 }, resized.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Resize))]
        public void ResizeTensorUp()
        {
            {
                var input = torch.rand(16, 3, 25, 25);
                var resized = torchvision.transforms.Resize(50).call(input);

                Assert.Equal(new long[] { 16, 3, 50, 50 }, resized.shape);
            }
            {
                var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, int32);
                var resized = torchvision.transforms.Resize(50).call(input);

                Assert.Equal(new long[] { 16, 3, 50, 50 }, resized.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Grayscale))]
        public void GrayscaleTensor()
        {
            {
                var input = torch.rand(16, 3, 25, 25);
                var gray = torchvision.transforms.Grayscale().call(input);

                Assert.Equal(new long[] { 16, 1, 25, 25 }, gray.shape);
            }
            {
                var input = torch.rand(16, 3, 25, 25);
                var gray = torchvision.transforms.Grayscale(3).call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, gray.shape);
                for (int n = 0; n < 16; n++)
                    for (int i = 0; i < 15; i++) {
                        // Test the diagonal only.
                        Assert.Equal(gray[n, 0, i, i], gray[n, 1, i, i]);
                        Assert.Equal(gray[n, 0, i, i], gray[n, 2, i, i]);
                    }
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Invert))]
        public void InvertTensor()
        {
            {
                using var input = torch.rand(25);
                var iData = input.data<float>();

                var poster = torchvision.transforms.Invert().call(input);
                var pData = poster.data<float>();

                Assert.Equal(new long[] { 25 }, poster.shape);
                for (int i = 0; i < poster.shape[0]; i++) {
                    Assert.Equal(1.0f - iData[i], pData[i]);
                }
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Posterize))]
        public void PosterizeTensor()
        {
            {
                using var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, torch.uint8);
                var poster = torchvision.transforms.Posterize(4).call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.randint(255, new long[] { 25 }, torch.uint8);
                var poster = torchvision.transforms.Posterize(4).call(input);

                Assert.Equal(new long[] { 25 }, poster.shape);
                Assert.All(poster.data<byte>().ToArray(), b => Assert.Equal(0, b & 0xf));
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.AdjustSharpness))]
        public void AdjustSharpnessTensor()
        {
            {
                using var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, torch.uint8);
                var poster = torchvision.transforms.AdjustSharpness(1.5).call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, torch.uint8);
                var poster = torchvision.transforms.AdjustSharpness(0.5).call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.functional.adjust_hue))]
        public void AdjustHueTensor()
        {
            {
                using var input = torch.stack(new Tensor[] { torch.zeros(1, 2, 2), torch.ones(1, 2, 2), torch.zeros(1, 2, 2) }, dim: -3);
                {
                    var poster = torchvision.transforms.functional.adjust_hue(input, 0.0);

                    Assert.Equal(new long[] { 1, 3, 2, 2 }, poster.shape);
                    Assert.True(poster.allclose(input));
                }

                {
                    var poster = torchvision.transforms.functional.adjust_hue(input, 0.15);

                    Assert.Equal(new long[] { 1, 3, 2, 2 }, poster.shape);
                    Assert.Equal(new long[] { 1, 3, 2, 2 }, poster.shape);
                    Assert.False(poster.allclose(input));
                }
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Rotate))]
        public void RotateTensor()
        {
            {
                using var input = torch.rand(16, 3, 25, 50);
                var poster = torchvision.transforms.Rotate(90).call(input);

                Assert.Equal(new long[] { 16, 3, 25, 50 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Equalize))]
        public void EqualizeTensor()
        {
            var eq = torchvision.transforms.Equalize();
            {
                using var input = torch.randint(0, 256, new long[] { 3, 25, 50 }, dtype: torch.uint8);
                var poster = eq.call(input);

                Assert.Equal(new long[] { 3, 25, 50 }, poster.shape);
            }
            {
                using var input = torch.randint(0, 256, new long[] { 16, 3, 25, 50 }, dtype: torch.uint8);
                var poster = eq.call(input);

                Assert.Equal(new long[] { 16, 3, 25, 50 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.AutoContrast))]
        public void AutocontrastTensor()
        {
            var ac = torchvision.transforms.AutoContrast();
            {
                using var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, torch.uint8);
                var poster = ac.call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.functional.perspective))]
        public void PerspectiveTest()
        {
            {
                using var input = torch.ones(1, 3, 8, 8);
                var startpoints = new List<IList<int>>();
                startpoints.Add(new int[] { 0, 0 });
                startpoints.Add(new int[] { 7, 0 });
                startpoints.Add(new int[] { 7, 7 });
                startpoints.Add(new int[] { 0, 7 });

                var endpoints = new List<IList<int>>();
                endpoints.Add(new int[] { 1, 1 });
                endpoints.Add(new int[] { 5, 2 });
                endpoints.Add(new int[] { 5, 6 });
                endpoints.Add(new int[] { 3, 7 });

                var poster = torchvision.transforms.functional.perspective(input, startpoints, endpoints);

                var pStr = poster.ToString(torch.julia);

                Assert.Equal(new long[] { 1, 3, 8, 8 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.GaussianBlur))]
        public void GaussianBlurTest()
        {
            var gb4 = torchvision.transforms.GaussianBlur(4);
            var gb5 = torchvision.transforms.GaussianBlur(5);

            {
                using var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, torch.uint8);
                var poster = gb4.call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.rand(16, 3, 25, 25);
                // Test even-number kernel size.
                var poster = gb4.call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.rand(16, 3, 25, 25);
                // Test odd-number kernel size.
                var poster = gb5.call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.rand(16, 3, 25, 25);
                var random = torchvision.transforms.Randomize(gb4, 0.5);
                var poster = random.call(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Solarize))]
        public void SolarizeTensor()
        {
            {
                using var input = torch.rand(25);
                var poster = torchvision.transforms.Solarize(0.55).call(input);

                Assert.Equal(new long[] { 25 }, poster.shape);
                Assert.All(poster.data<float>().ToArray(), f => Assert.True(f < 0.55f));
            }
        }



        [Fact]
        [TestOf(nameof(torchvision.transforms.HorizontalFlip))]
        public void HorizontalFlipTest()
        {
            var input = torch.tensor(new int[] {
                1,  2,  3,
                1,  2,  3,
                1,  2,  3,
            }).reshape(3, 3);

            var expected = torch.tensor(new int[] {
                3,  2,  1,
                3,  2,  1,
                3,  2,  1,
            }).reshape(3, 3);

            var res = torchvision.transforms.HorizontalFlip().call(input);
            Assert.Equal(res, expected);
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.VerticalFlip))]
        public void VerticalFlipTest()
        {
            var input = torch.tensor(new int[] {
                1,  1,  1,
                2,  2,  2,
                3,  3,  3,
            }).reshape(3, 3);

            var expected = torch.tensor(new int[] {
                3,  3,  3,
                2,  2,  2,
                1,  1,  1,
            }).reshape(3, 3);

            var res = torchvision.transforms.VerticalFlip().call(input);
            Assert.Equal(res, expected);
        }

        [Fact]
        [TestOf(nameof(Tensor.pin_memory))]
        public void PinnedMemory()
        {
            using var t = torch.rand(512);
            Assert.False(t.is_pinned());
            if (torch.cuda.is_available()) {
                var s = t.pin_memory();
                Assert.True(s.is_pinned());
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.reshape))]
        [TestOf(nameof(Tensor.flatten))]
        public void Reshape()
        {
            var input = torch.ones(4, 4, 4, 4);
            using (var t = input.reshape(16, 4, 4)) {
                Assert.Equal(new long[] { 16, 4, 4 }, t.shape);
            }
            using (var t = input.flatten()) {
                Assert.Equal(new long[] { 256 }, t.shape);
            }
            using (var t = input.flatten(1)) {
                Assert.Equal(new long[] { 4, 64 }, t.shape);
            }
            input = torch.ones(16, 4, 4);
            using (var t = input.unflatten(0, 4, 4)) {
                Assert.Equal(new long[] { 4, 4, 4, 4 }, t.shape);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.flatten))]
        [TestOf(nameof(Tensor.unflatten))]
        public void FlattenNamed()
        {
            var input = torch.ones(4, 4, 4, 4, names: new[] { "N", "C", "H", "W" });
            using (var t = input.flatten(new[] { "C", "H", "W" }, "CHW")) {
                Assert.Equal(new long[] { 4, 64 }, t.shape);
                Assert.Equal(new[] { "N", "CHW" }, t.names);
            }
            input = torch.ones(16, 4, 4, names: new[] { "C", "H", "W" });
            using (var t = input.unflatten("C", ("C1", 4), ("C2", 4))) {
                Assert.Equal(new long[] { 4, 4, 4, 4 }, t.shape);
                Assert.Equal(new[] { "C1", "C2", "H", "W" }, t.names);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.refine_names))]
        public void RefineNames()
        {
            var imgs = torch.randn(32, 3, 128, 128);
            using (var named_imgs = imgs.refine_names("N", "C", "H", "W")) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", "C", "H", "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.refine_names("N", "...", "W")) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", null, null, "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.refine_names("...", "H", "W")) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { null, null, "H", "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.refine_names("N", "C", "...")) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", "C", null, null }, named_imgs.names);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.rename))]
        public void RenameTensor()
        {
            var imgs = torch.randn(32, 3, 128, 128);
            using (var named_imgs = imgs.rename(new string?[] { "N", null, null, "W" })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", null, null, "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.rename(new string?[] { "N", "...", "W" })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", null, null, "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.rename(new string?[] { "N", "C", "..." })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", "C", null, null }, named_imgs.names);
            }
            using (var named_imgs = imgs.rename(new string?[] { "...", "H", "W" })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { null, null, "H", "W" }, named_imgs.names);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.rename))]
        public void RenameNamedTensor()
        {
            var imgs = torch.randn(32, 3, 128, 128, names: new[] { "A", "B", "C", "D" });
            using (var named_imgs = imgs.rename(new string?[] { "N", null, null, "W" })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", null, null, "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.rename(new string?[] { "N", "...", "W" })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", "B", "C", "W" }, named_imgs.names);
            }
            using (var named_imgs = imgs.rename(new string?[] { "N", "H", "..." })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "N", "H", "C", "D" }, named_imgs.names);
            }
            using (var named_imgs = imgs.rename(new string?[] { "...", "H", "W" })) {
                Assert.Equal(new long[] { 32, 3, 128, 128 }, named_imgs.shape);
                Assert.Equal(new[] { "A", "B", "H", "W" }, named_imgs.names);
            }
        }


        [Fact]
        [TestOf(nameof(Tensor.align_to))]
        public void AlignTo()
        {
            var input = torch.ones(8, 16, 32, names: new[] { "A", "B", "C" });
            using (var t = input.align_to(new[] { "D", "C", "A", "B" })) {
                Assert.Equal(new long[] { 1, 32, 8, 16 }, t.shape);
                Assert.Equal(new[] { "D", "C", "A", "B" }, t.names);
            }
            using (var t = input.align_to(new[] { "D", "C", "..." })) {
                Assert.Equal(new long[] { 1, 32, 8, 16 }, t.shape);
                Assert.Equal(new[] { "D", "C", "A", "B" }, t.names);
            }
            using (var t = input.align_to(new[] { "D", "...", "B" })) {
                Assert.Equal(new long[] { 1, 8, 32, 16 }, t.shape);
                Assert.Equal(new[] { "D", "A", "C", "B" }, t.names);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.align_as))]
        public void AlignAs()
        {
            using var input = torch.ones(8, 16, 32, names: new[] { "A", "B", "C" });
            using var t = input.align_as(torch.rand(16, 32, 8, 16, names: new[] { "D", "C", "A", "B" }));
            Assert.Equal(new long[] { 1, 32, 8, 16 }, t.shape);
            Assert.Equal(new[] { "D", "C", "A", "B" }, t.names);
        }

        [Fact]
        [TestOf(nameof(Tensor.unique))]
        public void Unique()
        {
            var input = torch.tensor(new long[] { 1, 1, 2, 2, 3, 1, 1, 2 });

            {
                var (output, i, c) = input.unique();
                Assert.NotNull(output);
                Assert.Null(i);
                Assert.Null(c);
            }
            {
                var (output, i, c) = input.unique(return_inverse: true);
                Assert.NotNull(output);
                Assert.NotNull(i);
                if (i is not null)
                    Assert.Equal(input.shape, i?.shape);
                Assert.Null(c);
            }
            {
                var (output, i, c) = input.unique(return_inverse: false, return_counts: true);
                Assert.NotNull(output);
                Assert.Null(i);
                Assert.NotNull(c);
            }
            {
                var (output, i, c) = input.unique(return_inverse: true, return_counts: true);
                Assert.NotNull(output);
                Assert.NotNull(i);
                if (i is not null)
                    Assert.Equal(input.shape, i?.shape);
                Assert.NotNull(c);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.unique_consecutive))]
        public void UniqueConsequtive()
        {
            var input = torch.tensor(new long[] { 1, 1, 2, 2, 3, 1, 1, 2 });

            {
                var (output, i, c) = input.unique_consecutive();
                Assert.NotNull(output);
                Assert.Null(i);
                Assert.Null(c);
            }
            {
                var (output, i, c) = input.unique_consecutive(return_inverse: true);
                Assert.NotNull(output);
                Assert.NotNull(i);
                if (i is not null)
                    Assert.Equal(input.shape, i?.shape);
                Assert.Null(c);
            }
            {
                var (output, i, c) = input.unique_consecutive(return_inverse: false, return_counts: true);
                Assert.NotNull(output);
                Assert.Null(i);
                Assert.NotNull(c);
            }
            {
                var (output, i, c) = input.unique_consecutive(return_inverse: true, return_counts: true);
                Assert.NotNull(output);
                Assert.NotNull(i);
                if (i is not null)
                    Assert.Equal(input.shape, i?.shape);
                Assert.NotNull(c);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Basic()
        {
            var x = torch.tensor(new long[] { 1, 1, 2, 2, 3, 1, 1, 2 });

            var st = x.storage<long>();

            Assert.NotNull(st);
            Assert.Equal(8, st.Count);
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_ToArray()
        {
            var data = new long[] { 1, 1, 2, 2, 3, 1, 1, 2 };
            var x = torch.tensor(data);

            var st = x.storage<long>();

            Assert.NotNull(st);

            Assert.Equal(8, st.Count);
            Assert.Equal(data, st.ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Modify1()
        {
            var data = new long[] { 1, 1, 2, 2, 3, 1, 1, 2 };
            var x = torch.tensor(data);

            var st = x.storage<long>();

            Assert.NotNull(st);

            Assert.IsType<long>(st[3]);

            st[3] = 5;
            Assert.Equal(new long[] { 1, 1, 2, 5, 3, 1, 1, 2 }, x.data<long>().ToArray());
            Assert.Equal(new long[] { 1, 1, 2, 5, 3, 1, 1, 2 }, st.ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Modify2()
        {
            var data = new long[] { 1, 1, 2, 2, 3, 1, 1, 2 };
            var x = torch.tensor(data);

            var st = x.storage<long>();

            Assert.NotNull(st);
            Assert.Equal(2, st[(3, 5)].Count);

            st[(3, 5)] = 5;
            Assert.Equal(new long[] { 1, 1, 2, 5, 5, 1, 1, 2 }, x.data<long>().ToArray());
            Assert.Equal(new long[] { 1, 1, 2, 5, 5, 1, 1, 2 }, st.ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Modify3()
        {
            var data = new long[] { 1, 1, 2, 2, 3, 1, 1, 2 };
            var x = torch.tensor(data);

            var st = x.storage<int>();

            Assert.NotNull(st);

            Assert.IsType<int>(st[3]);

            st[3] = 5;
            Assert.Equal(new int[] { 1, 1, 2, 5, 3, 1, 1, 2 }, st.ToArray());
            Assert.Equal(data, x.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Modify4()
        {
            var data = new long[] { 1, 1, 2, 2, 3, 1, 1, 2 };
            var x = torch.tensor(data);

            var st = x.storage<int>();

            Assert.NotNull(st);
            Assert.Equal(2, st[(3, 5)].Count);

            st[(3, 5)] = 17;

            Assert.Equal(data, x.data<long>().ToArray());
            Assert.Equal(new int[] { 1, 1, 2, 17, 17, 1, 1, 2 }, st.ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Fill()
        {
            var x = torch.tensor(new long[] { 1, 1, 2, 2, 3, 1, 1, 2 });

            var st = x.storage<long>();
            Assert.NotNull(st);
            Assert.Equal(8, st.Count);

            st.fill_(17);

            Assert.Equal(new long[] { 17, 17, 17, 17, 17, 17, 17, 17 }, st.ToArray());
            Assert.Equal(new long[] { 17, 17, 17, 17, 17, 17, 17, 17 }, x.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void Storage_Copy()
        {
            var x = torch.tensor(new long[] { 1, 1, 2, 2, 3, 1, 1, 2 });

            var st = x.storage<long>();
            Assert.NotNull(st);
            Assert.Equal(8, st.Count);

            var data = new long[] { 12, 4711, 26, 23, 35, 17, 13, 27 };

            st.copy_(data);

            Assert.Equal(data, st.ToArray());
            Assert.Equal(data, x.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(torch.stft))]
        public void Float32STFT()
        {
            long n_fft = 512;
            long hop_length = 160;
            long win_length = 400;
            long signal_length = 16000;
            Tensor window = torch.hann_window(win_length);
            var time = torch.linspace(0.0, 1.0, signal_length, dtype: ScalarType.Float32);
            var input = torch.sin(2 * Math.PI * 440 * time); // 440Hz
            var output = torch.stft(input, n_fft, hop_length: hop_length, win_length: win_length, window: window, return_complex: true);
            Assert.Equal(new long[] { n_fft / 2 + 1, input.shape[0] / hop_length + 1 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = torch.istft(output, n_fft, hop_length: hop_length, win_length: win_length, window: window);
            Assert.Equal(ScalarType.Float32, inverted.dtype);

            var mse = torch.mean(torch.square(input - inverted)).item<float>();
            Assert.True(mse < 1e-10);
        }

        [Fact]
        [TestOf(nameof(torch.Tensor.permute))]
        public void Permute()
        {
            using var t = torch.ones(32, 16, 8, 4);
            Assert.Multiple(
            () => Assert.Equal(new long[] { 32, 8, 16, 4 }, t.permute(0, 2, 1, 3).shape),
            () => Assert.Equal(new long[] { 32, 8, 4, 16 }, t.permute(0, 2, 3, 1).shape)
            );
        }

        [Fact]
        [TestOf(nameof(torch.Tensor.T))]
        public void Tensor_T()
        {
            using var t = torch.ones(32, 16, 8);
            Assert.Multiple(
            () => Assert.Equal(new long[] { 8, 16, 32 }, t.T.shape)
            );
        }

        [Fact]
        [TestOf(nameof(torch.Tensor.mT))]
        public void Tensor_mT()
        {
            using var t = torch.ones(32, 16, 8);
            Assert.Multiple(
            () => Assert.Equal(new long[] { 32, 8, 16 }, t.mT.shape)
            );
        }

        [Fact]
        [TestOf(nameof(TorchSharp.Utils.TensorAccessor<float>.ToNDArray))]
        public void ToNDArray()
        {
            {
                var t = torch.ones(10);
                var a = t[0].data<float>().ToNDArray() as float[];

                Assert.NotNull(a);
                Assert.Equal(1, a.Rank);
                Assert.Single(a);
            }
            {
                var t = torch.arange(10);
                var a = t.data<long>().ToNDArray() as long[];
                var expected = new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

                Assert.NotNull(a);
                Assert.Equal(1, a.Rank);
                Assert.Equal(10, a.Length);
                Assert.Equal(expected, a);
            }
            {
                var t = torch.arange(10).reshape(2, 5);
                var a = t.data<long>().ToNDArray() as long[,];
                var expected = new long[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } };

                Assert.NotNull(a);
                Assert.Equal(2, a.Rank);
                Assert.Equal(2, a.GetLength(0));
                Assert.Equal(5, a.GetLength(1));
                Assert.Equal(expected, a);
            }
            {
                var t = torch.arange(12).reshape(2, 2, 3);
                var a = t.data<long>().ToNDArray() as long[,,];
                var expected = new long[,,] { { { 0, 1, 2 }, { 3, 4, 5 } }, { { 6, 7, 8 }, { 9, 10, 11 } } };

                Assert.NotNull(a);
                Assert.Equal(3, a.Rank);
                Assert.Equal(2, a.GetLength(0));
                Assert.Equal(2, a.GetLength(1));
                Assert.Equal(3, a.GetLength(2));
                Assert.Equal(expected, a);
            }
            {
                var t = torch.arange(16).reshape(2, 2, 2, 2);
                var a = t.data<long>().ToNDArray() as long[,,,];
                var expected = new long[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } }, { { { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 } } } };

                Assert.NotNull(a);
                Assert.Equal(4, a.Rank);
                Assert.Equal(2, a.GetLength(0));
                Assert.Equal(2, a.GetLength(1));
                Assert.Equal(2, a.GetLength(2));
                Assert.Equal(2, a.GetLength(3));
                Assert.Equal(expected, a);
            }
            {
                var t = torch.arange(16).reshape(1, 2, 2, 2, 2);
                var a = t.data<long>().ToNDArray() as long[,,,,];
                var expected = new long[,,,,] { { { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } }, { { { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 } } } } };

                Assert.NotNull(a);
                Assert.Equal(5, a.Rank);
                Assert.Equal(1, a.GetLength(0));
                Assert.Equal(2, a.GetLength(1));
                Assert.Equal(2, a.GetLength(2));
                Assert.Equal(2, a.GetLength(3));
                Assert.Equal(2, a.GetLength(4));
                Assert.Equal(expected, a);
            }
            {
                var t = torch.arange(16).reshape(1, 1, 2, 2, 2, 2);
                var a = t.data<long>().ToNDArray() as long[,,,,,];
                var expected = new long[,,,,,] { { { { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } }, { { { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 } } } } } };

                Assert.NotNull(a);
                Assert.Equal(6, a.Rank);
                Assert.Equal(1, a.GetLength(0));
                Assert.Equal(1, a.GetLength(1));
                Assert.Equal(2, a.GetLength(2));
                Assert.Equal(2, a.GetLength(3));
                Assert.Equal(2, a.GetLength(4));
                Assert.Equal(2, a.GetLength(5));
                Assert.Equal(expected, a);
            }
            {
                var t = torch.rand(new long[] { 10, 20, 30 });
                var a = t.data<float>().ToNDArray() as float[,,,];
                Assert.Null(a);
            }
            {
                var t = torch.rand(new long[] { 10, 20, 30, 2, 4, 8, 16 });
                Array a = t.data<float>().ToNDArray();
                var t1 = torch.from_array(a);

                Assert.NotNull(a);
                Assert.Equal(7, a.Rank);
                Assert.Equal(10, a.GetLength(0));
                Assert.Equal(20, a.GetLength(1));
                Assert.Equal(30, a.GetLength(2));
                Assert.Equal(2, a.GetLength(3));
                Assert.Equal(4, a.GetLength(4));
                Assert.Equal(8, a.GetLength(5));
                Assert.Equal(16, a.GetLength(6));
                Assert.True(torch.eq(t, t1).data<bool>()[0]);
            }
            {
                var t = torch.rand(10, 20, 30, 2);
                var a = t[0].data<float>().ToNDArray() as float[,,];

                Assert.NotNull(a);
                Assert.Equal(3, a.Rank);
                Assert.Equal(20, a.GetLength(0));
                Assert.Equal(30, a.GetLength(1));
                Assert.Equal(2, a.GetLength(2));
            }
        }

        [Fact]
        public void MeshGrid()
        {
            var shifts_x = torch.arange(0, 32, dtype: torch.ScalarType.Int32, device: torch.CPU);
            var shifts_y = torch.arange(0, 32, dtype: torch.ScalarType.Int32, device: torch.CPU);

            Tensor[] shifts = new Tensor[] { shifts_y, shifts_x };

            var result = torch.meshgrid(shifts, indexing: "ij");
            Assert.NotNull(result);
            Assert.Equal(shifts.Length, result.Length);
        }

        [Fact]
        public void FromFile()
        {
            var location = "tensor_åöä_ασδφεες_አስድፋስድፍ.dat";
            if (File.Exists(location)) File.Delete(location);
            var t = torch.from_file(location, true, 256 * 16);
            Assert.True(File.Exists(location));
        }

        [Fact]
        public void CartesianProd()
        {
            var a = torch.arange(1, 4);
            var b = torch.arange(4, 6);

            var expected = torch.from_array(new int[] { 1, 4, 1, 5, 2, 4, 2, 5, 3, 4, 3, 5 }).reshape(6, 2);

            var res = torch.cartesian_prod(a, b);
            Assert.Equal(expected, res);
        }

        [Fact]
        public void Combinations()
        {
            var t = torch.arange(5);
            Assert.Equal(0, torch.combinations(t, 0).numel());
            Assert.Equal(5, torch.combinations(t, 1).numel());
            Assert.Equal(20, torch.combinations(t, 2).numel());
            Assert.Equal(30, torch.combinations(t, 3).numel());
            Assert.Equal(105, torch.combinations(t, 3, true).numel());
        }

        [Fact]
        public void CDist()
        {
            var a = torch.randn(3, 2);
            var b = torch.randn(2, 2);
            var res = torch.cdist(a, b);

            Assert.Equal(3, res.shape[0]);
            Assert.Equal(2, res.shape[1]);
        }

        [Fact]
        public void Rot90()
        {
            var a = torch.arange(8).view(2, 2, 2);
            var res = a.rot90();

            var data = res.data<long>().ToArray();
            Assert.Equal(new long[] { 2, 3, 6, 7, 0, 1, 4, 5 }, data);
        }

        [Fact]
        public void Diagembed()
        {
            var a = torch.randn(2, 3);
            var res = torch.diag_embed(a);

            Assert.Equal(3, res.ndim);
            Assert.Equal(2, res.shape[0]);
            Assert.Equal(3, res.shape[1]);
            Assert.Equal(3, res.shape[1]);
        }

        [Fact]
        public void SearchSorted()
        {
            var ss = torch.from_array(new long[] { 1, 3, 5, 7, 9, 2, 4, 6, 8, 10 }).reshape(2, -1);
            var v = torch.from_array(new long[] { 3, 6, 9, 3, 6, 9 }).reshape(2, -1);
            var sorted = torch.searchsorted(ss, v);

            Assert.Equal(new long[] { 1, 3, 4, 1, 2, 4 }, sorted.data<long>().ToArray());

            sorted = torch.searchsorted(ss, v, right: true);

            Assert.Equal(new long[] { 2, 3, 5, 1, 3, 4 }, sorted.data<long>().ToArray());
        }

        [Fact]
        public void Histogram()
        {
            // https://pytorch.org/docs/stable/generated/torch.histogram.html
            (torch.Tensor hist, torch.Tensor bin_edges) = torch.histogram(torch.tensor(new double[] { 1, 2, 1 }), 4, (0, 3), torch.tensor(new double[] { 1, 2, 4 }));
            Assert.True(hist.allclose(torch.tensor(new double[] { 0, 5, 2, 0 }), 0.001));
            Assert.True(bin_edges.allclose(torch.tensor(new double[] { 0, 0.75, 1.5, 2.25, 3 }), 0.001));
            (hist, bin_edges) = torch.histogram(torch.tensor(new double[] { 1, 2, 1 }), 4, (0, 3), torch.tensor(new double[] { 1, 2, 4 }), true);
            Assert.True(hist.allclose(torch.tensor(new double[] { 0, 0.9524, 0.3810, 0 }), 0.001));
            Assert.True(bin_edges.allclose(torch.tensor(new double[] { 0, 0.75, 1.5, 2.25, 3 }), 0.001));

            // https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
            (hist, bin_edges) = torch.histogram(torch.tensor(new double[] { 1, 2, 1 }), torch.tensor(new double[] { 0, 1, 2, 3 }));
            Assert.True(hist.allclose(torch.tensor(new double[] { 0, 2, 1 }), 0.001));
            Assert.True(bin_edges.allclose(torch.tensor(new double[] { 0, 1, 2, 3 }), 0.001));
            (hist, bin_edges) = torch.histogram(torch.arange(4, dtype: ScalarType.Float64), torch.arange(5, dtype: ScalarType.Float64), density: true);
            Assert.True(hist.allclose(torch.tensor(new double[] { 0.25, 0.25, 0.25, 0.25 }), 0.001));
            Assert.True(bin_edges.allclose(torch.tensor(new double[] { 0, 1, 2, 3, 4 }), 0.001));
            (hist, bin_edges) = torch.histogram(torch.tensor(new double[,] { { 1, 2, 1 }, { 1, 0, 1 } }), torch.tensor(new double[] { 0, 1, 2, 3 }));
            Assert.True(hist.allclose(torch.tensor(new double[] { 1, 4, 1 }), 0.001));
            Assert.True(bin_edges.allclose(torch.tensor(new double[] { 0, 1, 2, 3 }), 0.001));
        }

        [Fact(Skip = "Fails intermittently")]
        public void HistogramOptimBinNums()
        {
            // https://github.com/numpy/numpy/blob/b50568d9e758b489c2a3c409ef4e57b67820f090/numpy/lib/tests/test_histograms.py#L412
            // test_empty
            foreach (HistogramBinSelector estimator in (HistogramBinSelector[])Enum.GetValues(typeof(HistogramBinSelector))) {
                (Tensor a, Tensor b) = torch.histogram(torch.empty(0, ScalarType.Int32), bins: estimator);
                Assert.True(a.allclose(torch.tensor(new int[] { 0 }), 0.001));
                Assert.True(b.allclose(torch.tensor(new double[] { 0, 1 }), 0.001));
            }

            // test_simple
            Dictionary<int, Dictionary<HistogramBinSelector, int>> basic_test = new()
            {
                { 50, new() {
                    { HistogramBinSelector.Scott, 4 }, { HistogramBinSelector.Rice, 8 }, { HistogramBinSelector.Sturges, 7 },
                    { HistogramBinSelector.Doane, 8 }, { HistogramBinSelector.Sqrt, 8 }, { HistogramBinSelector.Stone, 2 }
                } },
                { 500, new() {
                    { HistogramBinSelector.Scott, 8 }, { HistogramBinSelector.Rice, 16 }, { HistogramBinSelector.Sturges, 10 },
                    { HistogramBinSelector.Doane, 12 }, { HistogramBinSelector.Sqrt, 23 }, { HistogramBinSelector.Stone, 9 }
                } },
                { 5000, new() {
                    { HistogramBinSelector.Scott, 17 }, { HistogramBinSelector.Rice, 35 }, { HistogramBinSelector.Sturges, 14 },
                    { HistogramBinSelector.Doane, 17 }, { HistogramBinSelector.Sqrt, 71 }, { HistogramBinSelector.Stone, 20 }
                } },
            };
            foreach ((int testlen, Dictionary<HistogramBinSelector, int> expectedResults) in basic_test.Select(item => (item.Key, item.Value))) {
                Tensor x1 = linspace(-10, -1, testlen / 5 * 2);
                Tensor x2 = linspace(1, 10, testlen / 5 * 3);
                Tensor x = concatenate(new Tensor[] { x1, x2 });
                foreach ((HistogramBinSelector estimator, int numbins) in expectedResults.Select(item => (item.Key, item.Value))) {
                    (Tensor a, Tensor b) = torch.histogram(x, bins: estimator);
                    Assert.Equal(numbins, a.shape[0]);
                }
            }

            // test_small
            Dictionary<int, Dictionary<HistogramBinSelector, int>> small_dat = new()
            {
                { 1, new() {
                    { HistogramBinSelector.Scott, 1 }, { HistogramBinSelector.Rice, 1 }, { HistogramBinSelector.Sturges, 1 },
                    { HistogramBinSelector.Doane, 1 }, { HistogramBinSelector.Sqrt, 1 }, { HistogramBinSelector.Stone, 1 }
                } },
                { 2, new() {
                    { HistogramBinSelector.Scott, 1 }, { HistogramBinSelector.Rice, 3 }, { HistogramBinSelector.Sturges, 2 },
                    { HistogramBinSelector.Doane, 1 }, { HistogramBinSelector.Sqrt, 2 }, { HistogramBinSelector.Stone, 1 }
                } },
                { 3, new() {
                    { HistogramBinSelector.Scott, 2 }, { HistogramBinSelector.Rice, 3 }, { HistogramBinSelector.Sturges, 3 },
                    { HistogramBinSelector.Doane, 3 }, { HistogramBinSelector.Sqrt, 2 }, { HistogramBinSelector.Stone, 1 }
                } },
            };
            foreach ((int testlen, Dictionary<HistogramBinSelector, int> expectedResults) in small_dat.Select(item => (item.Key, item.Value))) {
                Tensor testdat = arange(testlen);
                foreach ((HistogramBinSelector estimator, int expbins) in expectedResults.Select(item => (item.Key, item.Value))) {
                    (Tensor a, Tensor b) = torch.histogram(testdat, bins: estimator);
                    Assert.Equal(expbins, a.shape[0]);
                }
            }

            // test_novariance
            Tensor novar_dataset = ones(100);
            Dictionary<HistogramBinSelector, int> novar_resultdict = new() {
                { HistogramBinSelector.Scott, 1 }, { HistogramBinSelector.Rice, 1 }, { HistogramBinSelector.Sturges, 1 },
                { HistogramBinSelector.Doane, 1 }, { HistogramBinSelector.Sqrt, 1 }, { HistogramBinSelector.Stone, 1 }
            };
            foreach ((HistogramBinSelector estimator, int numbins) in novar_resultdict.Select(item => (item.Key, item.Value))) {
                (Tensor a, Tensor b) = torch.histogram(novar_dataset, bins: estimator);
                Assert.Equal(numbins, a.shape[0]);
            }

            // test_outlier
            Tensor xcenter = linspace(-10, 10, 50);
            Tensor outlier_dataset = hstack(linspace(-110, -100, 5), xcenter);

            Dictionary<HistogramBinSelector, int> outlier_resultdict = new() { { HistogramBinSelector.Scott, 5 }, { HistogramBinSelector.Doane, 11 }, { HistogramBinSelector.Stone, 6 } };
            foreach ((HistogramBinSelector estimator, int numbins) in outlier_resultdict.Select(item => (item.Key, item.Value))) {
                (Tensor a, Tensor b) = torch.histogram(outlier_dataset, bins: estimator);
                Assert.Equal(numbins, a.shape[0]);
            }

            // test_scott_vs_stone
            Dictionary<(int, int), Tensor> x_data = new() {
                { (0, 10), tensor(new double[] { 3.528104691935328, 0.8003144167344466, 1.9574759682114784, 4.481786398402916, 3.735115980299935, -1.954555759752822, 1.9001768350511787, -0.3027144165953958, -0.2064377035871157, 0.8211970038767447 }) },
                { (0, 22), tensor(new double[] { 3.528104691935328, 0.8003144167344466, 1.9574759682114784, 4.481786398402916, 3.735115980299935, -1.954555759752822, 1.9001768350511787, -0.3027144165953958, -0.2064377035871157, 0.8211970038767447, 0.288087142321756, 2.90854701392595, 1.5220754502939868, 0.24335003298565683, 0.8877264654908513, 0.6673486547485337, 2.9881581463152123, -0.41031652753160175, 0.6261354033018027, -1.7081914786034496, -5.105979631668157, 1.3072371908807212 }) },
                { (0, 46), tensor(new double[] { 3.528104691935328, 0.8003144167344466, 1.9574759682114784, 4.481786398402916, 3.735115980299935, -1.954555759752822, 1.9001768350511787, -0.3027144165953958, -0.2064377035871157, 0.8211970038767447, 0.288087142321756, 2.90854701392595, 1.5220754502939868, 0.24335003298565683, 0.8877264654908513, 0.6673486547485337, 2.9881581463152123, -0.41031652753160175, 0.6261354033018027, -1.7081914786034496, -5.105979631668157, 1.3072371908807212, 1.7288723977190115, -1.4843300408128839, 4.539509247975215, -2.9087313491975295, 0.09151703460289214, -0.3743677000516672, 3.065558428716915, 2.93871753980057, 0.3098948513938326, 0.7563250392043471, -1.7755714952602255, -3.961592936447854, -0.6958242986523052, 0.3126979382079601, 2.4605813614554415, 2.4047596975688226, -0.7746536348159045, -0.6046055011506711, -2.0971059301341852, -2.8400358743579504, -3.4125403812500252, 3.9015507904635793, -1.019304363503307, -0.8761486032223728 }) },
                { (0, 100), tensor(new double[] { 3.528104691935328, 0.8003144167344466, 1.9574759682114784, 4.481786398402916, 3.735115980299935, -1.954555759752822, 1.9001768350511787, -0.3027144165953958, -0.2064377035871157, 0.8211970038767447, 0.288087142321756, 2.90854701392595, 1.5220754502939868, 0.24335003298565683, 0.8877264654908513, 0.6673486547485337, 2.9881581463152123, -0.41031652753160175, 0.6261354033018027, -1.7081914786034496, -5.105979631668157, 1.3072371908807212, 1.7288723977190115, -1.4843300408128839, 4.539509247975215, -2.9087313491975295, 0.09151703460289214, -0.3743677000516672, 3.065558428716915, 2.93871753980057, 0.3098948513938326, 0.7563250392043471, -1.7755714952602255, -3.961592936447854, -0.6958242986523052, 0.3126979382079601, 2.4605813614554415, 2.4047596975688226, -0.7746536348159045, -0.6046055011506711, -2.0971059301341852, -2.8400358743579504, -3.4125403812500252, 3.9015507904635793, -1.019304363503307, -0.8761486032223728, -2.5055907200998524, 1.5549807116638201, -3.227795695115903, -0.4254805604279374, -1.7909331223873513, 0.773804995718524, -1.021610275137746, -2.361264368244824, -0.056364456677309736, 0.8566637410608353, 0.13303444476633577, 0.6049437954795628, -1.2686441873619272, -0.7254823319742763, -1.344920895551902, -0.7191063230810826, -1.626292564088908, -3.4525652046633537, 0.35485228450750567, -0.8035618724165238, -3.260396693932089, 0.9255645110515484, -1.8145967287664844, 0.1038907915922779, 1.4581811243550737, 0.25796582151482134, 2.2788013690866014, -2.4696516407073053, 0.804683282355098, -1.3696201818806264, -1.7415942983637636, -1.157699329528831, -0.6231050642547453, 0.11233068445949088, -2.330299681566713, 1.8016529739083742, 0.9313248794609197, -3.0724873725544475, 2.9765043875911994, 3.7917783520611663, 2.3575591423193014, -0.35984967162470183, -2.141505243021085, 2.1089034538622733, -0.8063538939463593, 2.444890140764855, 0.4165499561537206, 1.9532780729674255, 0.7127327943488038, 1.4131463363838963, 0.021000041441640957, 3.5717409878116704, 0.25382418540723983, 0.8039787268894033 }) },
                { (1, 10), tensor(new double[] { 3.2486907273264833, -1.2235128273001508, -1.0563435045269114, -2.145937244312341, 1.730815258649357, -4.603077393760565, 3.48962352843296, -1.5224138017902056, 0.6380781921141971, -0.4987407509548202 }) },
                { (1, 22), tensor(new double[] { 3.2486907273264833, -1.2235128273001508, -1.0563435045269114, -2.145937244312341, 1.730815258649357, -4.603077393760565, 3.48962352843296, -1.5224138017902056, 0.6380781921141971, -0.4987407509548202, 2.924215874089948, -4.120281418995308, -0.644834408027015, -0.7681087093368313, 2.267538884670875, -2.1997825346280617, -0.3448564151008715, -1.7557168358427435, 0.08442749343118566, 1.1656304274316445, -2.2012383544258425, 2.2894474196792283 }) },
                { (1, 46), tensor(new double[] { 3.2486907273264833, -1.2235128273001508, -1.0563435045269114, -2.145937244312341, 1.730815258649357, -4.603077393760565, 3.48962352843296, -1.5224138017902056, 0.6380781921141971, -0.4987407509548202, 2.924215874089948, -4.120281418995308, -0.644834408027015, -0.7681087093368313, 2.267538884670875, -2.1997825346280617, -0.3448564151008715, -1.7557168358427435, 0.08442749343118566, 1.1656304274316445, -2.2012383544258425, 2.2894474196792283, 1.803181441185591, 1.0049886778037365, 1.8017118985288236, -1.3674557183486662, -0.24578045103729634, -1.8715388685181376, -0.5357761592520318, 1.060710933476372, -1.383321503450618, -0.7935070537119547, -1.3743454002391988, -1.6904112829974391, -1.342492261673638, -0.02532919783780272, -2.2346206972705556, 0.4688313956341843, 3.319604354219741, 1.4840883211546712, -0.38367110472322985, -1.7752579281696725, -1.4943165875016753, 3.384909202055493, 0.10161550955205793, -1.2739912931387067 }) },
                { (1, 100), tensor(new double[] { 3.2486907273264833, -1.2235128273001508, -1.0563435045269114, -2.145937244312341, 1.730815258649357, -4.603077393760565, 3.48962352843296, -1.5224138017902056, 0.6380781921141971, -0.4987407509548202, 2.924215874089948, -4.120281418995308, -0.644834408027015, -0.7681087093368313, 2.267538884670875, -2.1997825346280617, -0.3448564151008715, -1.7557168358427435, 0.08442749343118566, 1.1656304274316445, -2.2012383544258425, 2.2894474196792283, 1.803181441185591, 1.0049886778037365, 1.8017118985288236, -1.3674557183486662, -0.24578045103729634, -1.8715388685181376, -0.5357761592520318, 1.060710933476372, -1.383321503450618, -0.7935070537119547, -1.3743454002391988, -1.6904112829974391, -1.342492261673638, -0.02532919783780272, -2.2346206972705556, 0.4688313956341843, 3.319604354219741, 1.4840883211546712, -0.38367110472322985, -1.7752579281696725, -1.4943165875016753, 3.384909202055493, 0.10161550955205793, -1.2739912931387067, 0.38183096933493205, 4.200510272957684, 0.2403179049632583, 1.2344062194148384, 0.600340639911655, -0.7044996929870373, -2.2850363960442803, -0.698685444825755, -0.4177884667495562, 1.1732463823643953, 1.6779668277490098, 1.8622041626071146, 0.5711746505085176, 1.7702823285414562, -1.5087958819933056, 2.5057363104665757, 1.0258596408360177, -0.5961856702054313, 0.9770362930749941, -0.15114342604211145, 2.263258774902854, 3.0396336328443976, 4.371150813066323, -2.7929926709762753, -2.888227610859179, -1.0089317258929025, 0.32007413889566094, 1.7523378422324498, 0.6312698944832105, -4.044402431648006, -0.6124080252567436, 1.6559492852144924, 0.4601894707287668, 1.5240223606240495, -0.44465628522071854, -0.4015161378599949, 0.3731227819765686, 0.8201032944165126, 0.3965994402535395, 0.23801729161491764, -1.3413245725780611, 0.7551275726418388, 0.24364254198287386, 2.2589678158238393, 2.397835759803014, 0.3703128349678877, -0.7505699001802284, -1.2774608149084448, 0.8469887081282258, 0.15468013669711883, -0.6877073511421512, 0.08719371366849388, -1.2400016878962585, 1.3960640681444378 }) },
                { (2, 10), tensor(new double[] { -0.8335156948109412, -0.11253365445265895, -4.272392191336908, 3.2805416168099772, -3.5868711703897262, -1.683494731312408, 1.0057628343160856, -2.490576173214463, -2.115904437724678, -1.8180152298536987 }) },
                { (2, 22), tensor(new double[] { -0.8335156948109412, -0.11253365445265895, -4.272392191336908, 3.2805416168099772, -3.5868711703897262, -1.683494731312408, 1.0057628343160856, -2.490576173214463, -2.115904437724678, -1.8180152298536987, 1.1029080890928487, 4.584416025629915, 0.08307878599680692, -2.2358508902270335, 1.0781166411615777, -1.192319399612934, -0.03826099304230295, 2.350002439000582, -1.4957418985877249, 0.018050501946650248, -1.756215786480684, -0.3128683407692473 }) },
                { (2, 46), tensor(new double[] { -0.8335156948109412, -0.11253365445265895, -4.272392191336908, 3.2805416168099772, -3.5868711703897262, -1.683494731312408, 1.0057628343160856, -2.490576173214463, -2.115904437724678, -1.8180152298536987, 1.1029080890928487, 4.584416025629915, 0.08307878599680692, -2.2358508902270335, 1.0781166411615777, -1.192319399612934, -0.03826099304230295, 2.350002439000582, -1.4957418985877249, 0.018050501946650248, -1.756215786480684, -0.3128683407692473, 0.5131409040025913, -1.9775580975392435, -0.6776439320583927, -0.47236806170520257, -1.2753100249686025, -2.3752245727705685, -2.8424344546082536, -0.3069903913538983, -0.538113920432027, 4.462733577773209, -4.869535153042088, 0.22545300963329784, 0.7408890733264685, 2.7192677253451922, 1.0037144135626266, -1.6884274076597237, 1.9522943192186958e-05, 1.0847051442980582, -0.6270163939868177, 1.542023476138823, -3.736181309127527, 3.462369331960936, 2.9353560211468257, -0.6713546770476949 }) },
                { (2, 100), tensor(new double[] { -0.8335156948109412, -0.11253365445265895, -4.272392191336908, 3.2805416168099772, -3.5868711703897262, -1.683494731312408, 1.0057628343160856, -2.490576173214463, -2.115904437724678, -1.8180152298536987, 1.1029080890928487, 4.584416025629915, 0.08307878599680692, -2.2358508902270335, 1.0781166411615777, -1.192319399612934, -0.03826099304230295, 2.350002439000582, -1.4957418985877249, 0.018050501946650248, -1.756215786480684, -0.3128683407692473, 0.5131409040025913, -1.9775580975392435, -0.6776439320583927, -0.47236806170520257, -1.2753100249686025, -2.3752245727705685, -2.8424344546082536, -0.3069903913538983, -0.538113920432027, 4.462733577773209, -4.869535153042088, 0.22545300963329784, 0.7408890733264685, 2.7192677253451922, 1.0037144135626266, -1.6884274076597237, 1.9522943192186958e-05, 1.0847051442980582, -0.6270163939868177, 1.542023476138823, -3.736181309127527, 3.462369331960936, 2.9353560211468257, -0.6713546770476949, 1.2226815591474347, 0.09594118373637675, -1.6582705780295575, 0.17542043681666283, 2.00073177310139, -0.7621850350306999, -0.7513388461579873, -0.14894152578796196, 0.8669926601531417, 2.5567584605437363, -1.2693586103587247, 1.0167924853668635, 0.43223201252736404, -3.717224772246995, -0.8386329643053777, -0.26465779687348673, -0.0791404793872114, 0.6520068667739612, -4.08064609745743, 0.09251104628339381, -1.355351154656083, -2.8788780534772362, 1.0485928600206973, 1.4705591521304051, -1.3065005355840713, 1.6849125631426802, -0.7630329635301724, 0.13297801829228972, -2.197477893992113, 3.1689741127913527, -5.318898912766977, -0.18290524578131628, 1.3902392100939829, -4.066933092245227, -0.37893852942476164, -0.1544373307059143, 1.6494060108739632, 2.496425841194068, -0.807784538675879, -2.7690373330717297, 2.734470847854249, 2.4357712661125808, -0.9240106963067609, 0.7017769881754926, 0.7637324682533939, 1.1325508828849122, 0.40841595782822393, 2.813392483322741, -3.475919008769136, 2.0816479067814577, 0.7609439404300009, -0.43427053748975875, 2.347062996989907, -4.687206381999924 }) },
                { (3, 10), tensor(new double[] { 3.577256946860637, 0.8730197010239789, 0.19299493614401725, -3.7269854067289816, -0.5547764050287981, -0.7095179585379735, -0.16548296296491954, -1.2540013536476946, -0.08763633795185648, -0.9544360607190053 }) },
                { (3, 22), tensor(new double[] { 3.577256946860637, 0.8730197010239789, 0.19299493614401725, -3.7269854067289816, -0.5547764050287981, -0.7095179585379735, -0.16548296296491954, -1.2540013536476946, -0.08763633795185648, -0.9544360607190053, -2.6277295067253643, 1.7692447609991693, 1.7626360844150597, 3.419146127305897, 0.10006728435372042, -0.8093548292017817, -1.0907198952390609, -3.0929546311659366, 1.9647348685163202, -2.2021352602229514, -2.3700930540403458, -0.41129979884508217 }) },
                { (3, 46), tensor(new double[] { 3.577256946860637, 0.8730197010239789, 0.19299493614401725, -3.7269854067289816, -0.5547764050287981, -0.7095179585379735, -0.16548296296491954, -1.2540013536476946, -0.08763633795185648, -0.9544360607190053, -2.6277295067253643, 1.7692447609991693, 1.7626360844150597, 3.419146127305897, 0.10006728435372042, -0.8093548292017817, -1.0907198952390609, -3.0929546311659366, 1.9647348685163202, -2.2021352602229514, -2.3700930540403458, -0.41129979884508217, 2.9722967101491804, 0.47343253445382466, -2.0475702798529363, -1.4259864002240987, 1.2504899323256586, -0.3210267263738478, -1.53767270063846, -0.4600614445558781, 1.4901125328107416, 3.952221566252605, -2.488246657911874, -1.2528338223767383, -1.6075321891531529, -4.838166346357339, -1.8475840433915771, -2.0477515216856754, 2.2479559179149367, -0.2638284656018018, -3.2465708916704945, 1.2933509045403444, -0.7125415188934897, -3.4862820739069176, -1.1932992833683984, -1.1771887593764858 }) },
                { (3, 100), tensor(new double[] { 3.577256946860637, 0.8730197010239789, 0.19299493614401725, -3.7269854067289816, -0.5547764050287981, -0.7095179585379735, -0.16548296296491954, -1.2540013536476946, -0.08763633795185648, -0.9544360607190053, -2.6277295067253643, 1.7692447609991693, 1.7626360844150597, 3.419146127305897, 0.10006728435372042, -0.8093548292017817, -1.0907198952390609, -3.0929546311659366, 1.9647348685163202, -2.2021352602229514, -2.3700930540403458, -0.41129979884508217, 2.9722967101491804, 0.47343253445382466, -2.0475702798529363, -1.4259864002240987, 1.2504899323256586, -0.3210267263738478, -1.53767270063846, -0.4600614445558781, 1.4901125328107416, 3.952221566252605, -2.488246657911874, -1.2528338223767383, -1.6075321891531529, -4.838166346357339, -1.8475840433915771, -2.0477515216856754, 2.2479559179149367, -0.2638284656018018, -3.2465708916704945, 1.2933509045403444, -0.7125415188934897, -3.4862820739069176, -1.1932992833683984, -1.1771887593764858, -1.7477645955245988, 0.05942763072203324, -4.496515535153212, -0.5355237296921124, 2.026366883772989, 1.705595681908299, 2.216374999869985, 2.238781310637783, 2.9750862639850792, -2.23660136880073, 1.691666814114364, -3.7217790578842274, -1.2057702080144366, -3.8289440868116285, 2.0962950244925103, 2.6674756388849734, -0.3948293583164998, 3.5492900628118695, -1.3494550206234268, 0.30123373073004456, 0.30589140552496336, -2.1283905478730487, 0.8758932225601641, 3.8779569208009885, -2.049861749216173, 1.7986768916816285, -0.30901370379857496, 3.5392546065874857, 0.9675766952010607, 1.3524327997261243, 1.286326561191437, 0.4981734134516682, -2.79152700528135, 2.783325815489615, -2.741338025104033, 0.4771263848118024, 1.2281541755081875, -1.6758245454045881, 0.29012642820705575, 2.3357645726032836, -0.048208940142460105, -1.7773148357058983, -5.831475503585424, -1.9436810051452187, -1.1821574769717507, -1.032834735576948, -1.9199923604544331, 0.7545904682264009, -1.1494168409854102, -0.2189086677612167, 1.358143199651014, -1.7108743373197626, -0.6004121490391013, 4.316298684113837 }) },
                { (4, 10), tensor(new double[] { 0.1011234142858791, 0.999902666475658, -1.9918178622137301, 1.3871970165826233, -0.8366030400538201, -3.1691544702242482, -1.295413534243701, 1.1971503479347545, 0.6645000652199288, -2.2949532658909595 }) },
                { (4, 22), tensor(new double[] { 0.1011234142858791, 0.999902666475658, -1.9918178622137301, 1.3871970165826233, -0.8366030400538201, -3.1691544702242482, -1.295413534243701, 1.1971503479347545, 0.6645000652199288, -2.2949532658909595, 1.2373393780537627, -0.1759738566805509, 0.8501447929740429, 0.6645062907446707, -2.3136325218382856, 0.7019943061650632, -1.213774566194804, 3.0939586580314544, 1.4466832174957158, 0.09227113446032961, -1.9659833068370645, 0.10886547773031535 }) },
                { (4, 46), tensor(new double[] { 0.1011234142858791, 0.999902666475658, -1.9918178622137301, 1.3871970165826233, -0.8366030400538201, -3.1691544702242482, -1.295413534243701, 1.1971503479347545, 0.6645000652199288, -2.2949532658909595, 1.2373393780537627, -0.1759738566805509, 0.8501447929740429, 0.6645062907446707, -2.3136325218382856, 0.7019943061650632, -1.213774566194804, 3.0939586580314544, 1.4466832174957158, 0.09227113446032961, -1.9659833068370645, 0.10886547773031535, 0.31978587014525606, -2.4178963182578412, 4.446720433822337, 0.7885904294287833, 3.3847154304763074, -2.2256243077075464, 3.2714950836030576, -2.721931183672987, -1.3024516666596, 1.0849026168066598, 0.09601249438137822, -4.716147266023114, -2.211168087638802, 1.6756727078402656, 4.175741736566238, 1.8296819156013655, -0.5524067085300758, 1.5930237974498618, -2.2875971436924494, 1.019839565940434, -2.694920590182932, -0.018720201259815775, -0.26140927717632007, 1.6041732272587161 }) },
                { (4, 100), tensor(new double[] { 0.1011234142858791, 0.999902666475658, -1.9918178622137301, 1.3871970165826233, -0.8366030400538201, -3.1691544702242482, -1.295413534243701, 1.1971503479347545, 0.6645000652199288, -2.2949532658909595, 1.2373393780537627, -0.1759738566805509, 0.8501447929740429, 0.6645062907446707, -2.3136325218382856, 0.7019943061650632, -1.213774566194804, 3.0939586580314544, 1.4466832174957158, 0.09227113446032961, -1.9659833068370645, 0.10886547773031535, 0.31978587014525606, -2.4178963182578412, 4.446720433822337, 0.7885904294287833, 3.3847154304763074, -2.2256243077075464, 3.2714950836030576, -2.721931183672987, -1.3024516666596, 1.0849026168066598, 0.09601249438137822, -4.716147266023114, -2.211168087638802, 1.6756727078402656, 4.175741736566238, 1.8296819156013655, -0.5524067085300758, 1.5930237974498618, -2.2875971436924494, 1.019839565940434, -2.694920590182932, -0.018720201259815775, -0.26140927717632007, 1.6041732272587161, -0.605927934256945, 2.4040051796240745, -0.3934905569566752, 1.6730574042612305, 1.5732045655083158, -3.6817517335971086, 0.0750949730211491, 0.07185610283171551, -1.5574798491138702, 0.35882142871436706, -2.91106865439113, 1.112370445900844, 1.0195577090767334, 0.6008910858964551, 4.953168322899704, 0.7046867930961711, 0.134942001101336, -1.4645293995384698, 0.5942824205600258, -1.9235536022044917, 2.543637234651811, -1.2952890659489653, 0.3169390739431986, 3.980166032665056, 2.3283751213172925, 0.4853203171343203, 2.7598401940995814, -0.10911741090580522, 1.5904678985860643, 0.03817992381598699, -1.8108762734689252, 0.8605426625169197, 1.8693001260937747, -0.6922037444504825, -2.194243767099635, -1.0563921382954649, -4.759550547974016, -1.2153673829822194, -2.150580180607475, 4.044810133184521, -1.129750593625792, -3.0858581011853383, 1.7416835578662238, -0.3504210536975061, 0.09720601337877206, 0.3772924064703606, 0.41862697694923606, -0.7488898330125531, 1.909397194501848, 1.0464953250209086, -0.991637039872908, -0.3429292192373716, -1.8887371988206292, 0.5617293508362434 }) },
                { (5, 10), tensor(new double[] { 0.8824549737700829, -0.6617403037881753, 4.86154237401556, -0.5041842592061538, 0.21921968315636556, 3.1649622341231267, -1.8184648097124838, -1.1832733158605768, 0.37520645167407096, -0.6597399155587185 }) },
                { (5, 22), tensor(new double[] { 0.8824549737700829, -0.6617403037881753, 4.86154237401556, -0.5041842592061538, 0.21921968315636556, 3.1649622341231267, -1.8184648097124838, -1.1832733158605768, 0.37520645167407096, -0.6597399155587185, -2.385529224843612, -0.4097530211751746, -0.7176578940024863, 1.2069432052189908, -3.329577058943389, -1.4003580753799028, 2.3027820189743404, 3.7146620144626237, -3.0223591153767315, 1.2896950217855567, -1.9612157704372437, -1.7137063094321798 }) },
                { (5, 46), tensor(new double[] { 0.8824549737700829, -0.6617403037881753, 4.86154237401556, -0.5041842592061538, 0.21921968315636556, 3.1649622341231267, -1.8184648097124838, -1.1832733158605768, 0.37520645167407096, -0.6597399155587185, -2.385529224843612, -0.4097530211751746, -0.7176578940024863, 1.2069432052189908, -3.329577058943389, -1.4003580753799028, 2.3027820189743404, 3.7146620144626237, -3.0223591153767315, 1.2896950217855567, -1.9612157704372437, -1.7137063094321798, -1.743758366511307, -0.8450158583247886, 1.992879653826724, 1.4248425417531356, 0.11828848643807979, -0.7266217569638348, 0.006577685868220151, -0.21186088411484646, 1.5861066389239395, -1.263143259584431, -0.012389816971518695, -0.20213522361848935, -0.10461630170371748, 0.49843531712981515, 0.39532018208499703, 2.6696971484831638, -0.17375121255527104, 3.123064586897781, -0.6117060423332616, -0.9554628345564251, 0.2014763775057042, 0.7108769446987042, 0.539224812893402, 2.5839267667759263 }) },
                { (5, 100), tensor(new double[] { 0.8824549737700829, -0.6617403037881753, 4.86154237401556, -0.5041842592061538, 0.21921968315636556, 3.1649622341231267, -1.8184648097124838, -1.1832733158605768, 0.37520645167407096, -0.6597399155587185, -2.385529224843612, -0.4097530211751746, -0.7176578940024863, 1.2069432052189908, -3.329577058943389, -1.4003580753799028, 2.3027820189743404, 3.7146620144626237, -3.0223591153767315, 1.2896950217855567, -1.9612157704372437, -1.7137063094321798, -1.743758366511307, -0.8450158583247886, 1.992879653826724, 1.4248425417531356, 0.11828848643807979, -0.7266217569638348, 0.006577685868220151, -0.21186088411484646, 1.5861066389239395, -1.263143259584431, -0.012389816971518695, -0.20213522361848935, -0.10461630170371748, 0.49843531712981515, 0.39532018208499703, 2.6696971484831638, -0.17375121255527104, 3.123064586897781, -0.6117060423332616, -0.9554628345564251, 0.2014763775057042, 0.7108769446987042, 0.539224812893402, 2.5839267667759263, 2.2786859576505685, 0.9888807962421765, -0.6726725182731058, -0.20122869261421655, 2.8267960358435675, 0.4425082457019994, -2.6215462667594913, -1.379130464096362, -1.1550264664628163, 2.3044095407513496, -0.2143279696482956, 4.520213547440541, 1.3132389405208544, 0.24961365066038937, -0.8714078380569799, 1.9443586193583449, -0.48142228452813873, -1.6482469068342414, 1.136265435461776, 0.025516633412328314, 2.378121451649088, -0.14718663749768296, -5.719375933245112, 1.578732808359959, -3.7554817678945254, 3.0775122908180736, 3.642729477168137, -0.8540627718096371, -2.32940382023079, -2.7941480493365227, 1.7453092425036614, -0.4042363552409382, -1.1967198668442696, -0.48683940249250734, 4.17702937519543, 0.6938386541754935, 1.4914539000649105, 1.5538151822460222, 2.0368422663758956, 2.122702877374759, -1.4209328939919523, -0.430375604847281, -1.5215206171913762, -1.4223264566101406, 2.2830154748693396, -1.003511094389, -0.15830272236628568, -1.385652684011645, -1.186805542086231, 1.5764758870793583, -0.8908599829972105, -0.9642403734144381, 0.9871153219028599, 1.0009746549940688 }) },
                { (6, 10), tensor(new double[] { -0.6235673469750332, 1.4580078472250082, 0.435641576152004, -1.798183593094278, -4.97356130325573, 1.8265030424717763, 2.2541274519436647, -3.0281864572604316, 3.278582165875382, -0.859787206012696 }) },
                { (6, 22), tensor(new double[] { -0.6235673469750332, 1.4580078472250082, 0.435641576152004, -1.798183593094278, -4.97356130325573, 1.8265030424717763, 2.2541274519436647, -3.0281864572604316, 3.278582165875382, -0.859787206012696, 5.262561114527638, 1.2036444999117242, -0.67176322988449, 2.4754756864512935, 0.22225633353499297, 0.2583024958070313, 0.15225522443067466, -0.31025631711722396, 1.2684506863358491, 1.6213100068521393, 0.7096172179446679, 3.6251806276200518 }) },
                { (6, 46), tensor(new double[] { -0.6235673469750332, 1.4580078472250082, 0.435641576152004, -1.798183593094278, -4.97356130325573, 1.8265030424717763, 2.2541274519436647, -3.0281864572604316, 3.278582165875382, -0.859787206012696, 5.262561114527638, 1.2036444999117242, -0.67176322988449, 2.4754756864512935, 0.22225633353499297, 0.2583024958070313, 0.15225522443067466, -0.31025631711722396, 1.2684506863358491, 1.6213100068521393, 0.7096172179446679, 3.6251806276200518, -2.71295160855742, -0.9272639317826679, 1.6493076892881262, -2.352862960834218, 3.1289793191655493, 1.4254101890705697, -0.36201319572452273, 1.0683990511739796, -1.173225919636525, -2.9637065375770644, 1.7144952368746258, 1.8861979747464068, 0.22888286853792886, -0.043913367187134844, -4.254289093015922, -1.6688149364098868, -0.9310166198979595, 0.4674211815318785, 2.77007045192328, -1.039254173348257, -1.5603042772457907, 1.9112191811069479, -0.25347275604609376, -2.737225646646945 }) },
                { (6, 100), tensor(new double[] { -0.6235673469750332, 1.4580078472250082, 0.435641576152004, -1.798183593094278, -4.97356130325573, 1.8265030424717763, 2.2541274519436647, -3.0281864572604316, 3.278582165875382, -0.859787206012696, 5.262561114527638, 1.2036444999117242, -0.67176322988449, 2.4754756864512935, 0.22225633353499297, 0.2583024958070313, 0.15225522443067466, -0.31025631711722396, 1.2684506863358491, 1.6213100068521393, 0.7096172179446679, 3.6251806276200518, -2.71295160855742, -0.9272639317826679, 1.6493076892881262, -2.352862960834218, 3.1289793191655493, 1.4254101890705697, -0.36201319572452273, 1.0683990511739796, -1.173225919636525, -2.9637065375770644, 1.7144952368746258, 1.8861979747464068, 0.22888286853792886, -0.043913367187134844, -4.254289093015922, -1.6688149364098868, -0.9310166198979595, 0.4674211815318785, 2.77007045192328, -1.039254173348257, -1.5603042772457907, 1.9112191811069479, -0.25347275604609376, -2.737225646646945, 2.436961291872087, -1.7150028765599912, -1.122941752900342, -2.067039798791926, 0.7175419114937635, 2.1473626840043476, -0.7510094488672335, 0.7927351461205945, -0.9428925537057736, 4.673215624350502, 3.0055710622753447, -1.1909194379589794, 1.0566821130053863, 1.879649595164134, 0.8525707734545889, -1.5163140602961955, -0.3247339530007185, 1.4536199217830588, 0.8881659378192454, -1.7136452783608773, 0.8938568489791445, -2.029295973636528, -4.2646467936119405, 0.3477261359605195, 1.9024025102688775, 0.8837947742623206, 2.93803478664086, 3.499032701860824, 0.707061243317552, -1.2866745851210697, -0.09447410784476262, -2.898079572960495, -0.07237953158911983, -0.1816939716812585, 0.352580679870084, 2.1892408289776513, -4.252949179250003, 1.5028976647626628, -1.0812140876196576, 1.586444711558372, 0.3473066957388522, -2.070868571929124, 1.7485367801579967, -1.4791441542241248, 1.0458907065510672, -1.1837529622733187, -0.9549742120095328, 0.22505935287834872, 3.809484645367548, 1.3883062922610998, -0.039162676370709394, 3.325685704596122, 0.061216857661650755, -0.5949984117348242 }) },
                { (7, 10), tensor(new double[] { 3.381051407600712, -0.9318747410816656, 0.0656403273571688, 0.8150325659930157, -1.5778460572514772, 0.00413114581189626, -0.0017807717158627257, -3.5094486126908415, 2.0353160113269864, 1.2009970318390988 }) },
                { (7, 22), tensor(new double[] { 3.381051407600712, -0.9318747410816656, 0.0656403273571688, 0.8150325659930157, -1.5778460572514772, 0.00413114581189626, -0.0017807717158627257, -3.5094486126908415, 2.0353160113269864, 1.2009970318390988, -1.2508579479335193, -0.34309652239144234, 1.0105987483935033, -0.522712830383294, -0.4854981573450932, -2.9064828249815813, 1.1091606237837757, 0.24776181057407687, 0.5489198475199272, -3.0530490637396803, 3.301399382372951, 0.30867107091271606 }) },
                { (7, 46), tensor(new double[] { 3.381051407600712, -0.9318747410816656, 0.0656403273571688, 0.8150325659930157, -1.5778460572514772, 0.00413114581189626, -0.0017807717158627257, -3.5094486126908415, 2.0353160113269864, 1.2009970318390988, -1.2508579479335193, -0.34309652239144234, 1.0105987483935033, -0.522712830383294, -0.4854981573450932, -2.9064828249815813, 1.1091606237837757, 0.24776181057407687, 0.5489198475199272, -3.0530490637396803, 3.301399382372951, 0.30867107091271606, -0.7742798865727762, 4.058144441522224, -0.09077205972129218, -2.9013573982931495, -0.8104557108553734, -4.576630203943445, 2.0987930986865093, -0.8329486370400371, -1.4851070504091355, 2.144940263150991, -3.3021511788116777, 1.070858712652182, -4.128829606242351, -1.3243186793336175, -2.408439691199465, 2.923951254427048, 3.5323217558586677, -0.6588275038261302, 1.6814664842870715, -0.35997280250470065, 1.1361237746559638, -1.5056743928429295, -3.416678406334112, -3.606197316959482 }) },
                { (7, 100), tensor(new double[] { 3.381051407600712, -0.9318747410816656, 0.0656403273571688, 0.8150325659930157, -1.5778460572514772, 0.00413114581189626, -0.0017807717158627257, -3.5094486126908415, 2.0353160113269864, 1.2009970318390988, -1.2508579479335193, -0.34309652239144234, 1.0105987483935033, -0.522712830383294, -0.4854981573450932, -2.9064828249815813, 1.1091606237837757, 0.24776181057407687, 0.5489198475199272, -3.0530490637396803, 3.301399382372951, 0.30867107091271606, -0.7742798865727762, 4.058144441522224, -0.09077205972129218, -2.9013573982931495, -0.8104557108553734, -4.576630203943445, 2.0987930986865093, -0.8329486370400371, -1.4851070504091355, 2.144940263150991, -3.3021511788116777, 1.070858712652182, -4.128829606242351, -1.3243186793336175, -2.408439691199465, 2.923951254427048, 3.5323217558586677, -0.6588275038261302, 1.6814664842870715, -0.35997280250470065, 1.1361237746559638, -1.5056743928429295, -3.416678406334112, -3.606197316959482, 0.7662437035150239, 4.495190105615399, 0.538823261015774, -1.0492092388017047, 3.824037724663881, 0.4746036932743869, 0.20286797042323088, 0.5051554711106351, -0.2647543951336943, -0.6189526811777608, -2.8699269320711083, 1.0032482463620815, -0.18955090087978832, 2.3861718446880213, -0.7376369368956927, -3.8127397570609363, -0.199221263789729, 3.3990745934933617, -0.7668462458630251, -1.7797137220414987, -2.387183844516366, -2.1000336221507467, -0.6003874736003261, -2.3599641869164105, 2.995278231195167, -0.5652704721507609, 0.21729674298946036, 2.876479041467513, 3.0066370408338217, -0.4254659340351217, 0.6639484309769964, 1.4700531687547196, -0.3857109206995608, -3.5560257059170173, 1.3094114074062202, 1.7887046096410653, 0.8310052285960212, -1.8470893134264352, -0.39205462461593094, -1.1815396377262732, -0.5994224746556549, 2.593770385453346, 3.0591592667863114, 1.3388363868193223, 1.097490239556659, 1.3532579791117716, -0.024484373212886216, -0.15132692297413255, -1.3472903747521479, -0.11173490010014778, 4.519893973252523, 1.7380786585076513, -0.6842340468537261, -0.9438533042670428 }) },
                { (8, 10), tensor(new double[] { 0.18240943323963954, 2.182565466544221, -3.8939406180450926, -2.7726990644738785, -4.592983148404327, 4.8196686066830825, 3.4556723371598386, 4.40911256911227, 1.589655278696802, 1.9528421928018809 }) },
                { (8, 22), tensor(new double[] { 0.18240943323963954, 2.182565466544221, -3.8939406180450926, -2.7726990644738785, -4.592983148404327, 4.8196686066830825, 3.4556723371598386, 4.40911256911227, 1.589655278696802, 1.9528421928018809, -2.36685429466603, 3.8327272210709586, -2.246653601102277, -1.3280709391682108, -0.7567171407371178, -1.5832305429926725, 1.7190962183607463, -0.46157799097116997, -0.1313220562095843, -0.41727246946431223, 2.6937371402280994, -1.2139055947142467 }) },
                { (8, 46), tensor(new double[] { 0.18240943323963954, 2.182565466544221, -3.8939406180450926, -2.7726990644738785, -4.592983148404327, 4.8196686066830825, 3.4556723371598386, 4.40911256911227, 1.589655278696802, 1.9528421928018809, -2.36685429466603, 3.8327272210709586, -2.246653601102277, -1.3280709391682108, -0.7567171407371178, -1.5832305429926725, 1.7190962183607463, -0.46157799097116997, -0.1313220562095843, -0.41727246946431223, 2.6937371402280994, -1.2139055947142467, -0.34849642728926467, 0.8481028420634456, -3.291980209647524, -0.9670818999305805, 1.0709365000315243, 2.3322809727474336, 0.3233662174901603, -1.3456678541470983, 2.8850482633668095, -0.44296122186364845, 0.06252862737229306, 0.3638591760602077, -0.12585745973159113, -1.0831369199738545, 0.9490368357699023, -2.3291823442469544, -1.0018187811428538, -0.8171219182149926, -1.8686967143389213, 0.1690203080328945, 0.28738203601589485, 0.15109888267131694, -0.311290192833044, 1.1017001126126513 }) },
                { (8, 100), tensor(new double[] { 0.18240943323963954, 2.182565466544221, -3.8939406180450926, -2.7726990644738785, -4.592983148404327, 4.8196686066830825, 3.4556723371598386, 4.40911256911227, 1.589655278696802, 1.9528421928018809, -2.36685429466603, 3.8327272210709586, -2.246653601102277, -1.3280709391682108, -0.7567171407371178, -1.5832305429926725, 1.7190962183607463, -0.46157799097116997, -0.1313220562095843, -0.41727246946431223, 2.6937371402280994, -1.2139055947142467, -0.34849642728926467, 0.8481028420634456, -3.291980209647524, -0.9670818999305805, 1.0709365000315243, 2.3322809727474336, 0.3233662174901603, -1.3456678541470983, 2.8850482633668095, -0.44296122186364845, 0.06252862737229306, 0.3638591760602077, -0.12585745973159113, -1.0831369199738545, 0.9490368357699023, -2.3291823442469544, -1.0018187811428538, -0.8171219182149926, -1.8686967143389213, 0.1690203080328945, 0.28738203601589485, 0.15109888267131694, -0.311290192833044, 1.1017001126126513, 3.404059911341491, 2.0121874050891417, -6.2698395352412675, 2.2790670005877773, -3.7395987172040672, -3.558226072287537, 0.25552579458500824, -2.702035519605439, 3.1079343978558094, 1.3939315028551718, -1.622922305140739, -3.2865018577583096, 0.6613177364758646, 2.449912535313525, 0.19234313724599764, 3.753630840561297, 1.0675611688234792, 1.5967200820361314, -0.6121489994279035, 0.7559477745830434, -0.4529560932831035, -3.7714890853261327, 4.9450219493374155, -0.2937131743548936, 0.23129839539863772, 2.6608612446028563, 0.21918942470706543, 0.4309581439137754, 0.8581401755007989, -2.302531231806933, 3.049506953473475, 0.7210716970107113, -1.2217860449687064, 1.8604920668017024, 0.7130260304843515, -1.904321190591214, 0.3038555725097946, 0.0012697719426443212, 0.6700464552777111, 0.7495987758865342, 1.4276323153821784, -2.401626958837376, 0.0027623223110997963, -1.842920053877936, 0.9924557321884082, -2.6633333376794663, -4.38114097619445, -0.8721441295887408, 5.0901744348411295, 2.300162554696251, 0.16889093357317847, 4.972495590197573, -0.3887108052257306, -0.37242669508297144 }) },
                { (9, 10), tensor(new double[] { 0.0022171094244219465, -0.5790881385003976, -2.2321326061288285, -0.02576551349158859, -0.7567229284158362, -0.9622707257538078, -3.034662357591344, -0.9817439624434929, -0.48136115700534554, -1.2958949204432073 }) },
                { (9, 22), tensor(new double[] { 0.0022171094244219465, -0.5790881385003976, -2.2321326061288285, -0.02576551349158859, -0.7567229284158362, -0.9622707257538078, -3.034662357591344, -0.9817439624434929, -0.48136115700534554, -1.2958949204432073, 1.2717821604936634, 3.480234610570353, 0.5933644354228066, 1.4150073237037766, 3.6456315262364734, 0.8615380589977852, 3.085459250976174, -1.8014423428333775, -0.2742500207560107, 2.5951580241601, 1.3505423348643708, 0.06391623393339702 }) },
                { (9, 46), tensor(new double[] { 0.0022171094244219465, -0.5790881385003976, -2.2321326061288285, -0.02576551349158859, -0.7567229284158362, -0.9622707257538078, -3.034662357591344, -0.9817439624434929, -0.48136115700534554, -1.2958949204432073, 1.2717821604936634, 3.480234610570353, 0.5933644354228066, 1.4150073237037766, 3.6456315262364734, 0.8615380589977852, 3.085459250976174, -1.8014423428333775, -0.2742500207560107, 2.5951580241601, 1.3505423348643708, 0.06391623393339702, 1.836291792594963, 0.7610189328804824, 1.0327349744758147, -0.7104789166922073, 0.4175540024544418, 0.6568221519368889, -0.9964495329589874, -4.183553547080935, -0.16517548124306228, 4.910365304600017, -5.34422000834388, -1.8265585611734927, -0.454628714489157, 0.5386307539220379, 2.260922513553367, 2.0847948949279225, 2.607620886161084, 2.7788014647236783, -1.3129051550505986, -0.11251459175486392, -0.9998052392132262, 0.8728387619215878, -0.7516260937818804, -1.8461231657025574 }) },
                { (9, 100), tensor(new double[] { 0.0022171094244219465, -0.5790881385003976, -2.2321326061288285, -0.02576551349158859, -0.7567229284158362, -0.9622707257538078, -3.034662357591344, -0.9817439624434929, -0.48136115700534554, -1.2958949204432073, 1.2717821604936634, 3.480234610570353, 0.5933644354228066, 1.4150073237037766, 3.6456315262364734, 0.8615380589977852, 3.085459250976174, -1.8014423428333775, -0.2742500207560107, 2.5951580241601, 1.3505423348643708, 0.06391623393339702, 1.836291792594963, 0.7610189328804824, 1.0327349744758147, -0.7104789166922073, 0.4175540024544418, 0.6568221519368889, -0.9964495329589874, -4.183553547080935, -0.16517548124306228, 4.910365304600017, -5.34422000834388, -1.8265585611734927, -0.454628714489157, 0.5386307539220379, 2.260922513553367, 2.0847948949279225, 2.607620886161084, 2.7788014647236783, -1.3129051550505986, -0.11251459175486392, -0.9998052392132262, 0.8728387619215878, -0.7516260937818804, -1.8461231657025574, 3.834500484311393, -0.30060568360722734, -1.2774595117949668, 1.6495406610042813, -2.4216774879088234, -1.0068107538537676, -1.4038313865858614, -3.9485420502470707, -5.311464330059418, -0.11536449755639643, -1.312373172223766, -1.32341342550675, 1.5386968772487721, -1.7980097844695342, 3.3872758429378296, -3.3946704780163564, -5.5867404100704015, -0.45230130131331836, 0.7948576812321693, 3.3194072339261402, -0.9874936278117045, -0.7521953687090217, -0.33947953927732993, 4.834212646948517, -3.6176809803796184, 0.6795023047669002, -0.0454595208257243, -1.9199941577052373, -0.7662288414968311, 0.21905998966287715, -1.7103256762998855, 0.00044321332836016065, 1.3277101223963004, 1.4989618958224626, -0.931636744581809, -0.554878541043956, 0.07099906775513774, 1.696442329023832, 0.3259971144598705, 2.4172472150808613, 1.0050415476988732, -3.1676442186439204, 2.046066975537686, -1.3060348183684556, 1.0740902011539302, -0.015954124691934038, 1.84956833956054, 0.09321666599650066, 0.2840397364870608, -3.3073227509488703, 2.7359842077530403, 0.050212491349364174, -0.7909645013395453, 2.5687751084848207 }) }
            };

            Tensor nbins_ratio(int seed, int size)
            {
                Tensor x = x_data[(seed, size)];
                Tensor a = histogram(x, HistogramBinSelector.Stone).hist.shape[0];
                Tensor b = histogram(x, HistogramBinSelector.Scott).hist.shape[0];
                return a / (a + b);
            }

            double[] m_size = new double[] { 10, 22, 46, 100 };
            List<Tensor> lls = new();

            for (int seed = 0; seed < 10; seed++) {
                List<Tensor> temp = new();
                foreach (double size in m_size) {
                    temp.Add(nbins_ratio(seed, (int)size));
                }
                lls.Add(torch.stack(temp));
            }

            Tensor ll = torch.stack(lls);
            Tensor avg = abs(mean(ll.to_type(ScalarType.Float32), new long[] { 0 }) - 0.5);
            string g = avg.ToString(TensorStringStyle.Default);
            g.ToString();
            Assert.True(avg.allclose(torch.tensor(new float[] { 0.15f, 0.09f, 0.08f, 0.03f }), atol: 0.01));

            // test_simple_range
            basic_test = new()
            {
                { 50, new() {
                    { HistogramBinSelector.Scott, 8 }, { HistogramBinSelector.Rice, 15 }, { HistogramBinSelector.Sturges, 14 }, { HistogramBinSelector.Stone, 8 }
                } },
                { 500, new() {
                    { HistogramBinSelector.Scott, 16 }, { HistogramBinSelector.Rice, 32 }, { HistogramBinSelector.Sturges, 20 }, { HistogramBinSelector.Stone, 80 }
                } },
                { 5000, new() {
                    { HistogramBinSelector.Scott, 33 }, { HistogramBinSelector.Rice, 69 }, { HistogramBinSelector.Sturges, 27 }, { HistogramBinSelector.Stone, 80 }
                } },
            };

            foreach ((int testlen, Dictionary<HistogramBinSelector, int> expectedResults) in basic_test.Select(item => (item.Key, item.Value))) {
                Tensor x1 = linspace(-10, -1, testlen / 5 * 2);
                Tensor x2 = linspace(1, 10, testlen / 5 * 3);
                Tensor x3 = linspace(-100, -50, testlen);
                Tensor x = concatenate(new Tensor[] { x1, x2, x3 });
                foreach ((HistogramBinSelector estimator, int numbins) in expectedResults.Select(item => (item.Key, item.Value))) {
                    (Tensor a, Tensor b) = torch.histogram(x, bins: estimator, range: (-20, 20));
                    Assert.Equal(numbins, a.shape[0]);
                }
            }
        }


        [Fact(Skip = "Very heavy on the compute")]
        public void TestSaveAndLoadLarger2GBTensor()
        {
            var tensor = torch.rand((long)int.MaxValue + 128, device: torch.CPU);

            var tempFile = Path.GetTempFileName();
            try {
                // Save to memory
                using (var fs = File.OpenWrite(tempFile))
                    tensor.Save(fs);

                // Create a new copy of zeros
                var copyTensor = torch.zeros_like(tensor, device: torch.CPU);

                // Read it in
                using (var fs = File.OpenRead(tempFile))
                    copyTensor.Load(new BinaryReader(fs));

                Assert.Equal(tensor.npstr(), copyTensor.npstr());
            } finally {
                File.Delete(tempFile);
            }
        }

        [Fact(Skip = "Very heavy on the compute")]
        public void TestSaveAndLoadLarger2GBTensorCUDA()
        {
            if (torch.cuda.is_available()) {
                var tensor = torch.rand((long)int.MaxValue + 128, device: torch.CUDA);

                var tempFile = Path.GetTempFileName();
                try {
                    // Save to memory
                    using (var fs = File.OpenWrite(tempFile))
                        tensor.Save(fs);

                    // Create a new copy of zeros
                    var copyTensor = torch.zeros_like(tensor, device: torch.CUDA);

                    // Read it in
                    using (var fs = File.OpenRead(tempFile))
                        copyTensor.Load(new BinaryReader(fs));

                    Assert.Equal(tensor.npstr(), copyTensor.npstr());
                } finally {
                    File.Delete(tempFile);
                }
            }
        }


        [Fact(Skip = "Very heavy on the compute")]
        public void TestSaveAndLoadModuleWithLarger2GBTensor()
        {
            // Create a sequential with a parameter slightly larger than 2GB
            var seq = nn.Sequential(("lin1", torch.nn.Linear(int.MaxValue / 2048, 2049, false)));

            var tempFile = Path.GetTempFileName();
            try {
                // Save to memory
                using (var fs = File.OpenWrite(tempFile))
                    seq.save(fs);

                // Create a new sequence, and make sure it is equal
                var copySeq = nn.Sequential(("lin1", torch.nn.Linear(int.MaxValue / 2048, 2049, false)));

                // Read it in
                using (var fs = File.OpenRead(tempFile))
                    copySeq.load(fs);

                // Compare results
                Assert.Equal(copySeq.parameters().First().npstr(), seq.parameters().First().npstr());
            } finally {
                File.Delete(tempFile);
            }
        }

        [Fact(Skip = "Very heavy on the compute")]
        public void TestSaveAndLoadModuleWithLarger2GBTensorCUDA()
        {
            if (torch.cuda.is_available()) {
                // Create a sequential with a parameter slightly larger than 2GB
                var seq = nn.Sequential(("lin1", torch.nn.Linear(int.MaxValue / 2048, 2049, false))).cuda();

                var tempFile = Path.GetTempFileName();
                try {
                    // Save to memory
                    using (var fs = File.OpenWrite(tempFile))
                        seq.save(fs);

                    // Create a new sequence, and make sure it is equal
                    var copySeq = nn.Sequential(("lin1", torch.nn.Linear(int.MaxValue / 2048, 2049, false))).cuda();

                    // Read it in
                    using (var fs = File.OpenRead(tempFile))
                        copySeq.load(fs);

                    // Compare results
                    Assert.Equal(copySeq.parameters().First().npstr(), seq.parameters().First().npstr());
                } finally {
                    File.Delete(tempFile);
                }
            }
        }

        [Fact]
        public void DefaultDTypeCreation()
        {
            var dt = torch.get_default_dtype();

            var t = torch.zeros(5, 5);
            Assert.Equal(torch.float32, t.dtype);

            try {
                torch.set_default_dtype(torch.float64);

                t = torch.zeros(5, 5);
                Assert.Equal(torch.float64, t.dtype);

                t = torch.ones(5, 5);
                Assert.Equal(torch.float64, t.dtype);

                t = torch.rand(5, 5);
                Assert.Equal(torch.float64, t.dtype);

                t = torch.randn(5, 5);
                Assert.Equal(torch.float64, t.dtype);

                t = torch.logspace(5, 15, 20);
                Assert.Equal(torch.float64, t.dtype);
            } finally {
                torch.set_default_dtype(dt);
            }
        }

        [Fact]
        public void DefaultDeviceCreation()
        {
            var dt = torch.get_default_device();

            var t = torch.zeros(5, 5);
            Assert.Equal(DeviceType.CPU, t.device_type);

            try {
                torch.set_default_device(torch.META);

                t = torch.zeros(5, 5);
                Assert.Equal(DeviceType.META, t.device_type);

                t = torch.ones(5, 5);
                Assert.Equal(DeviceType.META, t.device_type);

                t = torch.rand(5, 5);
                Assert.Equal(DeviceType.META, t.device_type);

                t = torch.randn(5, 5);
                Assert.Equal(DeviceType.META, t.device_type);

                t = torch.logspace(5, 15, 20);
                Assert.Equal(DeviceType.META, t.device_type);
            } finally {
                torch.set_default_device(dt);
            }
        }
    }
}