// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Globalization;
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

#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestTensor
    {
        [Fact]
        public void TestScalarCreation()
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
        public void TestScalarToString()
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

        private string _sep = Environment.NewLine;

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        [TestOf(nameof(TensorExtensionMethods.jlstr))]
        public void Test1DToString()
        {
            {
                Tensor t = torch.zeros(4);
                var str = t.ToString(torch.julia, cultureInfo: CultureInfo.InvariantCulture, newLine: _sep);
                Assert.Equal($"[4], type = Float32, device = cpu{_sep} 0 0 0 0{_sep}", str);
            }
            {
                Tensor t = torch.zeros(4);
                var str = t.jlstr(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[4], type = Float32, device = cpu\n 0 0 0 0\n", str);
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
        public void Test2DToString()
        {
            {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4);
                var str = t.ToString(true, cultureInfo: CultureInfo.InvariantCulture, newLine: _sep);
                Assert.Equal($"[2x4], type = Float32, device = cpu{_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}", str);
            }
            {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4);
                var str = t.str(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = Float32, device = cpu\n        0   3.141 6.2834 3.1415\n 6.28e-06 -13.142   0.01 4713.1\n", str);
            }
            if (torch.cuda.is_available()) {
                Tensor t = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4, device: torch.CUDA);
                var str = t.str(cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x4], type = Float32, device = cuda:0\n        0   3.141 6.2834 3.1415\n 6.28e-06 -13.142   0.01 4713.1\n", str);
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
        public void Test3DToString()
        {
            {
                Tensor t = torch.tensor(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, 2, 2, 4);
                var str = t.jlstr("0.0000000", cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[2x2x4], type = Float32, device = cpu\n[0,..,..] =\n 0.0000000   3.1410000 6.2834000    3.1415200\n" +
                             $" 0.0000063 -13.1415300 0.0100000 4713.1400000\n\n[1,..,..] =\n 0.0100000 0.0000000 0.0000000 0.0000000\n" +
                             $" 0.0000000 0.0000000 0.0000000 0.0000000\n", str);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test4DToString()
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
        public void Test5DToString()
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
        public void Test6DToString()
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
        [TestOf(nameof(Tensor.ToString))]
        public void Test1DToNumpyString()
        {
            Assert.Equal("[0 0 0 0]", torch.zeros(4).ToString(torch.numpy));
            Assert.Equal("[0 0 0 0]", torch.zeros(4, torch.complex64).ToString(torch.numpy));
            {
                Tensor t = torch.ones(4, torch.complex64);
                for (int i = 0; i < t.shape[0]; i++) t[i] = torch.tensor((1.0f * i, 2.43f * i * 2), torch.complex64);
                var str = t.ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[0 1+4.86i 2+9.72i 3+14.58i]", str);
            }
            Assert.Equal("[1 1 1 ... 1 1 1]", torch.ones(100, ScalarType.Int32).ToString(torch.numpy));
            Assert.Equal("[1 3 5 ... 195 197 199]", torch.tensor(Enumerable.Range(0, 100).Select(x => 2 * x + 1).ToList(), ScalarType.Float32).ToString(torch.numpy));
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test2DToNumpyString()
        {
            string str = torch.tensor(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
            Assert.Equal($"[[0 3.141 6.2834 3.1415]{_sep} [6.28e-06 -13.142 0.01 4713.1]]", str);
            {
                Tensor t = torch.zeros(5, 5, torch.complex64);
                for (int i = 0; i < t.shape[0]; i++)
                    for (int j = 0; j < t.shape[1]; j++)
                        t[i][j] = torch.tensor((1.24f * i, 2.491f * i * 2), torch.complex64);
                str = t.ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[[0 0 0 0 0]{_sep} [1.24+4.982i 1.24+4.982i 1.24+4.982i 1.24+4.982i 1.24+4.982i]{_sep} [2.48+9.964i 2.48+9.964i 2.48+9.964i 2.48+9.964i 2.48+9.964i]{_sep} [3.72+14.946i 3.72+14.946i 3.72+14.946i 3.72+14.946i 3.72+14.946i]{_sep} [4.96+19.928i 4.96+19.928i 4.96+19.928i 4.96+19.928i 4.96+19.928i]]", str);
            }
            Assert.Equal($"[[0 0 0 0]{_sep} [0 0 0 0]]", torch.zeros(2, 4, torch.complex64).ToString(torch.numpy));
            Assert.Equal($"[[1 1 1 1]{_sep} [1 1 1 1]]", torch.ones(2, 4, torch.complex64).ToString(torch.numpy));
            Assert.Equal($"[[7 9 11 ... 101 103 105]{_sep} [107 109 111 ... 201 203 205]]", torch.tensor(Enumerable.Range(1, 100).Select(x => x * 2 + 5).ToList(), new long[] { 2, 50 }, ScalarType.Float32).ToString(torch.numpy));
            Assert.Equal($"[[7 9 11 ... 201 203 205]{_sep} [207 209 211 ... 401 403 405]{_sep} [407 409 411 ... 601 603 605]{_sep} ...{_sep} [19407 19409 19411 ... 19601 19603 19605]{_sep} [19607 19609 19611 ... 19801 19803 19805]{_sep} [19807 19809 19811 ... 20001 20003 20005]]",
                torch.tensor(Enumerable.Range(1, 10000).Select(x => x * 2 + 5).ToList(), new long[] { 100, 100 },
                    ScalarType.Float32).ToString(torch.numpy));
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test3DToNumpyString()
        {
            {
                Assert.Equal(
                    $"[[[0 3.141 6.2834 3.1415]{_sep}  [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep} [[0.01 0 0 0]{_sep}  [0 0 0 0]]]",
                    torch.tensor(
                        new float[] {
                            0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f, 0.01f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        }, 2, 2, 4).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
                var actual = torch.tensor(Enumerable.Range(1, 250).Select(x => x * 2 + 5).ToList(), new long[] { 5, 5, 10 }, ScalarType.Float32).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture);
                Assert.Equal($"[[[7 9 11 ... 21 23 25]{_sep}  [27 29 31 ... 41 43 45]{_sep}  [47 49 51 ... 61 63 65]{_sep}  [67 69 71 ... 81 83 85]{_sep}  [87 89 91 ... 101 103 105]]{_sep}{_sep} [[107 109 111 ... 121 123 125]{_sep}  [127 129 131 ... 141 143 145]{_sep}  [147 149 151 ... 161 163 165]{_sep}  [167 169 171 ... 181 183 185]{_sep}  [187 189 191 ... 201 203 205]]{_sep}{_sep} [[207 209 211 ... 221 223 225]{_sep}  [227 229 231 ... 241 243 245]{_sep}  [247 249 251 ... 261 263 265]{_sep}  [267 269 271 ... 281 283 285]{_sep}  [287 289 291 ... 301 303 305]]{_sep}{_sep} [[307 309 311 ... 321 323 325]{_sep}  [327 329 331 ... 341 343 345]{_sep}  [347 349 351 ... 361 363 365]{_sep}  [367 369 371 ... 381 383 385]{_sep}  [387 389 391 ... 401 403 405]]{_sep}{_sep} [[407 409 411 ... 421 423 425]{_sep}  [427 429 431 ... 441 443 445]{_sep}  [447 449 451 ... 461 463 465]{_sep}  [467 469 471 ... 481 483 485]{_sep}  [487 489 491 ... 501 503 505]]]",
                    torch.tensor(Enumerable.Range(1, 250).Select(x => x * 2 + 5).ToList(), new long[] { 5, 5, 10 }, ScalarType.Float32).ToString(torch.numpy, cultureInfo: CultureInfo.InvariantCulture));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.ToString))]
        public void Test4DToNumpyString()
        {
            Assert.Equal($"[[[[0 3.141 6.2834 3.1415]{_sep}   [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}  [[0.01 0 0 0]{_sep}   [0 0 0 0]]]{_sep}{_sep}{_sep} [[[0 3.141 6.2834 3.1415]{_sep}   [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}  [[0.01 0 0 0]{_sep}   [0 0 0 0]]]]", torch.tensor(new float[] {
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
                $"[[[[[0 3.141 6.2834 3.1415]{_sep}    [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}   [[0.01 0 0 0]{_sep}    [0 0 0 0]]]{_sep}{_sep}{_sep}  [[[0 3.141 6.2834 3.1415]{_sep}    [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}   [[0.01 0 0 0]{_sep}    [0 0 0 0]]]]{_sep}{_sep}{_sep}{_sep} [[[[0 3.141 6.2834 3.1415]{_sep}    [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}   [[0.01 0 0 0]{_sep}    [0 0 0 0]]]{_sep}{_sep}{_sep}  [[[0 3.141 6.2834 3.1415]{_sep}    [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}   [[0.01 0 0 0]{_sep}    [0 0 0 0]]]]]",
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
                $"[[[[[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]{_sep}{_sep}{_sep}   [[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]]{_sep}{_sep}{_sep}{_sep}  [[[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]{_sep}{_sep}{_sep}   [[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]]]{_sep}{_sep}{_sep}{_sep}{_sep} [[[[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]{_sep}{_sep}{_sep}   [[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]]{_sep}{_sep}{_sep}{_sep}  [[[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]{_sep}{_sep}{_sep}   [[[0 3.141 6.2834 3.1415]{_sep}     [6.28e-06 -13.142 0.01 4713.1]]{_sep}{_sep}    [[0.01 0 0 0]{_sep}     [0 0 0 0]]]]]]",
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
        public void TestAlias()
        {
            var t = torch.randn(5);
            var t1 = t.alias();

            Assert.NotEqual(t.Handle, t1.Handle);
            t[0] = torch.tensor(3.14f);
            Assert.Equal(3.14f, t1[0].item<float>());
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void TestAliasDispose()
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

#if !LINUX
        [Fact(Skip = "Sensitive to parallelism in the xUnit test driver")]
        [TestOf(nameof(torch.randn))]
        public void TestUsings()
        {
            var tCount = Tensor.TotalCount;

            using (var t = torch.randn(5)) { }

            Assert.Equal(tCount, Tensor.TotalCount);
        }
#endif

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void TestDataBool()
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
        public void TestDataByte()
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
        public void TestDataInt8()
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
        public void TestDataInt16()
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
        public void TestDataInt32()
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
        public void TestDataInt64()
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
        public void TestDataFloat32()
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
        public void TestDataFloat64()
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
        public void TestDataItemBool()
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
        public void TestDataItemByte()
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
        public void TestDataItemInt8()
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
        public void TestDataItemInt16()
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
        public void TestDataItemInt32()
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
        public void TestDataItemInt64()
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
        public void TestDataItemFloat32()
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
        public void TestDataItemFloat64()
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
        public void TestFromArrayFactory()
        {
            {
                var array = new bool[8];
                var t = torch.tensor(array);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(ScalarType.Bool, t.dtype));
            }

            {
                var array = new bool[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(ScalarType.Bool, t.dtype));
            }

            {
                var array = new int[8];
                var t = torch.tensor(array);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(ScalarType.Int32, t.dtype));
            }

            {
                var array = new float[8];
                var t = torch.tensor(array);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(ScalarType.Float32, t.dtype));
            }

            {
                var array = new double[1, 2];
                var t = torch.from_array(array);
                Assert.Multiple(
                    () => Assert.Equal(2, t.ndim),
                    () => Assert.Equal(new long[] { 1, 2 }, t.shape),
                    () => Assert.Equal(ScalarType.Float64, t.dtype));
            }

            {
                var array = new long[1, 2, 3];
                var t = torch.from_array(array);
                Assert.Multiple(
                    () => Assert.Equal(3, t.ndim),
                    () => Assert.Equal(new long[] { 1, 2, 3 }, t.shape),
                    () => Assert.Equal(ScalarType.Int64, t.dtype));
            }

            {
                var array = new int[1, 2, 3, 4];
                var t = torch.from_array(array);
                Assert.Multiple(
                    () => Assert.Equal(4, t.ndim),
                    () => Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape),
                    () => Assert.Equal(ScalarType.Int32, t.dtype));
            }

            {
                var array = new System.Numerics.Complex[1, 2, 3, 4];
                var t = torch.from_array(array);
                Assert.Multiple(
                    () => Assert.Equal(4, t.ndim),
                    () => Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape),
                    () => Assert.Equal(ScalarType.ComplexFloat64, t.dtype));
            }

            {
                var array = new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.from_array(array);
                Assert.Multiple(
                    () => Assert.Equal(3, t.ndim),
                    () => Assert.Equal(new long[] { 2, 2, 2 }, t.shape),
                    () => Assert.Equal(ScalarType.Float64, t.dtype),
                    () => Assert.Equal(array.Cast<double>().ToArray(), t.data<double>().ToArray()));
            }
        }

        [Fact]
        [TestOf(nameof(torch.frombuffer))]
        public void TestFromBufferFactory()
        {
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
        public void TestMDTensorFactorySByte()
        {
            {
                var array = new sbyte[8];
                var t = torch.tensor(array);
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(ScalarType.Int8, t.dtype));
            }

            {
                var array = new sbyte[8];
                var t = torch.tensor(array, new long[] { 8 });
                Assert.Multiple(
                    () => Assert.Equal(1, t.ndim),
                    () => Assert.Equal(ScalarType.Int8, t.dtype));
            }

            {
                var array = new sbyte[1, 2];
                var t = torch.tensor(array);
                Assert.Multiple(
                () => Assert.Equal(2, t.ndim),
                () => Assert.Equal(new long[] { 1, 2 }, t.shape),
                () => Assert.Equal(ScalarType.Int8, t.dtype));
            }

            {
                var array = new sbyte[1, 2, 3];
                var t = torch.tensor(array);
                Assert.Multiple(
                () => Assert.Equal(3, t.ndim),
                () => Assert.Equal(new long[] { 1, 2, 3 }, t.shape),
                () => Assert.Equal(ScalarType.Int8, t.dtype));
            }

            {
                var array = new sbyte[1, 2, 3, 4];
                var t = torch.tensor(array);
                Assert.Multiple(
                () => Assert.Equal(4, t.ndim),
                () => Assert.Equal(new long[] { 1, 2, 3, 4 }, t.shape),
                () => Assert.Equal(ScalarType.Int8, t.dtype));
            }

            {
                var array = new sbyte[100, 100, 100];
                var t = torch.tensor(array);
                Assert.Multiple(
                () => Assert.Equal(3, t.ndim),
                () => Assert.Equal(new long[] { 100, 100, 100 }, t.shape),
                () => Assert.Equal(ScalarType.Int8, t.dtype));
            }

            {
                var array = new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
                var t = torch.tensor(array);
                Assert.Multiple(
                () => Assert.Equal(3, t.ndim),
                () => Assert.Equal(new long[] { 2, 2, 2 }, t.shape),
                () => Assert.Equal(ScalarType.Int8, t.dtype),
                () => Assert.Equal(array.Cast<sbyte>().ToArray(), t.data<sbyte>().ToArray()));
            }
        }

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void TestMDTensorFactoryInt16()
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
        public void TestMDTensorFactoryInt32()
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
        public void TestMDTensorFactoryInt64()
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

        [Fact]
        [TestOf(nameof(torch.tensor))]
        public void TestMDTensorFactoryFloat32()
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
        public void TestMDTensorFactoryFloat64()
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
        public void TestMDTensorFactoryComplexFloat32()
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
        public void TestMDTensorFactoryComplexFloat64()
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
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = torch.zeros(shape, device: device, dtype: torch.float16);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(0.0f, t[0, 0].ToSingle());
                    Assert.Equal(0.0f, t[1, 1].ToSingle());
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.zeros))]
        public void CreateBFloat16TensorZeros()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = torch.zeros(shape, device: device, dtype: torch.bfloat16);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(0.0f, t[0, 0].ToSingle());
                    Assert.Equal(0.0f, t[1, 1].ToSingle());
                }
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

            t.rename_(null);
            //Assert.False(t.has_names());  FAILS ON RELEASE BUILDS

            t.rename_(new[] { "Batch", "Channels" });
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
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = torch.empty(shape, float16, device: device);
                    Assert.Equal(shape, t.shape);
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.empty))]
        public void CreateBFloat16TensorEmpty()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = torch.empty(shape, bfloat16, device: device);
                    Assert.Equal(shape, t.shape);
                }
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
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = torch.ones(shape, float16, device: device);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(1.0f, t[0, 0].ToSingle());
                    Assert.Equal(1.0f, t[1, 1].ToSingle());
                }
            }
        }

        [Fact]
        [TestOf(nameof(torch.ones))]
        public void CreateBFloat16TensorOnes()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = torch.ones(shape, bfloat16, device: device);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(1.0f, t[0, 0].ToSingle());
                    Assert.Equal(1.0f, t[1, 1].ToSingle());
                }
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
        [TestOf(nameof(TensorExtensionMethods.ToTensor))]
        public void TestScalarToTensor()
        {
            Assert.Throws<ArgumentException>(() => 1.ToTensor(requires_grad: true));
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void TestScalarToTensor2()
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
        public void TestScalarToTensor3()
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
        public void TestNegativeScalarToTensor()
        {
            Scalar s = 10;
            TensorIndex ti = 10;
            Tensor t;

            Assert.Throws<InvalidOperationException>(() => { t = s; });
            Assert.Throws<InvalidOperationException>(() => { t = ti; });
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.calculate_gain))]
        public void TestCalculateGain()
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
            tensor.fill_(float.PositiveInfinity);
            Assert.True(tensor.isposinf().data<bool>().ToArray().All(b => b));
            Assert.True(tensor.isinf().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isneginf().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isfinite().data<bool>().ToArray().All(b => b));
            Assert.False(tensor.isnan().data<bool>().ToArray().All(b => b));

            tensor.fill_(float.NegativeInfinity);
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
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.zeros_(tensor)) { }
            using (var res = torch.nn.init.zeros_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.ones_))]
        public void InitOnes()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.ones_(tensor)) { }
            using (var res = torch.nn.init.ones_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.dirac_))]
        public void InitDirac()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.dirac_(tensor)) { }
            using (var res = torch.nn.init.dirac_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.eye_))]
        public void InitEye()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.eye_(tensor)) { }
            using (var res = torch.nn.init.eye_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.constant_))]
        public void InitConstant()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.constant_(tensor, Math.PI)) { }
            using (var res = torch.nn.init.constant_(tensor, Math.PI)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.uniform_))]
        public void InitUniform()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.uniform_(tensor)) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = torch.nn.init.uniform_(tensor, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.normal_))]
        public void InitNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.normal_(tensor)) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = torch.nn.init.normal_(tensor, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.trunc_normal_))]
        public void InitTruncNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.trunc_normal_(tensor, 0, 1, -0.5, 0.5)) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = torch.nn.init.trunc_normal_(tensor, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.orthogonal_))]
        public void InitOrthogonal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.orthogonal_(tensor)) { }
            using (var res = torch.nn.init.orthogonal_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.sparse_))]
        public void InitSparse()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.sparse_(tensor, 0.25)) { }
            using (var res = torch.nn.init.sparse_(tensor, 0.25)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.kaiming_uniform_))]
        public void InitKaimingUniform()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.kaiming_uniform_(tensor)) { }
            using (var res = torch.nn.init.kaiming_uniform_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.kaiming_normal_))]
        public void InitKaimingNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.kaiming_normal_(tensor)) { }
            using (var res = torch.nn.init.kaiming_normal_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.xavier_uniform_))]
        public void InitXavierUniform()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.xavier_uniform_(tensor)) { }
            using (var res = torch.nn.init.xavier_uniform_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(torch.nn.init.xavier_normal_))]
        public void InitXavierNormal()
        {
            using Tensor tensor = torch.zeros(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = torch.nn.init.xavier_normal_(tensor)) { }
            using (var res = torch.nn.init.xavier_normal_(tensor)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.bernoulli_))]
        public void InplaceBernoulli()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.bernoulli_()) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.bernoulli_(generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.cauchy_))]
        public void InplaceCauchy()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.cauchy_()) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.cauchy_(generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.exponential_))]
        public void InplaceExponential()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.exponential_()) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.exponential_(generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.geometric_))]
        public void InplaceGeometric()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.geometric_(0.25)) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.geometric_(0.25, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.log_normal_))]
        public void InplaceLogNormal()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.log_normal_()) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.log_normal_(generator: gen)) { }
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
            // that there are two handles to the tensor is okay.
            using (var res = tensor.normal_()) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.normal_(generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.random_))]
        public void InplaceRandom()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.random_(0.0, 1.0)) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.random_(0, 1, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(Tensor.uniform_))]
        public void InplaceUniform()
        {
            using Tensor tensor = torch.empty(new long[] { 2, 2 });
            // Really just testing that the native interop works and that the fact
            // that there are two handles to the tensor is okay.
            using (var res = tensor.uniform_(0.0, 1.0)) { }
            using (var gen = new torch.Generator(4711L))
            using (var res = tensor.uniform_(0, 1, generator: gen)) { }
        }

        [Fact]
        [TestOf(nameof(torch.sparse))]
        public void TestSparse()
        {
            using var i = torch.tensor(new long[] { 0, 1, 1, 2, 0, 2 }, new long[] { 2, 3 });
            using Tensor v = new float[] { 3, 4, 5 };
            var sparse = torch.sparse(i, v, new long[] { 2, 3 });

            Assert.True(sparse.is_sparse);
            Assert.False(i.is_sparse);
            Assert.False(v.is_sparse);
            Assert.Equal(sparse.SparseIndices.data<long>().ToArray(), new long[] { 0, 1, 1, 2, 0, 2 });
            Assert.Equal(sparse.SparseValues.data<float>().ToArray(), new float[] { 3, 4, 5 });
        }

        [Fact]
        public void TestIndexerAndImplicitConversion()
        {
            var points = torch.zeros(6, torch.ScalarType.Float16);
            points[0] = 4.0f;
            points[1] = 1.0d;
            points[2] = 7;
        }

        [Fact]
        public void TestIndexSingle()
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
        public void TestIndexEllipsis()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Single(0) });
            Assert.Equal(0, t1[0].ToInt32());
            Assert.Equal(6, t1[1].ToInt32());
        }

        [Fact]
        public void TestIndexNull()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.None, TensorIndex.Single(0) });
            Assert.Equal(0, t1[0, 0].ToInt32());
            Assert.Equal(1, t1[0, 1].ToInt32());
            Assert.Equal(2, t1[0, 2].ToInt32());
        }

        [Fact]
        public void TestIndexNone()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
            var t1 = i.index(new TensorIndex[] { TensorIndex.None, TensorIndex.Single(0) });
            Assert.Equal(0, t1[0, 0].ToInt32());
            Assert.Equal(1, t1[0, 1].ToInt32());
            Assert.Equal(2, t1[0, 2].ToInt32());
        }

        [Fact]
        public void TestIndexSlice1()
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
        public void TestIndexSlice2()
        {
            using var i = torch.tensor(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 });
#if NET472_OR_GREATER
            var t1 = i[(0, 2), 0];
            Assert.Equal(0, t1[0].ToInt32());
            Assert.Equal(6, t1[1].ToInt32());

            // one slice
            var t2 = i[(1, 2), 0];
            Assert.Equal(6, t2[0].ToInt32());

            // two slice
            var t3 = i[(1, 2), (1, 3)];
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

            // two slice
            var t3 = i[1..2, 1..3];
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
            Tensor cpu = torch.ones(new long[] { 2, 2 }, device: torch.DirectML);
            Assert.Equal("privateuse1:0", cpu.device.ToString());

            if (torch.cuda.is_available()) {
                var cuda = cpu.cuda();
                Assert.Equal("cuda:0", cuda.device.ToString());

                // Copy back to CPU to inspect the elements
                var cpu2 = cuda.cpu();
                Assert.Equal("cpu", cpu2.device.ToString());
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
                Assert.Equal("cpu", cpu.device.ToString());

                var data = cpu.data<float>();
                for (int i = 0; i < 4; i++) {
                    Assert.Equal(1, data[i]);
                }
            } else {
                Assert.Throws<InvalidOperationException>((Action)(() => { torch.ones(new long[] { 2, 2 }, device: torch.CUDA); }));
            }
        }

        [Fact]
        public void TestSquareEuclideanDistance()
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
        public void TestCat()
        {
            var zeros = torch.zeros(new long[] { 1, 9 });
            var ones = torch.ones(new long[] { 1, 9 });
            var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);

            var shape = centroids.shape;
            Assert.Equal(new long[] { 2, 9 }, shape);
        }

        [Fact]
        public void TestCatCuda()
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
        public void TestCopy()
        {
            var first = torch.rand(new long[] { 1, 9 });
            var second = torch.zeros(new long[] { 1, 9 });

            second.copy_(first);

            Assert.Equal(first, second);
        }

        [Fact]
        [TestOf(nameof(Tensor.copy_))]
        public void TestCopyCuda()
        {
            if (torch.cuda.is_available()) {
                var first = torch.rand(new long[] { 1, 9 }).cuda();
                var second = torch.zeros(new long[] { 1, 9 }).cuda();

                second.copy_(first);

                Assert.Equal(first, second);
            }
        }

        void TestStackGen(Device device)
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
        public void TestCast()
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
        public void TestMoveAndCast()
        {
            var input = torch.rand(new long[] { 128 }, float64, torch.CPU);

            if (torch.cuda.is_available()) {
                var moved = input.to(ScalarType.Float32, torch.CUDA);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CUDA, moved.device_type);
            } else {
                var moved = input.to(ScalarType.Float32);
                Assert.Equal(ScalarType.Float32, moved.dtype);
                Assert.Equal(DeviceType.CPU, moved.device_type);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.to))]
        public void TestMeta()
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
        [TestOf(nameof(Tensor.masked_scatter))]
        public void TestMaskedScatter()
        {
            var input = torch.zeros(new long[] { 4, 4 });
            var mask = torch.zeros(new long[] { 4, 4 }, torch.@bool);
            var tTrue = torch.tensor(true);
            mask[0, 1] = tTrue;
            mask[2, 3] = tTrue;

            var res = input.masked_scatter(mask, torch.tensor(new float[] { 3.14f, 2 * 3.14f }));
            Assert.Equal(3.14f, res[0, 1].item<float>());
            Assert.Equal(2 * 3.14f, res[2, 3].item<float>());

            input.masked_scatter_(mask, torch.tensor(new float[] { 3.14f, 2 * 3.14f }));
            Assert.Equal(3.14f, input[0, 1].item<float>());
            Assert.Equal(2 * 3.14f, input[2, 3].item<float>());
        }

        [Fact]
        [TestOf(nameof(Tensor.masked_select))]
        public void TestMaskedSelect()
        {
            var input = torch.zeros(new long[] { 4, 4 });
            var mask = torch.eye(4, 4, torch.@bool);

            var res = input.masked_select(mask);
            Assert.Equal(4, res.numel());
        }

        [Fact]
        [TestOf(nameof(Tensor.diagonal_scatter))]
        public void TestDiagonalScatter()
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
        public void TestSliceScatter()
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
        public void TestStackCpu()
        {
            TestStackGen(torch.CPU);
        }

        [Fact]
        [TestOf(nameof(torch.CUDA))]
        public void TestStackCuda()
        {
            if (torch.cuda.is_available()) {
                TestStackGen(torch.CUDA);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.block_diag))]
        public void TestBlockDiag()
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
        public void TestDiag1D()
        {
            var input = torch.ones(new long[] { 3 }, int64);
            var expected = new long[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            var res = input.diag();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.diag))]
        public void TestDiag2D()
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
        public void TestDiagFlat1D()
        {
            var input = torch.ones(new long[] { 3 }, int64);
            var expected = new long[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            var res = input.diagflat();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.data<long>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.diagflat))]
        public void TestDiagFlat2D()
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
        public void TestDimensions()
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
        public void TestNumberofElements()
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
        public void TestElementSize()
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
        public void TestAtleast1d()
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
        public void TestAtleast2d()
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
        public void TestAtleast3d()
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
        public void TestSetGrad()
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
        public void TestAutoGradMode()
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
                    var grad = x.grad();
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
                    var grad = x.grad();
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
        public void TestSubInPlace()
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
        public void TestMemoryDisposalZeros()
        {
            for (int i = 0; i < 1024; i++) {
                var x = torch.zeros(new long[] { 1024, 1024 }, float64);
                x.Dispose();
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void TestMemoryDisposalOnes()
        {
            for (int i = 0; i < 1024; i++) {
                var x = torch.ones(new long[] { 1024, 1024 }, float64);
                x.Dispose();
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.Dispose))]
        public void TestMemoryDisposalScalarTensors()
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
        public void TestMemoryDisposalScalars()
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
        public void TestSaveLoadTensorDouble()
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
        public void TestSaveLoadTensorFloat()
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
        public void TestPositive()
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
        public void UnsqueezeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.1f };

            using (var res = torch.tensor(data).unsqueeze(0)) {
                Assert.Equal(new long[] { 1, 4 }, res.shape);
                Assert.Equal(1.1f, res[0, 0].ToSingle());
                Assert.Equal(2.0f, res[0, 1].ToSingle());
                Assert.Equal(3.1f, res[0, 2].ToSingle());
                Assert.Equal(4.1f, res[0, 3].ToSingle());
            }
            using (var res = torch.tensor(data).unsqueeze(1)) {
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


        [Fact]
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
        public void TestSpecialEntropy()
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
        public void TestSpecialErrorFunction()
        {
            var a = torch.tensor(new float[] { 0.0f, -1.0f, 10.0f });
            var expected = torch.tensor(
                    new float[] { 0.0f, -0.8427f, 1.0f });
            var b = torch.special.erf(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.erfc))]
        public void TestSpecialComplementaryErrorFunction()
        {
            var a = torch.tensor(new float[] { 0.0f, -1.0f, 10.0f });
            var expected = torch.tensor(
                    new float[] { 1.0f, 1.8427f, 0.0f });
            var b = torch.special.erfc(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.erfinv))]
        public void TestSpecialInverseErrorFunction()
        {
            var a = torch.tensor(new float[] { 0.0f, 0.5f, -1.0f });
            var expected = torch.tensor(
                    new float[] { 0.0f, 0.476936281f, Single.NegativeInfinity });
            var b = torch.special.erfinv(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.expit))]
        public void TestSpecialExpit()
        {
            var a = torch.randn(new long[] { 10 });
            var expected = torch.tensor(a.data<float>().ToArray().Select(x => 1.0f / (1.0f + MathF.Exp(-x))).ToArray());
            var b = torch.special.expit(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(special.expm1))]
        public void TestSpecialExpm1()
        {
            var a = torch.randn(new long[] { 10 });
            var expected = torch.tensor(a.data<float>().ToArray().Select(x => MathF.Exp(x) - 1.0f).ToArray());
            var b = torch.special.expm1(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(special.exp2))]
        public void TestSpecialExp2()
        {
            var a = torch.randn(new long[] { 10 });
            var expected = torch.tensor(a.data<float>().ToArray().Select(x => MathF.Pow(2.0f, x)).ToArray());
            var b = torch.special.exp2(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(special.gammaln))]
        public void TestSpecialGammaLN()
        {
            var a = torch.arange(0.5f, 2f, 0.5f);
            var expected = torch.tensor(new float[] { 0.5723649f, 0.0f, -0.120782226f });
            var b = torch.special.gammaln(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i0))]
        public void TestSpeciali0()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 0.99999994f, 1.266066f, 2.27958512f, 4.88079262f, 11.3019209f });
            var b = torch.special.i0(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i0e))]
        public void TestSpeciali0e()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 1.0f, 0.465759635f, 0.3085083f, 0.243000358f, 0.20700191f });
            var b = torch.special.i0e(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i1))]
        public void TestSpeciali1()
        {
            var a = torch.arange(0.0f, 5.0f, 1.0f);
            var expected = torch.tensor(new float[] { 0.0000f, 0.5651591f, 1.59063685f, 3.95337057f, 9.759467f });
            var b = torch.special.i1(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(special.i1e))]
        public void TestSpeciali1e()
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

        [Fact]
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
                var cropped = torchvision.transforms.RandomResizedCrop(100, 0.1, 0.5).forward(input);

                Assert.Equal(new long[] { 4, 3, 100, 100 }, cropped.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.CenterCrop))]
        public void CenterCropTensor()
        {
            {
                var input = torch.rand(25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).forward(input);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 13; i++) {
                    // Test the diagonal only.
                    Assert.Equal(input[5 + i, 6 + i], cropped[i, i]);
                }
            }
            {
                var input = torch.rand(3, 25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).forward(input);

                Assert.Equal(new long[] { 3, 15, 13 }, cropped.shape);
                for (int c = 0; c < 3; c++)
                    for (int i = 0; i < 13; i++) {
                        // Test the diagonal only.
                        Assert.Equal(input[c, 5 + i, 6 + i], cropped[c, i, i]);
                    }
            }
            {
                var input = torch.rand(16, 3, 25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).forward(input);

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
                var resized = torchvision.transforms.Resize(15).forward(input);

                Assert.Equal(new long[] { 16, 3, 15, 15 }, resized.shape);
            }
            {
                var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, int32);
                var resized = torchvision.transforms.Resize(15).forward(input);

                Assert.Equal(new long[] { 16, 3, 15, 15 }, resized.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Resize))]
        public void ResizeTensorUp()
        {
            {
                var input = torch.rand(16, 3, 25, 25);
                var resized = torchvision.transforms.Resize(50).forward(input);

                Assert.Equal(new long[] { 16, 3, 50, 50 }, resized.shape);
            }
            {
                var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, int32);
                var resized = torchvision.transforms.Resize(50).forward(input);

                Assert.Equal(new long[] { 16, 3, 50, 50 }, resized.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Grayscale))]
        public void GrayscaleTensor()
        {
            {
                var input = torch.rand(16, 3, 25, 25);
                var gray = torchvision.transforms.Grayscale().forward(input);

                Assert.Equal(new long[] { 16, 1, 25, 25 }, gray.shape);
            }
            {
                var input = torch.rand(16, 3, 25, 25);
                var gray = torchvision.transforms.Grayscale(3).forward(input);

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

                var poster = torchvision.transforms.Invert().forward(input);
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
                var poster = torchvision.transforms.Posterize(4).forward(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.randint(255, new long[] { 25 }, torch.uint8);
                var poster = torchvision.transforms.Posterize(4).forward(input);

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
                var poster = torchvision.transforms.AdjustSharpness(1.5).forward(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.randint(255, new long[] { 16, 3, 25, 25 }, torch.uint8);
                var poster = torchvision.transforms.AdjustSharpness(0.5).forward(input);

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
                var poster = torchvision.transforms.Rotate(90).forward(input);

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
                var poster = eq.forward(input);

                Assert.Equal(new long[] { 3, 25, 50 }, poster.shape);
            }
            {
                using var input = torch.randint(0, 256, new long[] { 16, 3, 25, 50 }, dtype: torch.uint8);
                var poster = eq.forward(input);

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
                var poster = ac.forward(input);

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
                var poster = gb4.forward(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.rand(16, 3, 25, 25);
                // Test even-number kernel size.
                var poster = gb4.forward(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.rand(16, 3, 25, 25);
                // Test odd-number kernel size.
                var poster = gb5.forward(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
            {
                using var input = torch.rand(16, 3, 25, 25);
                var random = torchvision.transforms.Randomize(gb4, 0.5);
                var poster = random.forward(input);

                Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
            }
        }

        [Fact]
        [TestOf(nameof(torchvision.transforms.Solarize))]
        public void SolarizeTensor()
        {
            {
                using var input = torch.rand(25);
                var poster = torchvision.transforms.Solarize(0.55).forward(input);

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

            var res = torchvision.transforms.HorizontalFlip().forward(input);
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

            var res = torchvision.transforms.VerticalFlip().forward(input);
            Assert.Equal(res, expected);
        }

        [Fact]
        [TestOf(nameof(Tensor.pin_memory))]
        public void TestPinnedMemory()
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
        public void TestReshape()
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
        public void TestFlattenNamed()
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
        public void TestRefineNames()
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
        public void TestRenameTensor()
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
        public void TestRenameNamedTensor()
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
        public void TestAlignTo()
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
        public void TestAlignAs()
        {
            using var input = torch.ones(8, 16, 32, names: new[] { "A", "B", "C" });
            using var t = input.align_as(torch.rand(16, 32, 8, 16, names: new[] { "D", "C", "A", "B" }));
            Assert.Equal(new long[] { 1, 32, 8, 16 }, t.shape);
            Assert.Equal(new[] { "D", "C", "A", "B" }, t.names);
        }

        [Fact]
        [TestOf(nameof(Tensor.unique))]
        public void TestUnique()
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
        public void TestUniqueConsequtive()
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
        public void TestStorage_Basic()
        {
            var x = torch.tensor(new long[] { 1, 1, 2, 2, 3, 1, 1, 2 });

            var st = x.storage<long>();

            Assert.NotNull(st);
            Assert.Equal(8, st.Count);
        }

        [Fact]
        [TestOf(nameof(Tensor.storage))]
        public void TestStorage_ToArray()
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
        public void TestStorage_Modify1()
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
        public void TestStorage_Modify2()
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
        public void TestStorage_Modify3()
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
        public void TestStorage_Modify4()
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
        public void TestStorage_Fill()
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
        public void TestStorage_Copy()
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
            var output = torch.stft(input, n_fft, hop_length: hop_length, win_length: win_length, window: window);
            Assert.Equal(new long[] { n_fft / 2 + 1, input.shape[0] / hop_length + 1, 2 }, output.shape);
            Assert.Equal(ScalarType.Float32, output.dtype);

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
        public void TestMeshGrid()
        {
            var shifts_x = torch.arange(0, 32, dtype: torch.ScalarType.Int32, device: torch.CPU);
            var shifts_y = torch.arange(0, 32, dtype: torch.ScalarType.Int32, device: torch.CPU);

            Tensor[] shifts = new Tensor[] { shifts_y, shifts_x };

            var result = torch.meshgrid(shifts, indexing: "ij");
            Assert.NotNull(result);
            Assert.Equal(shifts.Length, result.Length);
        }

        [Fact]
        public void TestFromFile()
        {
            var location = "tensor_åöä_ασδφεες_አስድፋስድፍ.dat";
            if (File.Exists(location)) File.Delete(location);
            var t = torch.from_file(location, true, 256 * 16);
            Assert.True(File.Exists(location));
        }

        [Fact]
        public void TestCartesianProd()
        {
            var a = torch.arange(1, 4);
            var b = torch.arange(4, 6);

            var expected = torch.from_array(new int[] { 1, 4, 1, 5, 2, 4, 2, 5, 3, 4, 3, 5 }).reshape(6, 2);

            var res = torch.cartesian_prod(a, b);
            Assert.Equal(expected, res);
        }

        [Fact]
        public void TestCombinations()
        {
            var t = torch.arange(5);
            Assert.Equal(0, torch.combinations(t, 0).numel());
            Assert.Equal(5, torch.combinations(t, 1).numel());
            Assert.Equal(20, torch.combinations(t, 2).numel());
            Assert.Equal(30, torch.combinations(t, 3).numel());
            Assert.Equal(105, torch.combinations(t, 3, true).numel());
        }

        [Fact]
        public void TestCDist()
        {
            var a = torch.randn(3, 2);
            var b = torch.randn(2, 2);
            var res = torch.cdist(a, b);

            Assert.Equal(3, res.shape[0]);
            Assert.Equal(2, res.shape[1]);
        }

        [Fact]
        public void TestRot90()
        {
            var a = torch.arange(8).view(2, 2, 2);
            var res = a.rot90();

            var data = res.data<long>().ToArray();
            Assert.Equal(new long[] { 2, 3, 6, 7, 0, 1, 4, 5 }, data);
        }

        [Fact]
        public void TestDiagembed()
        {
            var a = torch.randn(2, 3);
            var res = torch.diag_embed(a);

            Assert.Equal(3, res.ndim);
            Assert.Equal(2, res.shape[0]);
            Assert.Equal(3, res.shape[1]);
            Assert.Equal(3, res.shape[1]);
        }
    }
}