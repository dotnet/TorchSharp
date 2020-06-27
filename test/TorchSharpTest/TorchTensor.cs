// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Tensor;
using Xunit;

#nullable enable

namespace TorchSharp
{
    public class TestTorchTensor
    {
        [Fact]
        public void CreateFloatTensorOnes()
        {
            var shape = new long[] { 2, 2 };
            TorchTensor t = FloatTensor.Ones(shape);
            Assert.Equal(shape, t.Shape);
            Assert.Equal(1.0f, t[0, 0].DataItem<float>());
            Assert.Equal(1.0f, t[1, 1].DataItem<float>());
        }

        [Fact]
        public void CreateByteTensorOnes()
        {
            var shape = new long[] { 2, 2 };
            TorchTensor t = ByteTensor.Ones(shape);
            Assert.Equal(shape, t.Shape);
            Assert.Equal((byte)1, t[0,0].DataItem<byte>());
            Assert.Equal((byte)1, t[1,1].DataItem<byte>());
        }

        [Fact]
        public void CreateIntTensorOnes()
        {
            var shape = new long[] { 2, 2 };
            TorchTensor t = IntTensor.Ones(shape);
            Assert.Equal(shape, t.Shape);
            Assert.Equal(1, t[0,0].DataItem<int>());
            Assert.Equal(1, t[1,1].DataItem<int>());
        }

        [Fact]
        public void CreateLongTensorOnes()
        {
            var shape = new long[] { 2, 2 };

            TorchTensor t = LongTensor.Ones(shape);
            Assert.Equal(shape, t.Shape);
            Assert.Equal((long)1, t[0,0].DataItem<int>());
            Assert.Equal((long)1, t[1,1].DataItem<int>());
        }

        [Fact]
        public void CreateBoolTensorOnes()
        {
            var shape = new long[] { 2, 2 };

            TorchTensor t = BoolTensor.Ones(shape);
            Assert.Equal(shape, t.Shape);
            Assert.Equal((object)true, t[0,0].DataItem<bool>());
            Assert.Equal((object)true, t[1,1].DataItem<bool>());
        }

        [Fact]
        public void CreateHalfTensorOnes()
        {
            var shape = new long[] { 2, 2 };

            TorchTensor t = HalfTensor.Ones(shape);
            Assert.Equal(shape, t.Shape);
            Assert.Equal(1.0f, t.ReadHalf(0));
            Assert.Equal(1.0f, t.ReadHalf(3));
        }

        //[Fact]
        //public void CreateComplexFloatTensorZeros()
        //{
        //    var shape = new long[] { 2, 2 };

        //    TorchTensor t = ComplexFloatTensor.Zeros(shape);
        //    Assert.Equal(shape, t.Shape);
        //    t.ReadComplexFloat(0, out var r3, out var i3);
        //    Assert.Equal(0.0f, r3);
        //    Assert.Equal(0.0f, i3);
        //    t.ReadComplexFloat(3, out var r4, out var i4);
        //    Assert.Equal(0.0f, r4);
        //    Assert.Equal(0.0f, i4);

        //}

        //[Fact]
        //public void CreateComplexFloatTensorOnes()
        //{
        //    var shape = new long[] { 2, 2 };

        //    TorchTensor t = ComplexFloatTensor.Ones(shape);
        //    Assert.Equal(shape, t.Shape);
        //    t.ReadComplexFloat(0, out var r3, out var i3);
        //    Assert.Equal(1.0f, r3);
        //    Assert.Equal(0.0f, i3);
        //    t.ReadComplexFloat(3, out var r4, out var i4);
        //    Assert.Equal(1.0f, r4);
        //    Assert.Equal(0.0f, i4);

        //}

        //[Fact]
        //public void CreateComplexDoubleTensorZeros()
        //{
        //    var shape = new long[] { 2, 2 };

        //    TorchTensor t = ComplexDoubleTensor.Zeros(shape);
        //    Assert.Equal(shape, t.Shape);
        //    var v3 = t.ReadComplexDouble(0);
        //    Assert.Equal(0.0, v3.Real);
        //    Assert.Equal(0.0, v3.Imaginary);
        //    var v4 = t.ReadComplexDouble(3);
        //    Assert.Equal(0.0, v4.Real);
        //    Assert.Equal(0.0, v4.Imaginary);

        //}

        //[Fact]
        //public void CreateComplexDoubleTensorOnes()
        //{
        //    var shape = new long[] { 2, 2 };
        //    TorchTensor t = ComplexDoubleTensor.Ones(shape);
        //    Assert.Equal(shape, t.Shape);
        //    var v5 = t.ReadComplexDouble(0);
        //    Assert.Equal(new Complex(1.0, 0.0), v5);
        //    var v6 = t.ReadComplexDouble(3);
        //    Assert.Equal(new Complex(1.0, 0.0), v6);
        //}

        [Fact]
        public void CreateFloatTensorCheckMemory()
        {
            TorchTensor? ones = null;

            for (int i = 0; i < 10; i++)
            {
                using (var tmp = FloatTensor.Ones(new long[] { 100, 100, 100 }))
                {
                    ones = tmp;
                    Assert.NotNull(ones);
                }
            }
        }

        [Fact]
        public void CreateFloatTensorOnesCheckData()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            var data = ones.Data<float>();

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1.0, data[i]);
            }
        }

        [Fact]
        public void CreateFloatTensorZerosCheckData()
        {
            var zeros = FloatTensor.Zeros(new long[] { 2, 2 });
            var data = zeros.Data<float>();

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(0, data[i]);
            }
        }

        [Fact]
        public void CreateIntTensorOnesCheckData()
        {
            var ones = IntTensor.Ones(new long[] { 2, 2 });
            var data = ones.Data<int>();

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1, data[i]);
            }
        }

        [Fact]
        public void CreateFloatTensorCheckDevice()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            var device = ones.DeviceString;

            Assert.Equal("cpu", ones.DeviceString);
        }

        [Fact]
        public void CreateFloatTensorFromData()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = FloatTensor.From(data, new long[] { 100, 10 }))
            {
                Assert.Equal(1, tensor.Data<float>()[100]);
            }
        }

        [Fact]
        public void CreateFloatTensorFromDataCheckDispose()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = FloatTensor.From(data, new long[] { 100, 10 }))
            {
                Assert.Equal(1, tensor.Data<float>()[100]);
            }

            Assert.Equal(1, data[100]);
        }

        [Fact]
        public void CreateFloatTensorFromData2()
        {
            var data = new float[1000];

            using (var tensor = data.ToTorchTensor(new long[] { 10, 100 })) {
                Assert.Equal(default(float), tensor.Data<float>()[100]);
            }
        }

        [Fact]
        public void CreateFloatTensorFromDataCheckStrides()
        {
            var data = new double[] { 0.2663158, 0.1144736, 0.1147367, 0.1249998, 0.1957895, 0.1231576, 0.1944732, 0.111842, 0.1065789, 0.667881, 0.5682123, 0.5824502, 0.4824504, 0.4844371, 0.6463582, 0.5334439, 0.5079474, 0.2281452 };
            var dataTensor = data.ToTorchTensor(new long[] { 2, 9 });

            for (int r = 0; r < 2; r++)
            {
                for (int i = 0; i < 9; i++)
                {
                    var fromData = data[(r * 9) + i];
                    var fromTensor = dataTensor[r, i].DataItem<double>();
                    Assert.True(Math.Abs(fromData - fromTensor) < 0.0001);
                }
            }

            var firstHalf = dataTensor[0];

            for (int i = 0; i < 9; i++)
            {
                var fromData = data[i];
                var fromChunk = firstHalf[i].DataItem<double>();
                Assert.True(Math.Abs(fromData - fromChunk) < 0.0001);
            }
        }


        [Fact]
        public void CreateHalfTensorFromDataCheckStrides()
        {
            var data = new float[] { 0.2663158f, 0.1144736f, 0.1147367f, 0.1249998f, 0.1957895f, 0.1231576f, 0.1944732f, 0.111842f, 0.1065789f, 0.667881f, 0.5682123f, 0.5824502f, 0.4824504f, 0.4844371f, 0.6463582f, 0.5334439f, 0.5079474f, 0.2281452f };
            var dataTensor = HalfTensor.From(data, new long[] { 2, 9 });

            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < 9; i++) {
                    var fromData = data[(r * 9) + i];
                    var fromTensor = dataTensor.ReadHalf((r * 9) + i);
                    Assert.True(Math.Abs(fromData - fromTensor) < 0.01);
                }
            }

            var firstHalf = dataTensor[0];

            for (int i = 0; i < 9; i++) {
                var fromData = data[i];
                var fromChunk = firstHalf.ReadHalf(i);
                Assert.True(Math.Abs(fromData - fromChunk) < 0.01);
            }
        }


        [Fact]
        public void CreateFloatTensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = FloatTensor.From(scalar))
            {
                Assert.Equal(333.0f, tensor.DataItem<float>());
            }
        }

        [Fact]
        public void CreateHalfTensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = HalfTensor.From(scalar)) {
                Assert.Equal(333.0f, tensor.ReadHalf(0));
            }
        }

        [Fact]
        public void CreateFloatTensorFromScalar2()
        {
            float scalar = 333.0f;

            using (var tensor = scalar.ToTorchTensor())
            {
                Assert.Equal(333, tensor.DataItem<float>());
            }
        }

        [Fact]
        public void TestScalarToTensor()
        {
            Assert.Throws<ArgumentException>(() => 1.ToTorchTensor(requiresGrad: true));
        }

        [Fact]
        public void TestScalarToTensor2()
        {
            using (var tensor = 1.ToTorchTensor())
            {
                Assert.Equal(ScalarType.Int, tensor.Type);
                Assert.Equal(1, tensor.DataItem<int>());
            }
            using (var tensor = ((byte)1).ToTorchTensor())
            {
                Assert.Equal(ScalarType.Byte, tensor.Type);
                Assert.Equal(1, tensor.DataItem<byte>());
            }
            using (var tensor = ((sbyte)-1).ToTorchTensor())
            {
                Assert.Equal(ScalarType.SByte, tensor.Type);
                Assert.Equal(-1, tensor.DataItem<sbyte>());
            }
            using (var tensor = ((short)-1).ToTorchTensor())
            {
                Assert.Equal(ScalarType.Short, tensor.Type);
                Assert.Equal(-1, tensor.DataItem<short>());
            }
            using (var tensor = ((long)-1).ToTorchTensor())
            {
                Assert.Equal(ScalarType.Long, tensor.Type);
                Assert.Equal(-1L, tensor.DataItem<short>());
            }
            using (var tensor = ((float)-1).ToTorchTensor())
            {
                Assert.Equal(ScalarType.Float, tensor.Type);
                Assert.Equal(-1.0f, tensor.DataItem<float>());
            }
            using (var tensor = ((double)-1).ToTorchTensor())
            {
                Assert.Equal(ScalarType.Double, tensor.Type);
                Assert.Equal(-1.0, tensor.DataItem<double>());
            }
        }

        [Fact]
        public void InitUniform()
        {
            using (TorchTensor tensor = FloatTensor.Zeros(new long[] { 2, 2 }))
            {
                NN.Init.Uniform(tensor);
            }
        }

        [Fact]
        public void TestSparse()
        {
            using (var i = LongTensor.From(new long[] { 0, 1, 1, 2, 0, 2 }, new long[] { 2, 3 }))
            using (var v = FloatTensor.From(new float[] { 3, 4, 5 }, new long[] { 3 }))
            {
                var sparse = FloatTensor.Sparse(i, v, new long[] { 2, 3 });

                Assert.True(sparse.IsSparse);
                Assert.False(i.IsSparse);
                Assert.False(v.IsSparse);
                Assert.Equal(sparse.SparseIndices.Data<long>().ToArray(), new long[] { 0, 1, 1, 2, 0, 2 });
                Assert.Equal(sparse.SparseValues.Data<float>().ToArray(), new float[] { 3, 4, 5 });
            }
        }

        [Fact]
        public void CopyCpuToCuda()
        {
            TorchTensor cpu = FloatTensor.Ones(new long[] { 2, 2 });
            Assert.Equal("cpu", cpu.DeviceString);

            if (Torch.IsCudaAvailable())
            {
                var cuda = cpu.Cuda();
                Assert.Equal("cuda:0", cuda.DeviceString);

                // Copy back to CPU to inspect the elements
                var cpu2 = cuda.Cpu();
                Assert.Equal("cpu", cpu2.DeviceString);
                var data = cpu.Data<float>();
                for (int i = 0; i < 4; i++)
                {
                    Assert.Equal(1, data[i]);
                }
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => cpu.Cuda());
            }

        }

        [Fact]
        public void CopyCudaToCpu()
        {
            if (Torch.IsCudaAvailable())
            {
                var cuda = FloatTensor.Ones(new long[] { 2, 2 }, DeviceType.CUDA);
                Assert.Equal("cuda:0", cuda.DeviceString);

                var cpu = cuda.Cpu();
                Assert.Equal("cpu", cpu.DeviceString);

                var data = cpu.Data<float>();
                for (int i = 0; i < 4; i++)
                {
                    Assert.Equal(1, data[i]);
                }
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => { FloatTensor.Ones(new long[] { 2, 2 }, DeviceType.CUDA); });
            }
        }

        [Fact]
        public void TestSquareEuclideanDistance()
        {
            var input = new double[] { 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1 }.ToTorchTensor(new long[] { 9 }).ToType(ScalarType.Float);
            var zeros = FloatTensor.Zeros(new long[] { 1, 9 });
            var ones = FloatTensor.Ones(new long[] { 1, 9 });
            var centroids = new TorchTensor[] { zeros, ones }.Cat(0);

            var distanceFromZero = input.Reshape(new long[] { -1, 1, 9 }).Sub(zeros).Pow(2.ToScalar()).Sum(new long[] { 2 });
            var distanceFromOne = input.Reshape(new long[] { -1, 1, 9 }).Sub(ones).Pow(2.ToScalar()).Sum(new long[] { 2 });
            var distanceFromCentroids = input.Reshape(new long[] { -1, 1, 9 }).Sub(centroids).Pow(2.ToScalar()).Sum(new long[] { 2 });

            Assert.True(true);
        }

        [Fact]
        public void TestCat()
        {
            var zeros = FloatTensor.Zeros(new long[] { 1, 9 });
            var ones = FloatTensor.Ones(new long[] { 1, 9 });
            var centroids = new TorchTensor[] { zeros, ones }.Cat(0);

            var shape = centroids.Shape;
            Assert.Equal(new long[] { 2, 9 }, shape);
        }

        [Fact]
        public void TestCatCuda()
        {
            if (Torch.IsCudaAvailable()) {
                var zeros = FloatTensor.Zeros(new long[] { 1, 9 }).Cuda();
                var ones = FloatTensor.Ones(new long[] { 1, 9 }).Cuda();
                var centroids = new TorchTensor[] { zeros, ones }.Cat(0);
                var shape = centroids.Shape;
                Assert.Equal(new long[] { 2, 9 }, shape);
                Assert.Equal(DeviceType.CUDA, centroids.DeviceType);
            }
        }

        void TestStackGen(DeviceType device)
        {
            {
                var t1 = FloatTensor.Zeros( new long[] { }, device );
                var t2 = FloatTensor.Ones(new long[] { }, device);
                var t3 = FloatTensor.Ones(new long[] { }, device);
                var res = new TorchTensor[] { t1, t2, t3 }.Stack(0);

                var shape = res.Shape;
                Assert.Equal(new long[] { 3 }, shape);
                Assert.Equal(device, res.DeviceType);
            }
            {
                var t1 = FloatTensor.Zeros(new long[] { 2, 9 }, device);
                var t2 = FloatTensor.Ones(new long[] { 2, 9 }, device);
                var res = new TorchTensor[] { t1, t2 }.Stack(0);

                var shape = res.Shape;
                Assert.Equal(new long[] { 2, 2, 9 }, shape);
                Assert.Equal(device, res.DeviceType);
            }
        }

        [Fact]
        public void TestStackCpu()
        {
            TestStackGen(DeviceType.CPU);
        }

        [Fact]
        public void TestStackCuda()
        {
            if (Torch.IsCudaAvailable()) {
                TestStackGen(DeviceType.CUDA);
            }
        }

        [Fact]
        public void TestSetGrad()
        {
            var x = FloatTensor.Random(new long[] { 10, 10 });
            Assert.False(x.IsGradRequired);

            x.RequiresGrad(true);
            Assert.True(x.IsGradRequired);
            x.RequiresGrad(false);
            Assert.False(x.IsGradRequired);
        }

        [Fact(Skip = "Not working on MacOS (note: may now be working, we need to recheck)")]
        public void TestAutoGradMode()
        {
            var x = FloatTensor.RandomN(new long[] { 2, 3 }, requiresGrad: true);
            using (var mode = new AutoGradMode(false))
            {
                Assert.False(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                Assert.Throws<ExternalException>(() => sum.Backward());
                //var grad = x.Grad();
                //Assert.True(grad.Handle == IntPtr.Zero);
            }
            using (var mode = new AutoGradMode(true))
            {
                Assert.True(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                sum.Backward();
                var grad = x.Grad();
                Assert.False(grad.Handle == IntPtr.Zero);
                var data = grad.Data<float>();
                for (int i = 0; i < 2 * 3; i++)
                {
                    Assert.Equal(1.0, data[i]);
                }
            }
        }

        [Fact]
        public void TestSubInPlace()
        {
            var x = IntTensor.Ones(new long[] { 100, 100 });
            var y = IntTensor.Ones(new long[] { 100, 100 });

            x.SubInPlace(y);

            var xdata = x.Data<int>();

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.Equal(0, xdata[i + j]);
                }
            }
        }

        [Fact]
        public void TestMemoryDisposalZeros()
        {
            for (int i = 0; i < 1024; i++) {
                var x = DoubleTensor.Zeros(new long[] { 1024, 1024 });
                x.Dispose();
                //System.GC.Collect();
            }
        }

        [Fact]
        public void TestMemoryDisposalOnes()
        {
            for (int i = 0; i < 1024; i++) {
                var x = DoubleTensor.Ones(new long[] { 1024, 1024 });
                x.Dispose();
            }
        }

        [Fact]
        public void TestMemoryDisposalScalarTensors()
        {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 1000 * 100; j++) {
                    var x = DoubleTensor.From(i * j * 3.1415);
                    x.Dispose();
                }
                //System.GC.Collect();
            }
        }

        [Fact]
        public void TestMemoryDisposalScalars()
        {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 1000 * 100; j++) {
                    var x = (i * j * 3.1415).ToScalar();
                    x.Dispose();
                }
                //System.GC.Collect();
            }
        }


        [Fact]
        public void TestSaveLoadTensorDouble()
        {
            var file = ".saveload.double.ts";
            if (File.Exists(file)) File.Delete(file);
            var tensor = DoubleTensor.Ones(new long[] { 5, 6 });
            tensor.Save(file);
            var tensorLoaded = TorchTensor.Load(file);
            File.Delete(file);
            Assert.NotNull(tensorLoaded);
            Assert.Equal(tensorLoaded.Type, tensor.Type);
            Assert.Equal(tensorLoaded, tensor);
        }

        [Fact]
        public void TestSaveLoadTensorFloat()
        {
            var file = ".saveload.float.ts";
            if (File.Exists(file)) File.Delete(file);
            var tensor = FloatTensor.Ones(new long[] { 5, 6 });
            tensor.Save(file);
            var tensorLoaded = TorchTensor.Load(file);
            File.Delete(file);
            Assert.NotNull(tensorLoaded);
            Assert.Equal(tensorLoaded.Type, tensor.Type);
            Assert.Equal(tensorLoaded, tensor);
        }


        [Fact]
        public void TestArithmeticOperators()
        {
            // scalar-tensor operators
            TestOneTensor<float, float>(a => a + 0.5f, a => a + 0.5f);
            TestOneTensor<float, float>(a => 0.5f + a, a => 0.5f + a);
            TestOneTensor<float, float>(a => a - 0.5f, a => a - 0.5f);
            TestOneTensor<float, float>(a => 0.5f - a, a => 0.5f - a);
            TestOneTensor<float, float>(a => a * 0.5f, a => a * 0.5f);
            TestOneTensor<float, float>(a => 0.5f * a, a => 0.5f * a);
            TestOneTensor<float, float>(a => a / 0.5f, a => a / 0.5f);
            TestOneTensor<float, float>(a => 0.5f / a, a => 0.5f / a);

            TestOneTensor<float, float>(a => a.Add(0.5f), a => a + 0.5f);
            TestOneTensor<float, float>(a => a.Sub(0.5f), a => a - 0.5f);
            TestOneTensor<float, float>(a => a.Mul(0.5f), a => a * 0.5f);
            TestOneTensor<float, float>(a => a.Div(0.5f), a => a / 0.5f);

            TestOneTensorInPlace<float>(a => a.AddInPlace(0.5f), a => a + 0.5f);
            TestOneTensorInPlace<float>(a => a.SubInPlace(0.5f), a => a - 0.5f);
            TestOneTensorInPlace<float>(a => a.MulInPlace(0.5f), a => a * 0.5f);
            TestOneTensorInPlace<float>(a => a.DivInPlace(0.5f), a => a / 0.5f);

            // tensor-tensor operators
            TestTwoTensor<float, float>((a, b) => a + b, (a, b) => a + b);
            TestTwoTensor<float, float>((a, b) => a - b, (a, b) => a - b);
            TestTwoTensor<float, float>((a, b) => a * b, (a, b) => a * b);
            TestTwoTensor<float, float>((a, b) => a / b, (a, b) => a / b);

            TestTwoTensor<float, float>((a, b) => a.Add(b), (a, b) => a + b);
            TestTwoTensor<float, float>((a, b) => a.Sub(b), (a, b) => a - b);
            TestTwoTensor<float, float>((a, b) => a.Mul(b), (a, b) => a * b);
            TestTwoTensor<float, float>((a, b) => a.Div(b), (a, b) => a / b);

            TestTwoTensorInPlace<float>((a, b) => a.AddInPlace(b), (a, b) => a + b);
            TestTwoTensorInPlace<float>((a, b) => a.SubInPlace(b), (a, b) => a - b);
            TestTwoTensorInPlace<float>((a, b) => a.MulInPlace(b), (a, b) => a * b);
            TestTwoTensorInPlace<float>((a, b) => a.DivInPlace(b), (a, b) => a / b);
        }

        [Fact]
        public void TestComparisonOperators()
        {
            // scalar-tensor operators
            TestOneTensor<float, bool>(a => a == 5.0f, a => a == 5.0f);
            TestOneTensor<float, bool>(a => a != 5.0f, a => a != 5.0f);
            TestOneTensorInPlace<float>(a => a.EqInPlace(5.0f), a => a == 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.NeInPlace(5.0f), a => a != 5.0f ? 1.0f : 0.0f);

            TestOneTensor<float, bool>(a => a < 5.0f, a => a < 5.0f);
            TestOneTensor<float, bool>(a => 5.0f < a, a => 5.0f < a);
            TestOneTensor<float, bool>(a => a <= 5.0f, a => a <= 5.0f);
            TestOneTensor<float, bool>(a => 5.0f <= a, a => 5.0f <= a);
            TestOneTensor<float, bool>(a => a > 5.0f, a => a > 5.0f);
            TestOneTensor<float, bool>(a => 5.0f > a, a => 5.0f > a);
            TestOneTensor<float, bool>(a => a >= 5.0f, a => a >= 5.0f);
            TestOneTensor<float, bool>(a => 5.0f >= a, a => 5.0f >= a);

            TestOneTensorInPlace<float>(a => a.LtInPlace(5.0f), a => a < 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.LeInPlace(5.0f), a => a <= 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.GtInPlace(5.0f), a => a > 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.GeInPlace(5.0f), a => a >= 5.0f ? 1.0f : 0.0f);

            TestOneTensor<float, float>(a => 5.0f % a, a => 5.0f % a);
            TestOneTensor<float, float>(a => a % 5.0f, a => a % 5.0f);
            TestOneTensorInPlace<float>(a => a.RemainderInPlace(5.0f), a => a % 5.0f);

            // tensor-tensor operators
            TestTwoTensor<float, bool>((a, b) => a == b, (a, b) => a == b);
            TestTwoTensor<float, bool>((a, b) => a != b, (a, b) => a != b);
            TestTwoTensorInPlace<float>((a, b) => a.EqInPlace(b), (a, b) => a == b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.NeInPlace(b), (a, b) => a != b ? 1.0f : 0.0f);

            TestTwoTensor<float, bool>((a, b) => a < b, (a, b) => a < b);
            TestTwoTensor<float, bool>((a, b) => a <= b, (a, b) => a <= b);
            TestTwoTensor<float, bool>((a, b) => a > b, (a, b) => a > b);
            TestTwoTensor<float, bool>((a, b) => a >= b, (a, b) => a >= b);

            TestTwoTensorInPlace<float>((a, b) => a.LtInPlace(b), (a, b) => a < b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.LeInPlace(b), (a, b) => a <= b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.GtInPlace(b), (a, b) => a > b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.GeInPlace(b), (a, b) => a >= b ? 1.0f : 0.0f);

            TestTwoTensor<float, float>((a, b) => a % b, (a, b) => a % b);
            TestTwoTensorInPlace<float>((a, b) => a.RemainderInPlace(b), (a, b) => a % b);
        }

        private void TestOneTensor<Tin, Tout>(Func<TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tout> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c2;
            var y = tensorFunc(x);

            var xData = x.Data<Tin>();
            var yData = y.Data<Tout>();

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(yData[i + j], scalarFunc(xData[i + j]));
                }
            }
        }

        private void TestOneTensorInPlace<Tin>(Func<TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tin> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c2;
            var xClone = x.Clone();
            var y = tensorFunc(x);

            var xData = x.Data<Tin>();
            var xCloneData = xClone.Data<Tin>();
            var yData = y.Data<Tin>();

            Assert.True(xData == yData);
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(yData[i + j], scalarFunc(xCloneData[i + j]));
                }
            }
        }

        private void TestTwoTensor<Tin, Tout>(Func<TorchTensor, TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tin, Tout> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Arange(10, 0, -1);
            var c3 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c3;
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            var xData = x.Data<Tin>();
            var yData = y.Data<Tin>();
            var zData = z.Data<Tout>();

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(zData[i + j], scalarFunc(xData[i + j], yData[i + j]));
                }
            }
        }

        private void TestTwoTensorInPlace<Tin>(Func<TorchTensor, TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tin, Tin> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Arange(10, 0, -1);
            var c3 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c3;
            var xClone = x.Clone();
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            var xData = x.Data<Tin>();
            var xCloneData = xClone.Data<Tin>();
            var yData = y.Data<Tin>();
            var zData = z.Data<Tin>();

            Assert.True(xData == zData);
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(zData[i + j], scalarFunc(xCloneData[i + j], yData[i + j]));
                }
            }
        }

        [Fact]
        public void TestMul()
        {
            var x = FloatTensor.Ones(new long[] { 100, 100 });

            var y = x.Mul(0.5f.ToScalar());

            var ydata = y.Data<float>();
            var xdata = x.Data<float>();

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.Equal(ydata[i + j], xdata[i + j] * 0.5f);
                }
            }
        }

        [Fact]
        public void SinTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sin).ToArray();
            var res = FloatTensor.From(data).Sin();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void CosTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cos).ToArray();
            var res = FloatTensor.From(data).Cos();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void TanTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tan).ToArray();
            var res = FloatTensor.From(data).Tan();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void SinhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sinh).ToArray();
            var res = FloatTensor.From(data).Sinh();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void CoshTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cosh).ToArray();
            var res = FloatTensor.From(data).Cosh();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void TanhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tanh).ToArray();
            var res = FloatTensor.From(data).Tanh();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void AsinTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Asin).ToArray();
            var res = FloatTensor.From(data).Asin();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void AcosTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Acos).ToArray();
            var res = FloatTensor.From(data).Acos();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void AtanTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Atan).ToArray();
            var res = FloatTensor.From(data).Atan();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void LogTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(x => MathF.Log(x)).ToArray();
            var res = FloatTensor.From(data).Log();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void Log10Test()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Log10).ToArray();
            var res = FloatTensor.From(data).Log10();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void FloorTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Floor).ToArray();
            var res = FloatTensor.From(data).Floor();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void CeilTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Ceiling).ToArray();
            var res = FloatTensor.From(data).Ceil();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void RoundTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(x => MathF.Round(x)).ToArray();
            var res = FloatTensor.From(data).Round();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void ExpandTest()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2 });
            TorchTensor onesExpanded = ones.Expand(new long[] { 3, 2 });

            Assert.Equal(onesExpanded.Shape, new long[] { 3, 2 });
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Assert.Equal(1.0, onesExpanded[i, j].DataItem<float>());
                }
            }
        }

        [Fact]
        public void TopKTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res1 = FloatTensor.From(data).TopK(1);
            var res1_0 = res1.values[0].DataItem<float>();
            var index1_0 = res1.indexes[0].DataItem<long>();
            Assert.Equal(3.1f, res1_0);
            Assert.Equal(2L, index1_0);

            var res2 = FloatTensor.From(data).TopK(2, sorted: true);
            var res2_0 = res2.values[0].DataItem<float>();
            var index2_0 = res2.indexes[0].DataItem<long>();
            var res2_1 = res2.values[1].DataItem<float>();
            var index2_1 = res2.indexes[1].DataItem<long>();
            Assert.Equal(3.1f, res2_0);
            Assert.Equal(2L, index2_0);
            Assert.Equal(2.0f, res2_1);
            Assert.Equal(1L, index2_1);
        }

        [Fact]
        public void SumTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };

            var res1 = FloatTensor.From(data).Sum();
            var res1_0 = res1.DataItem<float>();
            Assert.Equal(6.0f, res1_0);

            var res2 = FloatTensor.From(data).Sum(type: ScalarType.Double);
            var res2_0 = res2.DataItem<double>();
            Assert.Equal(6.0, res2_0);

            // summing integers gives long unless type is explicitly specified
            var dataInt32 = new int[] { 1, 2, 3 };
            var res3 = IntTensor.From(dataInt32).Sum();
            Assert.Equal(ScalarType.Long, res3.Type);
            var res3_0 = res3.DataItem<long>();
            Assert.Equal(6L, res3_0);

            // summing integers gives long unless type is explicitly specified
            var res4 = IntTensor.From(dataInt32).Sum(type: ScalarType.Int);
            Assert.Equal(ScalarType.Int, res4.Type);
            var res4_0 = res4.DataItem<int>();
            Assert.Equal(6L, res4_0);

        }

        [Fact]
        public void UnbindTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).Unbind();
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { }, res[0].Shape);
            Assert.Equal(new long[] { }, res[1].Shape);
            Assert.Equal(new long[] { }, res[2].Shape);
            Assert.Equal(1.1f, res[0].DataItem<float>());
            Assert.Equal(2.0f, res[1].DataItem<float>());
            Assert.Equal(3.1f, res[2].DataItem<float>());
        }

        [Fact]
        public void SplitWithSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).SplitWithSizes(new long[] { 2, 1 });
            Assert.Equal(2, res.Length);
            Assert.Equal(new long[] { 2 }, res[0].Shape);
            Assert.Equal(new long[] { 1 }, res[1].Shape);
            Assert.Equal(1.1f, res[0][0].DataItem<float>());
            Assert.Equal(2.0f, res[0][1].DataItem<float>());
            Assert.Equal(3.1f, res[1][0].DataItem<float>());
        }

        [Fact]
        public void RandomTest()
        {
            var res = FloatTensor.Random(new long[] { 2 });
            Assert.Equal(new long[] { 2 }, res.Shape);

            var res1 = ShortTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res1.Shape);

            var res2 = IntTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res2.Shape);

            var res3 = LongTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res3.Shape);

            var res4 = ByteTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res4.Shape);

            var res5 = SByteTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res5.Shape);

            var res6 = HalfTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res6.Shape);

            //var res7 = ComplexHalfTensor.RandomIntegers(10, new long[] { 200 });
            //Assert.Equal(new long[] { 200 }, res7.Shape);

            //var res8 = ComplexFloatTensor.RandomIntegers(10, new long[] { 200 });
            //Assert.Equal(new long[] { 200 }, res8.Shape);

            //var res9 = ComplexDoubleTensor.RandomIntegers(10, new long[] { 200 });
            //Assert.Equal(new long[] { 200 }, res9.Shape);
        }

        [Fact]
        public void SqueezeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).Expand(new long[] { 1, 1, 3 }).Squeeze(0).Squeeze(0);
            Assert.Equal(new long[] { 3 }, res.Shape);
            Assert.Equal(1.1f, res[0].DataItem<float>());
            Assert.Equal(2.0f, res[1].DataItem<float>());
            Assert.Equal(3.1f, res[2].DataItem<float>());
        }

        [Fact]
        public void NarrowTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).Narrow(0, 1, 2);
            Assert.Equal(new long[] { 2 }, res.Shape);
            Assert.Equal(2.0f, res[0].DataItem<float>());
            Assert.Equal(3.1f, res[1].DataItem<float>());
        }

        [Fact]
        public void SliceTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.0f };

            var res = FloatTensor.From(data).Slice(0, 1, 1, 1);
            Assert.Equal(new long[] { 0 }, res.Shape);

            var res2 = FloatTensor.From(data).Slice(0, 1, 2, 1);
            Assert.Equal(new long[] { 1 }, res2.Shape);
            Assert.Equal(2.0f, res2[0].DataItem<float>());

            var res3 = FloatTensor.From(data).Slice(0, 1, 4, 2);
            Assert.Equal(new long[] { 2 }, res3.Shape);
            Assert.Equal(2.0f, res3[0].DataItem<float>());
            Assert.Equal(4.0f, res3[1].DataItem<float>());
        }
        [Fact]
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
            var t1t = FloatTensor.From(t1raw, new long[] { 3, 4, 5 });
            var t2t = FloatTensor.From(t2raw, new long[] { 2, 4, 3 });
            var t3t = t1t.Conv1D(t2t);

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
                var data = t3t.Data<float>();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++)
                        {
                            var itemCorrect = t3Correct[i, j, k];
                            var item = data[i * 2 * 3 + j * 3 + k];
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }
            
            var t3p2d3 = t1t.Conv1D(t2t, padding: 2, dilation: 3);

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
                var data = t3p2d3.Data<float>();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++)
                        {
                            var itemCorrect = t3p2d3Correct[i, j, k];
                            var item = data[i * 2 * 3 + j * 3 + k];
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }
        }
    }
}
