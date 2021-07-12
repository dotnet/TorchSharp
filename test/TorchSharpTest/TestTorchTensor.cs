// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.TensorExtensionMethods;

using Xunit;

#nullable enable

namespace TorchSharp
{
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
        public void TestToString1()
        {
            Tensor t = Float32Tensor.zeros(2, 2);
            var str = t.ToString();
            Assert.Equal("[2x2], type = Float32, device = cpu", str);
        }

        [Fact]
        public void TestScalarToString()
        {
            {
                Tensor t = Float32Tensor.from(3.14f);
                var str = t.ToString(true);
                Assert.Equal("[], type = Float32, device = cpu, value = 3.14", str);
            }
            {
                Tensor t = Float32Tensor.from(3.14f);
                var str = t.ToString(true, "E2");
                Assert.Equal("[], type = Float32, device = cpu, value = 3.14E+000", str);
            }
            {
                Tensor t = ComplexFloat32Tensor.from(3.14f, 6.28f);
                var str = t.ToString(true);
                Assert.Equal("[], type = ComplexFloat32, device = cpu, value = 3.14+6.28i", str);
            }
        }

        private string _sep = OperatingSystem.IsWindows() ? "\r\n" : "\n";

        [Fact]
        public void Test1DToString()
        {
            {
                Tensor t = Float32Tensor.zeros(4);
                var str = t.ToString(true);
                Assert.Equal($"[4], type = Float32, device = cpu{_sep} 0 0 0 0{_sep}", str);
            }
            {
                Tensor t = ComplexFloat32Tensor.zeros(4);
                var str = t.ToString(true);
                Assert.Equal($"[4], type = ComplexFloat32, device = cpu{_sep} 0 0 0 0{_sep}", str);
            }
            {
                Tensor t = ComplexFloat32Tensor.ones(4);
                for (int i = 0; i < t.shape[0]; i++) t[i] = ComplexFloat32Tensor.from((1.0f * i, 2.43f * i * 2));
                var str = t.ToString(true);
                Assert.Equal($"[4], type = ComplexFloat32, device = cpu{_sep} 0 1+4.86i 2+9.72i 3+14.58i{_sep}", str);
            }
        }

        [Fact]
        public void Test2DToString()
        {
            {
                Tensor t = Float32Tensor.from(new float[] { 0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f }, 2, 4);
                var str = t.ToString(true);
                Assert.Equal($"[2x4], type = Float32, device = cpu{_sep}{_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}", str);
            }
            {
                Tensor t = ComplexFloat32Tensor.zeros(2, 4);
                var str = t.ToString(true);
                Assert.Equal($"[2x4], type = ComplexFloat32, device = cpu{_sep}{_sep} 0 0 0 0{_sep} 0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        public void Test3DToString()
        {
            {
                Tensor t = Float32Tensor.from(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, 2, 2, 4);
                var str = t.ToString(true, "0.0000000");
                Assert.Equal($"[2x2x4], type = Float32, device = cpu{_sep}{_sep}[0,:,:] ={_sep} 0.0000000   3.1410000 6.2834000    3.1415200{_sep}" +
                             $" 0.0000063 -13.1415300 0.0100000 4713.1400000{_sep}{_sep}[1,:,:] ={_sep} 0.0100000 0.0000000 0.0000000 0.0000000{_sep}" +
                             $" 0.0000000 0.0000000 0.0000000 0.0000000{_sep}", str);
            }
        }

        [Fact]
        public void Test4DToString()
        {
            {
                Tensor t = Float32Tensor.from(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, 2, 2, 2, 4);
                var str = t.ToString(true);
                Assert.Equal($"[2x2x2x4], type = Float32, device = cpu{_sep}{_sep}[0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}" +
                             $"[1,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[1,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        public void Test5DToString()
        {
            {
                Tensor t = Float32Tensor.from(new float[] {
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 3.141f, 6.2834f, 3.14152f, 6.28e-06f, -13.141529f, 0.01f, 4713.14f,
                        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    }, new long[] { 2, 2, 2, 2, 4 });
                var str = t.ToString(true);
                Assert.Equal($"[2x2x2x2x4], type = Float32, device = cpu{_sep}{_sep}[0,0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,0,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[0,1,0,:,:] ={_sep}" +
                             $"        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,1,1,:,:] ={_sep} 0.01 0 0 0{_sep}   " +
                             $" 0 0 0 0{_sep}{_sep}[1,0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[1,0,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,1,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,1,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        public void Test6DToString()
        {
            {
                Tensor t = Float32Tensor.from(new float[] {
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
                var str = t.ToString(true);
                Assert.Equal($"[2x2x2x2x2x4], type = Float32, device = cpu{_sep}{_sep}[0,0,0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,0,0,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[0,0,1,0,:,:] ={_sep}" +
                             $"        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,0,1,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}" +
                             $"[0,1,0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[0,1,0,1,:,:] ={_sep} 0.01 0 0 0{_sep}" +
                             $"    0 0 0 0{_sep}{_sep}[0,1,1,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[0,1,1,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,0,0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,0,0,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,0,1,0,:,:] ={_sep}" +
                             $"        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,0,1,1,:,:] ={_sep} 0.01 0 0 0{_sep}" +
                             $"    0 0 0 0{_sep}{_sep}[1,1,0,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep} 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}" +
                             $"[1,1,0,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}{_sep}[1,1,1,0,:,:] ={_sep}        0   3.141 6.2834 3.1415{_sep}" +
                             $" 6.28e-06 -13.142   0.01 4713.1{_sep}{_sep}[1,1,1,1,:,:] ={_sep} 0.01 0 0 0{_sep}    0 0 0 0{_sep}", str);
            }
        }

        [Fact]
        public void CreateFloat32TensorZeros()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Float32Tensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0.0f, t[0, 0].ToSingle());
            Assert.Equal(0.0f, t[1, 1].ToSingle());
        }

        [Fact]
        public void CreateFloat32TensorZeros_()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0.0f, t[0, 0].ToSingle());
            Assert.Equal(0.0f, t[1, 1].ToSingle());
        }

        [Fact]
        public void CreateByteTensorZeros()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = ByteTensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal((byte)0, t[0, 0].ToByte());
            Assert.Equal((byte)0, t[1, 1].ToByte());
        }

        [Fact]
        public void CreateByteTensorZeros_()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, torch.uint8);
            Assert.Equal(shape, t.shape);
            Assert.Equal((byte)0, t[0, 0].ToByte());
            Assert.Equal((byte)0, t[1, 1].ToByte());
        }

        [Fact]
        public void CreateInt32TensorZeros()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Int32Tensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0, t[0, 0].ToInt32());
            Assert.Equal(0, t[1, 1].ToInt32());
        }

        [Fact]
        public void CreateInt32TensorZeros_()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = torch.zeros(shape, torch.int32);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0, t[0, 0].ToInt32());
            Assert.Equal(0, t[1, 1].ToInt32());
        }

        [Fact]
        public void CreateInt64TensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = Int64Tensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0L, t[0, 0].ToInt64());
            Assert.Equal(0L, t[1, 1].ToInt64());
        }

        [Fact]
        public void CreateInt64TensorZeros_()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = torch.zeros(shape, torch.int64);
            Assert.Equal(shape, t.shape);
            Assert.Equal(0L, t[0, 0].ToInt64());
            Assert.Equal(0L, t[1, 1].ToInt64());
        }

        [Fact]
        public void CreateBoolTensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = BoolTensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal((object)false, t[0, 0].ToBoolean());
            Assert.Equal((object)false, t[1, 1].ToBoolean());
        }

        [Fact]
        public void CreateFloat16TensorZeros()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = Float16Tensor.zeros(shape, device: device);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(0.0f, t[0, 0].ToSingle());
                    Assert.Equal(0.0f, t[1, 1].ToSingle());
                }
            }
        }

        [Fact]
        public void CreateBFloat16TensorZeros()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = BFloat16Tensor.zeros(shape, device: device);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(0.0f, t[0, 0].ToSingle());
                    Assert.Equal(0.0f, t[1, 1].ToSingle());
                }
            }
        }

        [Fact]
        public void CreateFloat32TensorEmpty()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Float32Tensor.empty(shape);
            Assert.Equal(shape, t.shape);
            t = torch.empty(shape);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        public void CreateFloat32TensorFull()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Float32Tensor.full(2, 2, 3.14f);
            Assert.Equal(shape, t.shape);
            Assert.Equal(3.14f, t[0, 0].ToSingle());
            Assert.Equal(3.14f, t[1, 1].ToSingle());
            t = torch.full(2, 2, 3.14f);
            Assert.Equal(shape, t.shape);
            Assert.Equal(3.14f, t[0, 0].ToSingle());
            Assert.Equal(3.14f, t[1, 1].ToSingle());
        }

        [Fact]
        public void CreateByteTensorEmpty()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = ByteTensor.empty(shape);
            Assert.Equal(shape, t.shape);
            t = torch.empty(shape, torch.uint8);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        public void CreateInt32TensorEmpty()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Int32Tensor.empty(shape);
            Assert.Equal(shape, t.shape);
            t = torch.empty(shape, torch.int32);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        public void CreateInt32TensorFull()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Int32Tensor.full(2, 2, 17);
            Assert.Equal(shape, t.shape);
            Assert.Equal(17, t[0, 0].ToInt32());
            Assert.Equal(17, t[1, 1].ToInt32());
            t = torch.full(2, 2, 17, torch.int32);
            Assert.Equal(shape, t.shape);
            Assert.Equal(17, t[0, 0].ToInt32());
            Assert.Equal(17, t[1, 1].ToInt32());
        }

        [Fact]
        public void CreateInt64TensorEmpty()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = Int64Tensor.empty(shape);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        public void CreateBoolTensorEmpty()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = BoolTensor.empty(shape);
            Assert.Equal(shape, t.shape);
        }

        [Fact]
        public void CreateTensorEmptyStrided()
        {
            var shape = new long[] { 2, 3 };
            var strides = new long[] { 1, 2 };

            Tensor t = Float32Tensor.empty_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
            t = torch.empty_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
        }

        [Fact]
        public void CreateTensorEmptyAsStrided()
        {
            var shape = new long[] { 2, 3 };
            var strides = new long[] { 1, 2 };

            Tensor t = Float32Tensor.empty(shape).as_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
            t = torch.empty(shape).as_strided(shape, strides);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t.stride(0));
            Assert.Equal(2, t.stride(1));
        }

        [Fact]
        public void CreateFloat16TensorEmpty()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = Float16Tensor.empty(shape, device: device);
                    Assert.Equal(shape, t.shape);
                }
            }
        }

        [Fact]
        public void CreateBFloat16TensorEmpty()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = BFloat16Tensor.empty(shape, device: device);
                    Assert.Equal(shape, t.shape);
                }
            }
        }

        [Fact]
        public void CreateFloat32Linspace()
        {
            Tensor t = Float32Tensor.linspace(0.0f, 10.0f, 101);
            Assert.Equal(101, t.shape[0]);
            Assert.Equal(0.0f, t[0].ToSingle());
            Assert.Equal(0.1f, t[1].ToSingle());
        }

        [Fact]
        public void CreateFloat32Logspace()
        {
            Tensor t = Float32Tensor.logspace(0.0f, 10.0f, 101);
            Assert.Equal(101, t.shape[0]);
            Assert.Equal(1.0f, t[0].ToSingle());
            Assert.Equal(10.0f, t[10].ToSingle());
        }

        [Fact]
        public void CreateFloat32TensorOnes()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Float32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 1].ToSingle());
        }

        [Fact]
        public void CreateByteTensorOnes()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = ByteTensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal((byte)1, t[0, 0].ToByte());
            Assert.Equal((byte)1, t[1, 1].ToByte());
        }

        [Fact]
        public void CreateInt32TensorOnes()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = Int32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1, t[0, 0].ToInt32());
            Assert.Equal(1, t[1, 1].ToInt32());
        }

        [Fact]
        public void CreateInt64TensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = Int64Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1L, t[0, 0].ToInt64());
            Assert.Equal(1L, t[1, 1].ToInt64());
        }

        [Fact]
        public void CreateBoolTensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = BoolTensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal((object)true, t[0, 0].ToBoolean());
            Assert.Equal((object)true, t[1, 1].ToBoolean());
        }

        [Fact]
        public void CreateFloat16TensorOnes()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = Float16Tensor.ones(shape, device: device);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(1.0f, t[0, 0].ToSingle());
                    Assert.Equal(1.0f, t[1, 1].ToSingle());
                }
            }
        }

        [Fact]
        public void CreateBFloat16TensorOnes()
        {
            foreach (var device in new Device[] { torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var shape = new long[] { 2, 2 };

                    Tensor t = BFloat16Tensor.ones(shape, device: device);
                    Assert.Equal(shape, t.shape);
                    Assert.Equal(1.0f, t[0, 0].ToSingle());
                    Assert.Equal(1.0f, t[1, 1].ToSingle());
                }
            }
        }

        [Fact]
        public void CreateComplexFloat32TensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = ComplexFloat32Tensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            var v3 = t.Data<(float Real, float Imaginary)>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(0.0f, v3[i].Real);
                Assert.Equal(0.0f, v3[i].Imaginary);
            }
        }

        [Fact]
        public void CreateComplexFloat32TensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = ComplexFloat32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            var v3 = t.Data<(float Real, float Imaginary)>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(1.0f, v3[i].Real);
                Assert.Equal(0.0f, v3[i].Imaginary);
            }
        }

        [Fact]
        public void CreateComplex32()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = Float32Tensor.randn(shape);
            Tensor i = Float32Tensor.randn(shape);

            Tensor x = torch.complex(r, i);

            Assert.Equal(shape, x.shape);
            Assert.Equal(ScalarType.ComplexFloat32, x.dtype);
            Assert.True(r.allclose(x.Real));
            Assert.True(i.allclose(x.Imag));
        }

        [Fact]
        public void CreateComplex64()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = Float64Tensor.randn(shape);
            Tensor i = Float64Tensor.randn(shape);

            Tensor x = torch.complex(r, i);

            Assert.Equal(shape, x.shape);
            Assert.Equal(ScalarType.ComplexFloat64, x.dtype);
            Assert.True(r.allclose(x.Real));
            Assert.True(i.allclose(x.Imag));
        }

        [Fact]
        public void CreatePolar32()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = Float32Tensor.randn(shape);
            Tensor i = Float32Tensor.randn(shape);

            Tensor x = torch.complex(r, i);

            Tensor p = torch.polar(x.abs(), x.angle());

            Assert.Equal(x.shape, p.shape);
            Assert.Equal(ScalarType.ComplexFloat32, p.dtype);
            Assert.True(r.allclose(p.Real, rtol: 1e-04, atol: 1e-07));
            Assert.True(i.allclose(p.Imag, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void CreatePolar64()
        {
            var shape = new long[] { 2, 2 };
            Tensor r = Float64Tensor.randn(shape);
            Tensor i = Float64Tensor.randn(shape);

            Tensor x = torch.complex(r, i);

            Tensor p = torch.polar(x.abs(), x.angle());

            Assert.Equal(x.shape, p.shape);
            Assert.Equal(ScalarType.ComplexFloat64, p.dtype);
            Assert.True(r.allclose(p.Real, rtol: 1e-04, atol: 1e-07));
            Assert.True(i.allclose(p.Imag, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void CreateComplexFloat32TensorRand()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = ComplexFloat32Tensor.rand(shape);
            Assert.Equal(shape, t.shape);
            var v3 = t.Data<(float Real, float Imaginary)>().ToArray();
            Assert.All(v3, t => Assert.True(t.Real >= 0.0f && t.Real < 1.0f && t.Imaginary >= 0.0f && t.Imaginary < 1.0f));
        }

        [Fact]
        public void CreateComplexFloat32TensorRandn()
        {
            var shape = new long[] { 2, 2 };
            Tensor t = ComplexFloat32Tensor.randn(shape);
            Assert.Equal(shape, t.shape);
            var v3 = t.Data<(float Real, float Imaginary)>().ToArray();
        }

        [Fact]
        public void CreateComplexFloat64TensorZeros()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = ComplexFloat64Tensor.zeros(shape);
            Assert.Equal(shape, t.shape);
            var v3 = t.Data<System.Numerics.Complex>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(0.0, v3[i].Real);
                Assert.Equal(0.0, v3[i].Imaginary);
            }
        }

        [Fact]
        public void CreateComplexFloat64TensorOnes()
        {
            var shape = new long[] { 2, 2 };

            Tensor t = ComplexFloat64Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            var v3 = t.Data<System.Numerics.Complex>().ToArray();
            for (var i = 0; i < v3.Length; i++) {
                Assert.Equal(1.0, v3[i].Real);
                Assert.Equal(0.0, v3[i].Imaginary);
            }
        }

        [Fact]
        public void CreateFloat32TensorCheckMemory()
        {
            Tensor? ones = null;

            for (int i = 0; i < 10; i++) {
                using (var tmp = Float32Tensor.ones(new long[] { 100, 100, 100 })) {
                    ones = tmp;
                    Assert.NotNull(ones);
                }
            }
        }

        [Fact]
        public void CreateFloat32TensorOnesCheckData()
        {
            var ones = Float32Tensor.ones(new long[] { 2, 2 });
            var data = ones.Data<float>();

            for (int i = 0; i < 4; i++) {
                Assert.Equal(1.0, data[i]);
            }
        }

        [Fact]
        public void CreateFloat32TensorZerosCheckData()
        {
            var zeros = Float32Tensor.zeros(new long[] { 2, 2 });
            var data = zeros.Data<float>();

            for (int i = 0; i < 4; i++) {
                Assert.Equal(0, data[i]);
            }
        }

        [Fact]
        public void CreateInt32TensorOnesCheckData()
        {
            var ones = Int32Tensor.ones(new long[] { 2, 2 });
            var data = ones.Data<int>();

            for (int i = 0; i < 4; i++) {
                Assert.Equal(1, data[i]);
            }
        }

        [Fact]
        public void CreateInt32TensorEyeCheckData1()
        {
            var ones = Int32Tensor.eye(4, 4);
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
        public void CreateInt32TensorEyeCheckData2()
        {
            var ones = Int32Tensor.eye(4);
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
        public void CreateComplexFloat32TensorEyeCheckData()
        {
            var ones = ComplexFloat32Tensor.eye(4, 4);
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
        public void CreateComplexFloat64TensorEyeCheckData()
        {
            var ones = ComplexFloat64Tensor.eye(4, 4);
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
        public void CreateFloat32TensorCheckDevice()
        {
            var ones = Float32Tensor.ones(new long[] { 2, 2 });
            var device = ones.device;

            Assert.Equal("cpu", ones.device.ToString());
        }

        [Fact]
        public void CreateFloat32TensorFromData()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = Float32Tensor.from(data, new long[] { 100, 10 })) {
                Assert.Equal(1, tensor.Data<float>()[100]);
            }
        }

        [Fact]
        public void CreateFloat32TensorFromDataCheckDispose()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = Float32Tensor.from(data, new long[] { 100, 10 })) {
                Assert.Equal(1, tensor.Data<float>()[100]);
            }

            Assert.Equal(1, data[100]);
        }

        [Fact]
        public void CreateFloat32TensorFromData2()
        {
            var data = new float[1000];

            using (var tensor = data.ToTensor(new long[] { 10, 100 })) {
                Assert.Equal(default(float), tensor.Data<float>()[100]);
            }
        }

        [Fact]
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
        public void CreateFloat16TensorFromDataCheckStrides()
        {
            var data = new float[] { 0.2663158f, 0.1144736f, 0.1147367f, 0.1249998f, 0.1957895f, 0.1231576f, 0.1944732f, 0.111842f, 0.1065789f, 0.667881f, 0.5682123f, 0.5824502f, 0.4824504f, 0.4844371f, 0.6463582f, 0.5334439f, 0.5079474f, 0.2281452f };
            var dataTensor = Float16Tensor.from(data, new long[] { 2, 9 });

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
        public void CreateBFloat16TensorFromDataCheckStrides()
        {
            var data = new float[] { 0.2663158f, 0.1144736f, 0.1147367f, 0.1249998f, 0.1957895f, 0.1231576f, 0.1944732f, 0.111842f, 0.1065789f, 0.667881f, 0.5682123f, 0.5824502f, 0.4824504f, 0.4844371f, 0.6463582f, 0.5334439f, 0.5079474f, 0.2281452f };
            var dataTensor = BFloat16Tensor.from(data, new long[] { 2, 9 });

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
        public void CreateFloat32BartlettWindow()
        {
            Tensor t = Float32Tensor.bartlett_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        public void CreateFloat32BlackmanWindow()
        {
            Tensor t = Float32Tensor.blackman_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        public void CreateFloat32HammingWindow()
        {
            Tensor t = Float32Tensor.hamming_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        public void CreateFloat32HannWindow()
        {
            Tensor t = Float32Tensor.hann_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        public void CreateFloat32KaiserWindow()
        {
            Tensor t = Float32Tensor.kaiser_window(100);
            Assert.Equal(100, t.shape[0]);
        }

        [Fact]
        public void CreateFloat32TensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = Float32Tensor.from(scalar)) {
                Assert.Equal(333.0f, tensor.ToSingle());
            }
        }

        [Fact]
        public void CreateFloat16TensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = Float16Tensor.from(scalar)) {
                Assert.Equal(333.0f, tensor.ToSingle());
            }
        }

        [Fact]
        public void CreateBFloat16TensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = BFloat16Tensor.from(scalar)) {
                Assert.Equal(332.0f, tensor.ToSingle()); // NOTE: bfloat16 loses precision, this really is 332.0f
            }
        }

        [Fact]
        public void CreateFloat32TensorFromScalar2()
        {
            float scalar = 333.0f;

            using (var tensor = scalar.ToTensor()) {
                Assert.Equal(333, tensor.ToSingle());
            }
        }

        [Fact]
        public void CreateFloat32TensorOnesLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.ones_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.Data<float>().ToArray(), t => Assert.True(t == 1.0f));
        }

        [Fact]
        public void CreateFloat32TensorOnesLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.ones_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.Data<double>().ToArray(), t => Assert.True(t == 1.0));
        }

        [Fact]
        public void CreateFloat32TensorZerosLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.zeros_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.Data<float>().ToArray(), t => Assert.True(t == 0.0f));
        }

        [Fact]
        public void CreateFloat32TensorZerosLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.zeros_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.Data<double>().ToArray(), t => Assert.True(t == 0.0));
        }

        [Fact]
        public void CreateFloat32TensorEmptyLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.empty_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
        }

        [Fact]
        public void CreateFloat32TensorEmptyLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.empty_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
        }

        [Fact]
        public void CreateFloat32TensorFullLike()
        {
            var shape = new long[] { 10, 20, 30 };
            Scalar value = 3.14f;

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.full_like(value);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.Data<float>().ToArray(), t => Assert.True(t == 3.14f));
        }

        [Fact]
        public void CreateFloat32TensorFullLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };
            Scalar value = 3.14;

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.full_like(value, dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.Data<double>().ToArray(), t => Assert.True(t == 3.14));
        }

        [Fact]
        public void CreateFloat32TensorRandLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.rand_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.Data<float>().ToArray(), t => Assert.True(t >= 0.0f && t < 1.0f));
        }

        [Fact]
        public void CreateFloat32TensorRandLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.rand_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.Data<double>().ToArray(), t => Assert.True(t >= 0.0 && t < 1.0));
        }

        [Fact]
        public void CreateComplexFloat32TensorRandLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = ComplexFloat32Tensor.empty(shape);
            Tensor t2 = t1.rand_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        public void CreateComplexFloat32TensorRandLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.rand_like(dtype: ScalarType.ComplexFloat32);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        public void CreateFloat32TensorRandnLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.randn_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
        }

        [Fact]
        public void CreateFloat32TensorRandnLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.randn_like(dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
        }

        [Fact]
        public void CreateComplexFloat32TensorRandnLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = ComplexFloat32Tensor.empty(shape);
            Tensor t2 = t1.randn_like();

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        public void CreateComplexFloat32TensorRandnLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.randn_like(dtype: ScalarType.ComplexFloat32);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.ComplexFloat32, t2.dtype);
        }

        [Fact]
        public void CreateFloat32TensorRandintLike()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.randint_like(5, 15);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float32, t2.dtype);
            Assert.All(t2.Data<float>().ToArray(), t => Assert.True(t >= 5.0f && t < 15.0f));
        }

        [Fact]
        public void CreateFloat32TensorRandintLikeWithType()
        {
            var shape = new long[] { 10, 20, 30 };

            Tensor t1 = Float32Tensor.empty(shape);
            Tensor t2 = t1.randint_like(5, 15, dtype: ScalarType.Float64);

            Assert.Equal(shape, t1.shape);
            Assert.Equal(ScalarType.Float32, t1.dtype);
            Assert.Equal(t1.shape, t2.shape);
            Assert.Equal(ScalarType.Float64, t2.dtype);
            Assert.All(t2.Data<double>().ToArray(), t => Assert.True(t >= 5.0 && t < 15.0));
        }

        [Fact]
        public void Float32Mean()
        {
            using (var tensor = Float32Tensor.arange(1, 100)) {
                var mean = tensor.mean().DataItem<float>();
                Assert.Equal(50.0f, mean);
            }
        }


        [Fact]
        public void Int32Mode()
        {
            using (var tensor = Int32Tensor.from(new int[] { 1, 5, 4, 5, 3, 3, 5, 5 })) {
                var mode = tensor.mode();
                Assert.Equal(new int[] { 5 }, mode.values.Data<int>().ToArray());
                Assert.Equal(new long[] { 7 }, mode.indices.Data<long>().ToArray());
            }
        }

        [Fact]
        public void GetSetItem2()
        {
            var shape = new long[] { 2, 3 };
            Tensor t = Float32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2].ToSingle());
            t[1, 2] = Float32Tensor.from(2.0f);
            Assert.Equal(2.0f, t[1, 2].ToSingle());
        }

        [Fact]
        public void GetSetItemComplexFloat2()
        {
            var shape = new long[] { 2, 3 };
            Tensor t = ComplexFloat32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToComplex32().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex32().Real);
            Assert.Equal(1.0f, t[1, 2].ToComplex32().Real);

            t[1, 1] = ComplexFloat32Tensor.from(0.5f, 1.0f);
            Assert.Equal(0.5f, t[1, 1].ToComplex32().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex32().Imaginary);

            t[1, 2] = ComplexFloat32Tensor.from((2.0f, 3.0f));
            Assert.Equal(2.0f, t[1, 2].ToComplex32().Real);
            Assert.Equal(3.0f, t[1, 2].ToComplex32().Imaginary);
        }

        [Fact]
        public void GetSetItemComplexDouble2()
        {
            var shape = new long[] { 2, 3 };
            Tensor t = ComplexFloat64Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0].ToComplex64().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex64().Real);
            Assert.Equal(1.0f, t[1, 2].ToComplex64().Real);

            t[1, 1] = ComplexFloat64Tensor.from(0.5, 1.0);
            Assert.Equal(0.5f, t[1, 1].ToComplex64().Real);
            Assert.Equal(1.0f, t[1, 1].ToComplex64().Imaginary);

            t[1, 2] = ComplexFloat64Tensor.from(new System.Numerics.Complex(2.0, 3.0f));
            Assert.Equal(2.0f, t[1, 2].ToComplex64().Real);
            Assert.Equal(3.0f, t[1, 2].ToComplex64().Imaginary);
        }

        [Fact]
        public void GetSetItem3()
        {
            var shape = new long[] { 2, 3, 4 };
            Tensor t = Float32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3].ToSingle());
            t[1, 2, 3] = Float32Tensor.from(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3].ToSingle());
        }

        [Fact]
        public void GetSetItem4()
        {
            var shape = new long[] { 2, 3, 4, 5 };
            Tensor t = Float32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3, 4].ToSingle());
            t[1, 2, 3, 4] = Float32Tensor.from(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3, 4].ToSingle());
        }

        [Fact]
        public void GetSetItem5()
        {
            var shape = new long[] { 2, 3, 4, 5, 6 };
            Tensor t = Float32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3, 4, 5].ToSingle());
            t[1, 2, 3, 4, 5] = Float32Tensor.from(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3, 4, 5].ToSingle());
        }


        [Fact]
        public void GetSetItem6()
        {
            var shape = new long[] { 2, 3, 4, 5, 6, 7 };
            Tensor t = Float32Tensor.ones(shape);
            Assert.Equal(shape, t.shape);
            Assert.Equal(1.0f, t[0, 0, 0, 0, 0, 0].ToSingle());
            Assert.Equal(1.0f, t[1, 2, 3, 4, 5, 6].ToSingle());
            t[1, 2, 3, 4, 5, 6] = Float32Tensor.from(2.0f);
            Assert.Equal(2.0f, t[1, 2, 3, 4, 5, 6].ToSingle());
        }

        [Fact]
        public void TestScalarToTensor()
        {
            Assert.Throws<ArgumentException>(() => 1.ToTensor(requiresGrad: true));
        }

        [Fact]
        public void TestScalarToTensor2()
        {
            using (var tensor = 1.ToTensor()) {
                Assert.Equal(ScalarType.Int32, tensor.dtype);
                Assert.Equal(1, tensor.ToInt32());
            }
            using (var tensor = ((byte)1).ToTensor()) {
                Assert.Equal(ScalarType.Byte, tensor.dtype);
                Assert.Equal(1, tensor.ToByte());
            }
            using (var tensor = ((sbyte)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int8, tensor.dtype);
                Assert.Equal(-1, tensor.ToSByte());
            }
            using (var tensor = ((short)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int16, tensor.dtype);
                Assert.Equal(-1, tensor.ToInt16());
            }
            using (var tensor = ((long)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int64, tensor.dtype);
                Assert.Equal(-1L, tensor.ToInt64());
            }
            using (var tensor = ((float)-1).ToTensor()) {
                Assert.Equal(ScalarType.Float32, tensor.dtype);
                Assert.Equal(-1.0f, tensor.ToSingle());
            }
            using (var tensor = ((double)-1).ToTensor()) {
                Assert.Equal(ScalarType.Float64, tensor.dtype);
                Assert.Equal(-1.0, tensor.ToDouble());
            }
        }

        [Fact]
        public void TestScalarToTensor3()
        {
            using (var tensor = 1.ToTensor()) {
                Assert.Equal(ScalarType.Int32, tensor.dtype);
                Assert.Equal(1, (int)tensor);
            }
            using (var tensor = ((byte)1).ToTensor()) {
                Assert.Equal(ScalarType.Byte, tensor.dtype);
                Assert.Equal(1, (byte)tensor);
            }
            using (var tensor = ((sbyte)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int8, tensor.dtype);
                Assert.Equal(-1, (sbyte)tensor);
            }
            using (var tensor = ((short)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int16, tensor.dtype);
                Assert.Equal(-1, (short)tensor);
            }
            using (var tensor = ((long)-1).ToTensor()) {
                Assert.Equal(ScalarType.Int64, tensor.dtype);
                Assert.Equal(-1L, (long)tensor);
            }
            using (var tensor = ((float)-1).ToTensor()) {
                Assert.Equal(ScalarType.Float32, tensor.dtype);
                Assert.Equal(-1.0f, (float)tensor);
            }
            using (var tensor = ((double)-1).ToTensor()) {
                Assert.Equal(ScalarType.Float64, tensor.dtype);
                Assert.Equal(-1.0, (double)tensor);
            }
        }

        [Fact]
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
        public void InfinityTest()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                tensor.fill_(Single.PositiveInfinity);
                Assert.True(tensor.isposinf().Data<bool>().ToArray().All(b => b));
                Assert.True(tensor.isinf().Data<bool>().ToArray().All(b => b));
                Assert.False(tensor.isneginf().Data<bool>().ToArray().All(b => b));
                Assert.False(tensor.isfinite().Data<bool>().ToArray().All(b => b));

                tensor.fill_(Single.NegativeInfinity);
                Assert.True(tensor.isneginf().Data<bool>().ToArray().All(b => b));
                Assert.True(tensor.isinf().Data<bool>().ToArray().All(b => b));
                Assert.False(tensor.isposinf().Data<bool>().ToArray().All(b => b));
                Assert.False(tensor.isfinite().Data<bool>().ToArray().All(b => b));
            }
        }

        [Fact]
        public void TorchNormal()
        {
            using (var tensor = torch.bernoulli(torch.rand(5))) {
                Assert.Equal(5, tensor.shape[0]);
            }
        }

        [Fact]
        public void TorchMultinomial()
        {
            using (var tensor = torch.multinomial(torch.rand(5), 17, true)) {
                Assert.Equal(17, tensor.shape[0]);
            }
        }

        [Fact]
        public void TorchPoisson()
        {
            using (var tensor = torch.poisson(torch.rand(5))) {
                Assert.Equal(5, tensor.shape[0]);
            }
        }

        [Fact]
        public void InitZeros()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.zeros(tensor)) { }
            }
        }

        [Fact]
        public void InitOnes()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.ones(tensor)) { }
            }
        }

        [Fact]
        public void InitDirac()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.dirac(tensor)) { }
            }
        }

        [Fact]
        public void InitEye()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.eye(tensor)) { }
            }
        }

        [Fact]
        public void InitConstant()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.constant(tensor, Math.PI)) { }
            }
        }

        [Fact]
        public void InitUniform()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.uniform(tensor)) { }
            }
        }

        [Fact]
        public void InitNormal()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.normal(tensor)) { }
            }
        }

        [Fact]
        public void InitOrthogonal()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.orthogonal(tensor)) { }
            }
        }

        [Fact]
        public void InitSparse()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.sparse(tensor, 0.25)) { }
            }
        }

        [Fact]
        public void InitKaimingUniform()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.kaiming_uniform(tensor)) { }
            }
        }

        [Fact]
        public void InitKaimingNormal()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.kaiming_normal(tensor)) { }
            }
        }

        [Fact]
        public void InitXavierUniform()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.xavier_uniform(tensor)) { }
            }
        }

        [Fact]
        public void InitXavierNormal()
        {
            using (Tensor tensor = Float32Tensor.zeros(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = torch.nn.init.xavier_normal(tensor)) { }
            }
        }

        [Fact]
        public void InplaceBernoulli()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.bernoulli_()) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.bernoulli_(generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceCauchy()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.cauchy_()) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.cauchy_(generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceExponential()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.exponential_()) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.exponential_(generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceGeometric()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.geometric_(0.25)) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.geometric_(0.25, generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceLogNormal()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.log_normal_()) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.log_normal_(generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceNormal()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.normal_()) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.normal_(generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceRandom()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.random_(0.0, 1.0)) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.random_(0, 1, generator: gen)) { }
            }
        }

        [Fact]
        public void InplaceUniform()
        {
            using (Tensor tensor = Float32Tensor.empty(new long[] { 2, 2 })) {
                // Really just testing that the native interop works and that the fact
                // that there are two handles to the tensor is okay.
                using (var res = tensor.uniform_(0.0, 1.0)) { }
                using (var gen = new torch.Generator(4711L))
                using (var res = tensor.uniform_(0, 1, generator: gen)) { }
            }
        }

        [Fact]
        public void TestSparse()
        {
            using (var i = Int64Tensor.from(new long[] { 0, 1, 1, 2, 0, 2 }, new long[] { 2, 3 }))
            using (var v = Float32Tensor.from(new float[] { 3, 4, 5 }, new long[] { 3 })) {
                var sparse = Float32Tensor.sparse(i, v, new long[] { 2, 3 });

                Assert.True(sparse.IsSparse);
                Assert.False(i.IsSparse);
                Assert.False(v.IsSparse);
                Assert.Equal(sparse.SparseIndices.Data<long>().ToArray(), new long[] { 0, 1, 1, 2, 0, 2 });
                Assert.Equal(sparse.SparseValues.Data<float>().ToArray(), new float[] { 3, 4, 5 });
            }
        }

        [Fact]
        public void TestIndexSingle()
        {
            using (var i = Int64Tensor.from(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 })) {
                Assert.Equal(0, i.index(new TensorIndex[] { TensorIndex.Single(0), TensorIndex.Single(0) }).ToInt32());
                Assert.Equal(1, i.index(new TensorIndex[] { TensorIndex.Single(0), TensorIndex.Single(1) }).ToInt32());
                Assert.Equal(2, i.index(new TensorIndex[] { TensorIndex.Single(0), TensorIndex.Single(2) }).ToInt32());
                Assert.Equal(6, i.index(new TensorIndex[] { TensorIndex.Single(1), TensorIndex.Single(0) }).ToInt32());
                Assert.Equal(5, i.index(new TensorIndex[] { TensorIndex.Single(1), TensorIndex.Single(1) }).ToInt32());
                Assert.Equal(4, i.index(new TensorIndex[] { TensorIndex.Single(1), TensorIndex.Single(2) }).ToInt32());
            }
        }

        [Fact]
        public void TestIndexEllipsis()
        {
            using (var i = Int64Tensor.from(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 })) {
                var t1 = i.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Single(0) });
                Assert.Equal(0, t1[0].ToInt32());
                Assert.Equal(6, t1[1].ToInt32());
            }
        }

        [Fact]
        public void TestIndexNull()
        {
            using (var i = Int64Tensor.from(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 })) {
                var t1 = i.index(new TensorIndex[] { TensorIndex.None, TensorIndex.Single(0) });
                Assert.Equal(0, t1[0, 0].ToInt32());
                Assert.Equal(1, t1[0, 1].ToInt32());
                Assert.Equal(2, t1[0, 2].ToInt32());
            }
        }

        [Fact]
        public void TestIndexNone()
        {
            using (var i = Int64Tensor.from(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 })) {
                var t1 = i.index(new TensorIndex[] { TensorIndex.None, TensorIndex.Single(0) });
                Assert.Equal(0, t1[0, 0].ToInt32());
                Assert.Equal(1, t1[0, 1].ToInt32());
                Assert.Equal(2, t1[0, 2].ToInt32());
            }
        }

        [Fact]
        public void TestIndexSlice()
        {
            using (var i = Int64Tensor.from(new long[] { 0, 1, 2, 6, 5, 4 }, new long[] { 2, 3 })) {
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
        }


        [Fact]
        public void CopyCpuToCuda()
        {
            Tensor cpu = Float32Tensor.ones(new long[] { 2, 2 });
            Assert.Equal("cpu", cpu.device.ToString());

            if (torch.cuda.is_available()) {
                var cuda = cpu.cuda();
                Assert.Equal("cuda:0", cuda.device.ToString());

                // Copy back to CPU to inspect the elements
                var cpu2 = cuda.cpu();
                Assert.Equal("cpu", cpu2.device.ToString());
                var data = cpu.Data<float>();
                for (int i = 0; i < 4; i++) {
                    Assert.Equal(1, data[i]);
                }
            } else {
                Assert.Throws<InvalidOperationException>(() => cpu.cuda());
            }

        }

        [Fact]
        public void CopyCudaToCpu()
        {
            if (torch.cuda.is_available()) {
                var cuda = Float32Tensor.ones(new long[] { 2, 2 }, torch.CUDA);
                Assert.Equal("cuda:0", cuda.device.ToString());

                var cpu = cuda.cpu();
                Assert.Equal("cpu", cpu.device.ToString());

                var data = cpu.Data<float>();
                for (int i = 0; i < 4; i++) {
                    Assert.Equal(1, data[i]);
                }
            } else {
                Assert.Throws<InvalidOperationException>((Action)(() => { Float32Tensor.ones(new long[] { 2, 2 }, (Device)torch.CUDA); }));
            }
        }

        [Fact]
        public void TestSquareEuclideanDistance()
        {
            var input = new double[] { 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1 }.ToTensor(new long[] { 9 }).to_type(ScalarType.Float32);
            var zeros = Float32Tensor.zeros(new long[] { 1, 9 });
            var ones = Float32Tensor.ones(new long[] { 1, 9 });
            var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);

            var distanceFromZero = input.reshape(new long[] { -1, 1, 9 }).sub(zeros).pow(2.ToScalar()).sum(new long[] { 2 });
            var distanceFromOne = input.reshape(new long[] { -1, 1, 9 }).sub(ones).pow(2.ToScalar()).sum(new long[] { 2 });
            var distanceFromCentroids = input.reshape(new long[] { -1, 1, 9 }).sub(centroids).pow(2.ToScalar()).sum(new long[] { 2 });

            Assert.True(true);
        }

        [Fact]
        public void TestCat()
        {
            var zeros = Float32Tensor.zeros(new long[] { 1, 9 });
            var ones = Float32Tensor.ones(new long[] { 1, 9 });
            var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);

            var shape = centroids.shape;
            Assert.Equal(new long[] { 2, 9 }, shape);
        }

        [Fact]
        public void TestCatCuda()
        {
            if (torch.cuda.is_available()) {
                var zeros = Float32Tensor.zeros(new long[] { 1, 9 }).cuda();
                var ones = Float32Tensor.ones(new long[] { 1, 9 }).cuda();
                var centroids = torch.cat(new Tensor[] { zeros, ones }, 0);
                var shape = centroids.shape;
                Assert.Equal(new long[] { 2, 9 }, shape);
                Assert.Equal(DeviceType.CUDA, centroids.device_type);
            }
        }

        [Fact]
        public void TestCopy()
        {
            var first = Float32Tensor.rand(new long[] { 1, 9 });
            var second = Float32Tensor.zeros(new long[] { 1, 9 });

            second.copy_(first);

            Assert.Equal(first, second);
        }

        [Fact]
        public void TestCopyCuda()
        {
            if (torch.cuda.is_available()) {
                var first = Float32Tensor.rand(new long[] { 1, 9 }).cuda();
                var second = Float32Tensor.zeros(new long[] { 1, 9 }).cuda();

                second.copy_(first);

                Assert.Equal(first, second);
            }
        }

        void TestStackGen(Device device)
        {
            {
                var t1 = Float32Tensor.zeros(new long[] { }, device);
                var t2 = Float32Tensor.ones(new long[] { }, device);
                var t3 = Float32Tensor.ones(new long[] { }, device);
                var res = torch.stack(new Tensor[] { t1, t2, t3 }, 0);

                var shape = res.shape;
                Assert.Equal(new long[] { 3 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = Float32Tensor.zeros(new long[] { }, device);
                var t2 = Float32Tensor.ones(new long[] { }, device);
                var t3 = Float32Tensor.ones(new long[] { }, device);
                var res = torch.hstack(new Tensor[] { t1, t2, t3 });

                var shape = res.shape;
                Assert.Equal(new long[] { 3 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = Float32Tensor.zeros(new long[] { 2, 9 }, device);
                var t2 = Float32Tensor.ones(new long[] { 2, 9 }, device);
                var res = torch.stack(new Tensor[] { t1, t2 }, 0);

                var shape = res.shape;
                Assert.Equal(new long[] { 2, 2, 9 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = Float32Tensor.zeros(new long[] { 2, 9 }, device);
                var t2 = Float32Tensor.ones(new long[] { 2, 9 }, device);
                var res = torch.hstack(new Tensor[] { t1, t2 });

                var shape = res.shape;
                Assert.Equal(new long[] { 2, 18 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = Float32Tensor.zeros(new long[] { 2, 9 }, device);
                var t2 = Float32Tensor.ones(new long[] { 2, 9 }, device);
                var res = torch.vstack(new Tensor[] { t1, t2 });

                var shape = res.shape;
                Assert.Equal(new long[] { 4, 9 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
            {
                var t1 = Float32Tensor.zeros(new long[] { 2, 9 }, device);
                var t2 = Float32Tensor.ones(new long[] { 2, 9 }, device);
                var res = torch.dstack(new Tensor[] { t1, t2 });

                var shape = res.shape;
                Assert.Equal(new long[] { 2, 9, 2 }, shape);
                Assert.Equal(device.type, res.device_type);
            }
        }

        [Fact]
        public void TestMoveAndCast()
        {
            var input = Float64Tensor.rand(new long[] { 128 }, torch.CPU);

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
        public void TestMaskedScatter()
        {
            var input = Float32Tensor.zeros(new long[] { 4, 4 });
            var mask = BoolTensor.zeros(new long[] { 4, 4 });
            var tTrue = BoolTensor.from(true);
            mask[0, 1] = tTrue;
            mask[2, 3] = tTrue;

            var res = input.masked_scatter(mask, Float32Tensor.from(new float[] { 3.14f, 2 * 3.14f }));
            Assert.Equal(3.14f, res[0, 1].DataItem<float>());
            Assert.Equal(2 * 3.14f, res[2, 3].DataItem<float>());

            input.masked_scatter_(mask, Float32Tensor.from(new float[] { 3.14f, 2 * 3.14f }));
            Assert.Equal(3.14f, input[0, 1].DataItem<float>());
            Assert.Equal(2 * 3.14f, input[2, 3].DataItem<float>());
        }

        [Fact]
        public void TestMaskedSelect()
        {
            var input = Float32Tensor.zeros(new long[] { 4, 4 });
            var mask = BoolTensor.eye(4, 4);

            var res = input.masked_select(mask);
            Assert.Equal(4, res.numel());
        }

        [Fact]
        public void TestStackCpu()
        {
            TestStackGen(torch.CPU);
        }

        [Fact]
        public void TestStackCuda()
        {
            if (torch.cuda.is_available()) {
                TestStackGen(torch.CUDA);
            }
        }

        [Fact]
        public void TestBlockDiag()
        {
            // Example from PyTorch documentation
            var A = Int64Tensor.from(new long[] { 0, 1, 1, 0 }, 2, 2);
            var B = Int64Tensor.from(new long[] { 3, 4, 5, 6, 7, 8 }, 2, 3);
            var C = Int64Tensor.from(7);
            var D = Int64Tensor.from(new long[] { 1, 2, 3 });
            var E = Int64Tensor.from(new long[] { 4, 5, 6 }, 3, 1);

            var expected = Int64Tensor.from(new long[] {
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
        public void TestDiag1D()
        {
            var input = Int64Tensor.ones(new long[] { 3 });
            var expected = new long[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            var res = input.diag();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.Data<long>().ToArray());
        }

        [Fact]
        public void TestDiag2D()
        {
            var input = Int64Tensor.zeros(new long[] { 5, 5 });
            for (int i = 0; i < 5; i++) {
                input[i, i] = Int64Tensor.from(1);
            }

            var expected = new long[] { 1, 1, 1, 1, 1 };

            var res = input.diag();
            Assert.Equal(1, res.Dimensions);
            Assert.Equal(expected, res.Data<long>().ToArray());
        }


        [Fact]
        public void TestDiagFlat1D()
        {
            var input = Int64Tensor.ones(new long[] { 3 });
            var expected = new long[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            var res = input.diagflat();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.Data<long>().ToArray());
        }

        [Fact]
        public void TestDiagFlat2D()
        {
            var input = Int64Tensor.ones(new long[] { 2, 2 });

            var expected = new long[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

            var res = input.diagflat();
            Assert.Equal(2, res.Dimensions);
            Assert.Equal(expected, res.Data<long>().ToArray());
        }

        [Fact]
        public void TestDimensions()
        {
            {
                var res = Float32Tensor.rand(new long[] { 5 });
                Assert.Equal(1, res.Dimensions);
                Assert.Equal(1, res.dim());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5, 5 });
                Assert.Equal(2, res.Dimensions);
                Assert.Equal(2, res.dim());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5, 5, 5 });
                Assert.Equal(3, res.Dimensions);
                Assert.Equal(3, res.dim());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
                Assert.Equal(4, res.Dimensions);
                Assert.Equal(4, res.dim());
            }
        }

        [Fact]
        public void TestNumberofElements()
        {
            {
                var res = Float32Tensor.rand(new long[] { 5 });
                Assert.Equal(5, res.NumberOfElements);
                Assert.Equal(5, res.numel());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5, 5 });
                Assert.Equal(25, res.NumberOfElements);
                Assert.Equal(25, res.numel());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5, 5, 5 });
                Assert.Equal(125, res.NumberOfElements);
                Assert.Equal(125, res.numel());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
                Assert.Equal(625, res.NumberOfElements);
                Assert.Equal(625, res.numel());
            }
        }

        [Fact]
        public void TestElementSize()
        {
            {
                var res = Int8Tensor.randint(100, new long[] { 5 });
                Assert.Equal(1, res.ElementSize);
                Assert.Equal(1, res.element_size());
            }
            {
                var res = Int16Tensor.randint(250, new long[] { 5 });
                Assert.Equal(2, res.ElementSize);
                Assert.Equal(2, res.element_size());
            }
            {
                var res = Int32Tensor.randint(250, new long[] { 5 });
                Assert.Equal(4, res.ElementSize);
                Assert.Equal(4, res.element_size());
            }
            {
                var res = Int64Tensor.randint(250, new long[] { 5 });
                Assert.Equal(8, res.ElementSize);
                Assert.Equal(8, res.element_size());
            }
            {
                var res = Float32Tensor.rand(new long[] { 5 });
                Assert.Equal(4, res.ElementSize);
                Assert.Equal(4, res.element_size());
            }
            {
                var res = Float64Tensor.rand(new long[] { 5 });
                Assert.Equal(8, res.ElementSize);
                Assert.Equal(8, res.element_size());
            }
        }

        [Fact]
        public void TestAtleast1d()
        {
            {
                var input = Float32Tensor.from(1.0f);
                var res = input.atleast_1d();
                Assert.Equal(1, res.Dimensions);
                Assert.Equal(1, res.dim());
                Assert.Equal(1, res.NumberOfElements);
                Assert.Equal(1, res.numel());
            }
            {
                var input = Float32Tensor.rand(new long[] { 5 });
                var res = input.atleast_1d();
                Assert.Equal(1, res.Dimensions);
                Assert.Equal(1, res.dim());
                Assert.Equal(5, res.NumberOfElements);
                Assert.Equal(5, res.numel());
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5 });
                var res = input.atleast_1d();
                Assert.Equal(2, res.Dimensions);
                Assert.Equal(2, res.dim());
                Assert.Equal(25, res.NumberOfElements);
                Assert.Equal(25, res.numel());
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5, 5 });
                var res = input.atleast_1d();
                Assert.Equal(3, res.Dimensions);
                Assert.Equal(3, res.dim());
                Assert.Equal(125, res.NumberOfElements);
                Assert.Equal(125, res.numel());
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
                var res = input.atleast_1d();
                Assert.Equal(4, res.Dimensions);
                Assert.Equal(4, res.dim());
                Assert.Equal(625, res.NumberOfElements);
                Assert.Equal(625, res.numel());
            }
        }

        [Fact]
        public void TestAtleast2d()
        {
            {
                var input = Float32Tensor.from(1.0f);
                var res = input.atleast_2d();
                Assert.Equal(2, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5 });
                var res = input.atleast_2d();
                Assert.Equal(2, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5 });
                var res = input.atleast_2d();
                Assert.Equal(2, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5, 5 });
                var res = input.atleast_2d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
                var res = input.atleast_2d();
                Assert.Equal(4, res.Dimensions);
            }
        }

        [Fact]
        public void TestAtleast3d()
        {
            {
                var input = Float32Tensor.from(1.0f);
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5 });
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5 });
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5, 5 });
                var res = input.atleast_3d();
                Assert.Equal(3, res.Dimensions);
            }
            {
                var input = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
                var res = input.atleast_3d();
                Assert.Equal(4, res.Dimensions);
            }
        }

        [Fact]
        public void TestSetGrad()
        {
            var x = Float32Tensor.rand(new long[] { 10, 10 });
            Assert.False(x.requires_grad);

            x.requires_grad = true;
            Assert.True(x.requires_grad);
            x.requires_grad = false;
            Assert.False(x.requires_grad);
        }

        [Fact]
        public void TestAutoGradMode()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var x = Float32Tensor.randn(new long[] { 2, 3 }, requiresGrad: true);
                using (var mode = new AutoGradMode(false)) {
                    Assert.False(AutoGradMode.IsAutogradEnabled());
                    var sum = x.sum();
                    Assert.Throws<ExternalException>(() => sum.backward());
                    //var grad = x.Grad();
                    //Assert.True(grad.Handle == IntPtr.Zero);
                }
                using (var mode = new AutoGradMode(true)) {
                    Assert.True(AutoGradMode.IsAutogradEnabled());
                    var sum = x.sum();
                    sum.backward();
                    var grad = x.grad();
                    Assert.False(grad.Handle == IntPtr.Zero);
                    var data = grad.Data<float>();
                    for (int i = 0; i < 2 * 3; i++) {
                        Assert.Equal(1.0, data[i]);
                    }
                }
            }
        }

        [Fact]
        public void TestSubInPlace()
        {
            var x = Int32Tensor.ones(new long[] { 100, 100 });
            var y = Int32Tensor.ones(new long[] { 100, 100 });

            x.sub_(y);

            var xdata = x.Data<int>();

            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    Assert.Equal(0, xdata[i + j]);
                }
            }
        }

        [Fact]
        public void TestMemoryDisposalZeros()
        {
            for (int i = 0; i < 1024; i++) {
                var x = Float64Tensor.zeros(new long[] { 1024, 1024 });
                x.Dispose();
                //System.GC.Collect();
            }
        }

        [Fact]
        public void TestMemoryDisposalOnes()
        {
            for (int i = 0; i < 1024; i++) {
                var x = Float64Tensor.ones(new long[] { 1024, 1024 });
                x.Dispose();
            }
        }

        [Fact]
        public void TestMemoryDisposalScalarTensors()
        {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 1000 * 100; j++) {
                    var x = Float64Tensor.from(i * j * 3.1415);
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
            var tensor = Float64Tensor.ones(new long[] { 5, 6 });
            tensor.save(file);
            var tensorLoaded = Tensor.load(file);
            File.Delete(file);
            Assert.NotNull(tensorLoaded);
            Assert.Equal(tensorLoaded.dtype, tensor.dtype);
            Assert.Equal(tensorLoaded, tensor);
        }

        [Fact]
        public void TestSaveLoadTensorFloat()
        {
            var file = ".saveload.float.ts";
            if (File.Exists(file)) File.Delete(file);
            var tensor = Float32Tensor.ones(new long[] { 5, 6 });
            tensor.save(file);
            var tensorLoaded = Tensor.load(file);
            File.Delete(file);
            Assert.NotNull(tensorLoaded);
            Assert.Equal(tensorLoaded.dtype, tensor.dtype);
            Assert.Equal(tensorLoaded, tensor);
        }


        [Fact]
        public void TestArithmeticOperatorsFloat16()
        {
            // Float16 arange_cuda not available on cuda in LibTorch 1.8.0
            // Float16 arange_cpu not available on cuda in LibTorch 1.8.0
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = Float16Tensor.ones(new long[] { 10, 10 }, device: device);
                    var c2 = Float16Tensor.ones(new long[] { 10, 10 }, device: device);
                    var c3 = Float16Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, float> getFunc = (tt, i, j) => tt[i, j].ToSingle();
                    // scalar-tensor operators
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a + 0.5f, a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f + a, a => 0.5f + a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a - 0.5f, a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a * 0.5f, a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f * a, a => 0.5f * a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a / 0.5f, a => a / 0.5f);

                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.add(0.5f), a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.sub(0.5f), a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.mul(0.5f), a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.div(0.5f), a => a / 0.5f);

                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.add_(0.5f), a => a + 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.sub_(0.5f), a => a - 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.mul_(0.5f), a => a * 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.div_(0.5f), a => a / 0.5f);

                    // tensor-tensor operators
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        public void TestArithmeticOperatorsBFloat16()
        {
            // BFloat16 arange_cuda not available on cuda in LibTorch 1.8.0
            // BFloat16 arange_cpu not available on cuda in LibTorch 1.8.0
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = Float16Tensor.ones(new long[] { 10, 10 }, device: device);
                    var c2 = Float16Tensor.ones(new long[] { 10, 10 }, device: device);
                    var c3 = Float16Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, float> getFunc = (tt, i, j) => tt[i, j].ToSingle();
                    // scalar-tensor operators
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a + 0.5f, a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f + a, a => 0.5f + a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a - 0.5f, a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a * 0.5f, a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f * a, a => 0.5f * a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a / 0.5f, a => a / 0.5f);

                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.add(0.5f), a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.sub(0.5f), a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.mul(0.5f), a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.div(0.5f), a => a / 0.5f);

                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.add_(0.5f), a => a + 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.sub_(0.5f), a => a - 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.mul_(0.5f), a => a * 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.div_(0.5f), a => a / 0.5f);

                    // tensor-tensor operators
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        public void TestArithmeticOperatorsFloat32()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = Float32Tensor.arange(0, 10, device: device).expand(new long[] { 10, 10 });
                    var c2 = Float32Tensor.arange(10, 0, -1, device: device).expand(new long[] { 10, 10 });
                    var c3 = Float32Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, float> getFunc = (tt, i, j) => tt[i, j].ToSingle();
                    // scalar-tensor operators
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a + 0.5f, a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f + a, a => 0.5f + a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a - 0.5f, a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a * 0.5f, a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f * a, a => 0.5f * a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a / 0.5f, a => a / 0.5f);

                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.add(0.5f), a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.sub(0.5f), a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.mul(0.5f), a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.div(0.5f), a => a / 0.5f);

                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.add_(0.5f), a => a + 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.sub_(0.5f), a => a - 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.mul_(0.5f), a => a * 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.div_(0.5f), a => a / 0.5f);

                    // tensor-tensor operators
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        public void TestArithmeticOperatorsComplexFloat32()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = Float32Tensor.arange(0, 10, device: device).expand(new long[] { 10, 10 });
                    var c2 = Float32Tensor.arange(10, 0, -1, device: device).expand(new long[] { 10, 10 });
                    var c3 = ComplexFloat32Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, float> getFunc = (tt, i, j) => tt[i, j].ToSingle();
                    // scalar-tensor operators
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a + 0.5f, a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f + a, a => 0.5f + a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a - 0.5f, a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a * 0.5f, a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => 0.5f * a, a => 0.5f * a);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a / 0.5f, a => a / 0.5f);

                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.add(0.5f), a => a + 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.sub(0.5f), a => a - 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.mul(0.5f), a => a * 0.5f);
                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a.div(0.5f), a => a / 0.5f);

                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.add_(0.5f), a => a + 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.sub_(0.5f), a => a - 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.mul_(0.5f), a => a * 0.5f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.div_(0.5f), a => a / 0.5f);

                    // tensor-tensor operators
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        public void TestArithmeticOperatorsFloat64()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = Float64Tensor.arange(0, 10, device: device).expand(new long[] { 10, 10 });
                    var c2 = Float64Tensor.arange(10, 0, -1, device: device).expand(new long[] { 10, 10 });
                    var c3 = Float64Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, double> getFunc = (tt, i, j) => tt[i, j].ToDouble();
                    // scalar-tensor operators
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a + 0.5, a => a + 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => 0.5 + a, a => 0.5 + a);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a - 0.5, a => a - 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a * 0.5, a => a * 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => 0.5 * a, a => 0.5 * a);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a / 0.5, a => a / 0.5);

                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.add(0.5), a => a + 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.sub(0.5), a => a - 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.mul(0.5), a => a * 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.div(0.5), a => a / 0.5);

                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.add_(0.5), a => a + 0.5);
                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.sub_(0.5), a => a - 0.5);
                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.mul_(0.5), a => a * 0.5);
                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.div_(0.5), a => a / 0.5);

                    // tensor-tensor operators
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        public void TestArithmeticOperatorsComplexFloat64()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = ComplexFloat64Tensor.arange(0, 10, device: device).expand(new long[] { 10, 10 });
                    var c2 = ComplexFloat64Tensor.arange(10, 0, -1, device: device).expand(new long[] { 10, 10 });
                    var c3 = ComplexFloat64Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, double> getFunc = (tt, i, j) => tt[i, j].ToDouble();
                    // scalar-tensor operators
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a + 0.5, a => a + 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => 0.5 + a, a => 0.5 + a);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a - 0.5, a => a - 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a * 0.5, a => a * 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => 0.5 * a, a => 0.5 * a);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a / 0.5, a => a / 0.5);

                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.add(0.5), a => a + 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.sub(0.5), a => a - 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.mul(0.5), a => a * 0.5);
                    TestOneTensor<double, double>(c1, c2, getFunc, getFunc, a => a.div(0.5), a => a / 0.5);

                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.add_(0.5), a => a + 0.5);
                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.sub_(0.5), a => a - 0.5);
                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.mul_(0.5), a => a * 0.5);
                    TestOneTensorInPlace<double>(c1, c2, getFunc, a => a.div_(0.5), a => a / 0.5);

                    // tensor-tensor operators
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    TestTwoTensor<double, double>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    TestTwoTensorInPlace<double>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        public void TestComparisonOperatorsFloat32()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = Float32Tensor.arange(0, 10, device: device).expand(new long[] { 10, 10 });
                    var c2 = Float32Tensor.arange(10, 0, -1, device: device).expand(new long[] { 10, 10 });
                    var c3 = Float32Tensor.ones(new long[] { 10, 10 }, device: device);
                    Func<Tensor, long, long, float> getFunc = (tt, i, j) => tt[i, j].ToSingle();
                    Func<Tensor, long, long, bool> getFuncBool = (tt, i, j) => tt[i, j].ToBoolean();
                    // scalar-tensor operators
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => a == 5.0f, a => a == 5.0f);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => a != 5.0f, a => a != 5.0f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.eq_(5.0f), a => a == 5.0f ? 1.0f : 0.0f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.ne_(5.0f), a => a != 5.0f ? 1.0f : 0.0f);

                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => a < 5.0f, a => a < 5.0f);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => 5.0f < a, a => 5.0f < a);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => a <= 5.0f, a => a <= 5.0f);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => 5.0f <= a, a => 5.0f <= a);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => a > 5.0f, a => a > 5.0f);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => 5.0f > a, a => 5.0f > a);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => a >= 5.0f, a => a >= 5.0f);
                    TestOneTensor<float, bool>(c1, c2, getFunc, getFuncBool, a => 5.0f >= a, a => 5.0f >= a);

                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.lt_(5.0f), a => a < 5.0f ? 1.0f : 0.0f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.le_(5.0f), a => a <= 5.0f ? 1.0f : 0.0f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.gt_(5.0f), a => a > 5.0f ? 1.0f : 0.0f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.ge_(5.0f), a => a >= 5.0f ? 1.0f : 0.0f);

                    TestOneTensor<float, float>(c1, c2, getFunc, getFunc, a => a % 5.0f, a => a % 5.0f);
                    TestOneTensorInPlace<float>(c1, c2, getFunc, a => a.remainder_(5.0f), a => a % 5.0f);

                    // tensor-tensor operators
                    TestTwoTensor<float, bool>(c1, c2, c3, getFunc, getFuncBool, (a, b) => a == b, (a, b) => a == b);
                    TestTwoTensor<float, bool>(c1, c2, c3, getFunc, getFuncBool, (a, b) => a != b, (a, b) => a != b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.eq_(b), (a, b) => a == b ? 1.0f : 0.0f);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.ne_(b), (a, b) => a != b ? 1.0f : 0.0f);

                    TestTwoTensor<float, bool>(c1, c2, c3, getFunc, getFuncBool, (a, b) => a < b, (a, b) => a < b);
                    TestTwoTensor<float, bool>(c1, c2, c3, getFunc, getFuncBool, (a, b) => a <= b, (a, b) => a <= b);
                    TestTwoTensor<float, bool>(c1, c2, c3, getFunc, getFuncBool, (a, b) => a > b, (a, b) => a > b);
                    TestTwoTensor<float, bool>(c1, c2, c3, getFunc, getFuncBool, (a, b) => a >= b, (a, b) => a >= b);

                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.lt_(b), (a, b) => a < b ? 1.0f : 0.0f);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.le_(b), (a, b) => a <= b ? 1.0f : 0.0f);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.gt_(b), (a, b) => a > b ? 1.0f : 0.0f);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.ge_(b), (a, b) => a >= b ? 1.0f : 0.0f);

                    TestTwoTensor<float, float>(c1, c2, c3, getFunc, getFunc, (a, b) => a % b, (a, b) => a % b);
                    TestTwoTensorInPlace<float>(c1, c2, c3, getFunc, (a, b) => a.remainder_(b), (a, b) => a % b);
                }
            }
        }

        private void TestOneTensor<Tin, Tout>(
            Tensor c1,
            Tensor c2,
            Func<Tensor, long, long, Tin> getFuncIn,
            Func<Tensor, long, long, Tout> getFuncOut,
            Func<Tensor, Tensor> tensorFunc,
            Func<Tin, Tout> scalarFunc)
        {
            var x = c1 * c2;
            var y = tensorFunc(x);

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    var xv = getFuncIn(x, i, j);
                    var yv = getFuncOut(y, i, j);
                    Assert.Equal<Tout>(yv, scalarFunc(xv));
                }
            }
        }

        private void TestOneTensorInPlace<Tin>(
            Tensor c1,
            Tensor c2,
            Func<Tensor, long, long, Tin> getFuncIn,
            Func<Tensor, Tensor> tensorFunc,
            Func<Tin, Tin> scalarFunc)
        {

            var x = c1 * c2;
            var xClone = x.clone();
            var y = tensorFunc(x);

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    var xClonev = getFuncIn(xClone, i, j);
                    var xv = getFuncIn(x, i, j);
                    var yv = getFuncIn(y, i, j);
                    Assert.Equal(yv, scalarFunc(xClonev));
                    Assert.Equal(yv, xv);
                }
            }
        }

        private void TestTwoTensor<Tin, Tout>(
            Tensor c1,
            Tensor c2,
            Tensor c3,
            Func<Tensor, long, long, Tin> getFuncIn,
            Func<Tensor, long, long, Tout> getFuncOut,
            Func<Tensor, Tensor, Tensor> tensorFunc,
            Func<Tin, Tin, Tout> scalarFunc)
        {

            var x = c1 * c3;
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    var xv = getFuncIn(x, i, j);
                    var yv = getFuncIn(y, i, j);
                    var zv = getFuncOut(z, i, j);
                    Assert.Equal(zv, scalarFunc(xv, yv));
                }
            }
        }

        private void TestTwoTensorInPlace<Tin>(
            Tensor c1,
            Tensor c2,
            Tensor c3,
            Func<Tensor, long, long, Tin> getFuncIn,
            Func<Tensor, Tensor, Tensor> tensorFunc,
            Func<Tin, Tin, Tin> scalarFunc)
        {

            var x = c1 * c3;
            var xClone = x.clone();
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            if (x.device_type == DeviceType.CPU) {
                var xData = x.Data<Tin>();
                var xCloneData = xClone.Data<Tin>();
                var yData = y.Data<Tin>();
                var zData = z.Data<Tin>();

                Assert.True(xData == zData);
            }

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    var xClonev = getFuncIn(xClone, i, j);
                    var xv = getFuncIn(x, i, j);
                    var yv = getFuncIn(y, i, j);
                    var zv = getFuncIn(z, i, j);
                    Assert.Equal(zv, scalarFunc(xClonev, yv));
                    Assert.Equal(zv, xv);
                }
            }
        }

        [Fact]
        public void TestMul()
        {
            var x = Float32Tensor.ones(new long[] { 100, 100 });

            var y = x.mul(0.5f.ToScalar());

            var ydata = y.Data<float>();
            var xdata = x.Data<float>();

            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    Assert.Equal(ydata[i + j], xdata[i + j] * 0.5f);
                }
            }
        }

        void TestMmGen(Device device)
        {
            {
                var x1 = Float32Tensor.ones(new long[] { 1, 2 }, device: device);
                var x2 = Float32Tensor.ones(new long[] { 2, 1 }, device: device);

                var y = x1.mm(x2).to(DeviceType.CPU);

                var ydata = y.Data<float>();

                Assert.Equal(2.0f, ydata[0]);
            }
            //System.Runtime.InteropServices.ExternalException : addmm for CUDA tensors only supports floating - point types.Try converting the tensors with.float() at C:\w\b\windows\pytorch\aten\src\THC / generic / THCTensorMathBlas.cu:453
            if (device.type == DeviceType.CPU) {
                var x1 = Int64Tensor.ones(new long[] { 1, 2 }, device: device);
                var x2 = Int64Tensor.ones(new long[] { 2, 1 }, device: device);

                var y = x1.mm(x2).to(DeviceType.CPU);

                var ydata = y.Data<long>();

                Assert.Equal(2L, ydata[0]);
            }
        }

        [Fact]
        public void TestMmCpu()
        {
            TestMmGen(torch.CPU);
        }

        [Fact]
        public void TestMmCuda()
        {
            if (torch.cuda.is_available()) {
                TestMmGen(torch.CUDA);
            }
        }

        void TestMVGen(Device device)
        {
            {
                var mat1 = Float32Tensor.ones(new long[] { 4, 3 }, device: device);
                var vec1 = Float32Tensor.ones(new long[] { 3 }, device: device);

                var y = mat1.mv(vec1).to(DeviceType.CPU);

                Assert.Equal(4, y.shape[0]);
            }
        }

        void TestAddMVGen(Device device)
        {
            {
                var x1 = Float32Tensor.ones(new long[] { 4 }, device: device);
                var mat1 = Float32Tensor.ones(new long[] { 4, 3 }, device: device);
                var vec1 = Float32Tensor.ones(new long[] { 3 }, device: device);

                var y = x1.addmv(mat1, vec1).to(DeviceType.CPU);

                Assert.Equal(4, y.shape[0]);
            }
        }

        [Fact]
        public void TestMVCpu()
        {
            TestMVGen(torch.CPU);
        }

        [Fact]
        public void TestMVCuda()
        {
            if (torch.cuda.is_available()) {
                TestMVGen(torch.CUDA);
            }
        }

        [Fact]
        public void TestAddMVCpu()
        {
            TestAddMVGen(torch.CPU);
        }

        [Fact]
        public void TestAddMVCuda()
        {
            if (torch.cuda.is_available()) {
                TestAddMVGen(torch.CUDA);
            }
        }

        void TestAddRGen(Device device)
        {
            {
                var x1 = Float32Tensor.ones(new long[] { 4, 3 }, device: device);
                var vec1 = Float32Tensor.ones(new long[] { 4 }, device: device);
                var vec2 = Float32Tensor.ones(new long[] { 3 }, device: device);

                var y = x1.addr(vec1, vec2).to(DeviceType.CPU);

                Assert.Equal(new long[] { 4, 3 }, y.shape);
            }
        }


        [Fact]
        public void TestPositive()
        {
            var a = Float32Tensor.randn(25, 25);
            var b = a.positive();

            Assert.Equal(a.Data<float>().ToArray(), b.Data<float>().ToArray());

            var c = BoolTensor.ones(25, 25);
            Assert.Throws<ArgumentException>(() => c.positive());
        }

        [Fact]
        public void TestFrexp()
        {
            var x = Float32Tensor.arange(9);
            var r = x.frexp();

            Assert.Equal(new float[] { 0.0000f, 0.5000f, 0.5000f, 0.7500f, 0.5000f, 0.6250f, 0.7500f, 0.8750f, 0.5000f }, r.Mantissa.Data<float>().ToArray());
            Assert.Equal(new int[] { 0, 1, 2, 2, 3, 3, 3, 3, 4 }, r.Exponent.Data<int>().ToArray());
        }

        [Fact]
        public void TestAddRCpu()
        {
            TestAddRGen(torch.CPU);
        }

        [Fact]
        public void TestAddRCuda()
        {
            if (torch.cuda.is_available()) {
                TestAddRGen(torch.CUDA);
            }
        }

        [Fact]
        public void Deg2RadTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(angl => (angl * MathF.PI) / 180.0f).ToArray();
            var res = Float32Tensor.from(data).deg2rad();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void Rad2DegTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(angl => (angl * 180.0f) / MathF.PI).ToArray();
            var res = Float32Tensor.from(data).rad2deg();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void AbsTest()
        {
            var data = Float32Tensor.arange(-10.0f, 10.0f, 1.0f);
            var expected = data.Data<float>().ToArray().Select(MathF.Abs).ToArray();
            var res = data.abs();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void AbsTestC32()
        {
            var data = ComplexFloat32Tensor.rand(new long[] { 25 });
            var expected = data.Data<(float R, float I)>().ToArray().Select(c => MathF.Sqrt(c.R * c.R + c.I * c.I)).ToArray();
            var res = data.abs();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void AbsTestC64()
        {
            var data = ComplexFloat64Tensor.rand(new long[] { 25 });
            var expected = data.Data<System.Numerics.Complex>().ToArray().Select(c => Math.Sqrt(c.Real * c.Real + c.Imaginary * c.Imaginary)).ToArray<double>();
            var res = data.abs();
            Assert.True(res.allclose(Float64Tensor.from(expected)));
        }

        [Fact]
        public void AngleTestC32()
        {
            var data = ComplexFloat32Tensor.randn(new long[] { 25 });
            var expected = data.Data<(float R, float I)>().ToArray().Select(c => {
                var x = c.R;
                var y = c.I;
                return (x > 0 || y != 0) ? 2 * MathF.Atan(y / (MathF.Sqrt(x * x + y * y) + x)) : (x < 0 && y == 0) ? MathF.PI : 0;
            }).ToArray();
            var res = data.angle();
            Assert.True(res.allclose(Float32Tensor.from(expected), rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void AngleTestC64()
        {
            var data = ComplexFloat64Tensor.randn(new long[] { 25 });
            var expected = data.Data<System.Numerics.Complex>().ToArray().Select(c => {
                var x = c.Real;
                var y = c.Imaginary;
                return (x > 0 || y != 0) ? 2 * Math.Atan(y / (Math.Sqrt(x * x + y * y) + x)) : (x < 0 && y == 0) ? Math.PI : 0;
            }).ToArray<double>();
            var res = data.angle();
            Assert.True(res.allclose(Float64Tensor.from(expected)));
        }

        [Fact]
        public void SqrtTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sqrt).ToArray();
            var res = Float32Tensor.from(data).sqrt();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void SinTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sin).ToArray();
            var res = Float32Tensor.from(data).sin();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
            res = torch.sin(torch.tensor(data));
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void CosTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cos).ToArray();
            var res = Float32Tensor.from(data).cos();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
            res = torch.cos(torch.tensor(data));
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void I0Test()
        {
            var data = Float32Tensor.arange(0, 5, 1);
            var expected = new float[] { 0.99999994f, 1.266066f, 2.27958512f, 4.88079262f, 11.3019209f };
            var res = data.i0();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void HypotTest()
        {
            var a = new float[] { 1.0f, 2.0f, 3.0f };
            var b = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = a.Select(x => MathF.Sqrt(2.0f) * x).ToArray();
            var res = Float32Tensor.from(a).hypot(Float32Tensor.from(b));
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void VdotTest()
        {
            var a = new float[] { 1.0f, 2.0f, 3.0f };
            var b = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = Float32Tensor.from(a.Zip(b).Select(x => x.First * x.Second).Sum());
            var res = Float32Tensor.from(a).vdot(Float32Tensor.from(b));
            Assert.True(res.allclose(expected));
        }

        [Fact]
        public void WhereTest()
        {
            var bits = 3;
            var mask = -(1 << (8 - bits));
            var condition = Float32Tensor.rand(25) > 0.5;
            var ones = Int32Tensor.ones(25);
            var zeros = Int32Tensor.zeros(25);

            var cond1 = ones.where(condition, zeros);
            var cond2 = condition.to_type(ScalarType.Int32);
            Assert.Equal(cond1, cond2);
        }

        [Fact]
        public void HeavisideTest()
        {
            var input = new float[] { -1.0f, 0.0f, 3.0f };
            var values = new float[] { 1.0f, 2.0f, 1.0f };
            var expected = new float[] { 0.0f, 2.0f, 1.0f };
            var res = Float32Tensor.from(input).heaviside(Float32Tensor.from(values));
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void MaximumTest()
        {
            var a = Float32Tensor.from(new float[] { 1.0f, 2.0f, 3.0f });
            var b = a.neg();
            var expected = a;
            var res = a.maximum(b);
            Assert.Equal(expected, res);
        }

        [Fact]
        public void MinimumTest()
        {
            var a = Float32Tensor.from(new float[] { 1.0f, 2.0f, 3.0f });
            var b = a.neg();
            var expected = b;
            var res = a.minimum(b);
            Assert.Equal(expected, res);
        }

        [Fact]
        public void ArgMaxTest()
        {
            var a = Float32Tensor.randn(new long[] { 15, 5 });
            var b = a.argmax();
            Assert.Equal(1, b.NumberOfElements);
            var c = a.argmax(0, keepDim: true);
            Assert.Equal(new long[] { 1, 5 }, c.shape);
            var d = a.argmax(0, keepDim: false);
            Assert.Equal(new long[] { 5 }, d.shape);
        }

        [Fact]
        public void ArgMinTest()
        {
            var a = Float32Tensor.randn(new long[] { 15, 5 });
            var b = a.argmin();
            Assert.Equal(1, b.NumberOfElements);
            var c = a.argmin(0, keepDim: true);
            Assert.Equal(new long[] { 1, 5 }, c.shape);
            var d = a.argmin(0, keepDim: false);
            Assert.Equal(new long[] { 5 }, d.shape);
        }

        [Fact]
        public void AMaxTest()
        {
            var a = Float32Tensor.randn(new long[] { 15, 5, 4, 3 });
            var b = a.amax(new long[] { 0, 1 });
            Assert.Equal(new long[] { 4, 3 }, b.shape);
            var c = a.amax(new long[] { 0, 1 }, keepDim: true);
            Assert.Equal(new long[] { 1, 1, 4, 3 }, c.shape);
        }

        [Fact]
        public void AMinTest()
        {
            var a = Float32Tensor.randn(new long[] { 15, 5, 4, 3 });
            var b = a.amax(new long[] { 0, 1 });
            Assert.Equal(new long[] { 4, 3 }, b.shape);
            var c = a.amax(new long[] { 0, 1 }, keepDim: true);
            Assert.Equal(new long[] { 1, 1, 4, 3 }, c.shape);
        }

        [Fact]
        public void TanTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tan).ToArray();
            var res = Float32Tensor.from(data).tan();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void SinhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sinh).ToArray();
            var res = Float32Tensor.from(data).sinh();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void CoshTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cosh).ToArray();
            var res = Float32Tensor.from(data).cosh();
            var tmp = res.Data<Single>();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void TanhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tanh).ToArray();
            var res = Float32Tensor.from(data).tanh();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void ArcSinhTest()
        {
            var data = new float[] { -0.1f, 0.0f, 0.1f };
            var expected = data.Select(MathF.Asinh).ToArray();
            var res = Float32Tensor.from(data).asinh();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void ArcCoshTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Acosh).ToArray();
            var res = Float32Tensor.from(data).acosh();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void ArcTanhTest()
        {
            var data = new float[] { -0.1f, 0.0f, 0.1f };
            var expected = data.Select(MathF.Atanh).ToArray();
            var res = Float32Tensor.from(data).atanh();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void AsinTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Asin).ToArray();
            {
                var res = Float32Tensor.from(data).asin();
                Assert.True(res.allclose(Float32Tensor.from(expected)));
            }
            {
                var res = Float32Tensor.from(data).arcsin();
                Assert.True(res.allclose(Float32Tensor.from(expected)));
            }
        }

        [Fact]
        public void AcosTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Acos).ToArray();
            {
                var res = Float32Tensor.from(data).acos();
                Assert.True(res.allclose(Float32Tensor.from(expected)));
            }
            {
                var res = Float32Tensor.from(data).arccos();
                Assert.True(res.allclose(Float32Tensor.from(expected)));
            }
        }

        [Fact]
        public void AtanTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Atan).ToArray();
            {
                var res = Float32Tensor.from(data).atan();
                Assert.True(res.allclose(Float32Tensor.from(expected)));
            }
            {
                var res = Float32Tensor.from(data).arctan();
                Assert.True(res.allclose(Float32Tensor.from(expected)));
            }
        }

        [Fact]
        public void LogTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(x => MathF.Log(x)).ToArray();
            var res = Float32Tensor.from(data).log();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void Log10Test()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Log10).ToArray();
            var res = Float32Tensor.from(data).log10();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void Log2Test()
        {
            var data = new float[] { 1.0f, 2.0f, 32.0f };
            var expected = data.Select(MathF.Log2).ToArray();
            var res = Float32Tensor.from(data).log2();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void LogitTest()
        {
            // From the PyTorch reference docs.
            var data = new float[] { 0.2796f, 0.9331f, 0.6486f, 0.1523f, 0.6516f };
            var expected = new float[] { -0.946446538f, 2.635313f, 0.6128909f, -1.71667457f, 0.6260796f };
            var res = Float32Tensor.from(data).logit(eps: 1f - 6);
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
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
            var res = Float32Tensor.from(data).logcumsumexp(dim: 0);
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void LogAddExpTest()
        {
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            var y = new float[] { 4.0f, 5.0f, 6.0f };
            var expected = new float[x.Length];
            for (int i = 0; i < x.Length; i++) {
                expected[i] = MathF.Log(MathF.Exp(x[i]) + MathF.Exp(y[i]));
            }
            var res = Float32Tensor.from(x).logaddexp(Float32Tensor.from(y));
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void LogAddExp2Test()
        {
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            var y = new float[] { 4.0f, 5.0f, 6.0f };
            var expected = new float[x.Length];
            for (int i = 0; i < x.Length; i++) {
                expected[i] = MathF.Log(MathF.Pow(2.0f, x[i]) + MathF.Pow(2.0f, y[i]), 2.0f);
            }
            var res = Float32Tensor.from(x).logaddexp2(Float32Tensor.from(y));
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void ReciprocalTest()
        {
            var x = Float32Tensor.ones(new long[] { 10, 10 });
            x.fill_(4.0f);
            var y = x.reciprocal();

            Assert.All(x.Data<float>().ToArray(), a => Assert.Equal(4.0f, a));
            Assert.All(y.Data<float>().ToArray(), a => Assert.Equal(0.25f, a));

            x.reciprocal_();
            Assert.All(x.Data<float>().ToArray(), a => Assert.Equal(0.25f, a));
        }

        [Fact]
        public void OuterTest()
        {
            var x = Float32Tensor.arange(1, 5, 1);
            var y = Float32Tensor.arange(1, 4, 1);
            var expected = new float[] { 1, 2, 3, 2, 4, 6, 3, 6, 9, 4, 8, 12 };

            var res = x.outer(y);
            Assert.Equal(Float32Tensor.from(expected, 4, 3), res);
        }

        [Fact]
        public void Exp2Test()
        {
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = new float[] { 2.0f, 4.0f, 8.0f };
            var res = Float32Tensor.from(x).exp2();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void FloorTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Floor).ToArray();
            var input = Float32Tensor.from(data);
            var res = input.floor();
            Assert.True(res.allclose(Float32Tensor.from(expected)));

            input.floor_();
            Assert.True(input.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void TruncTest()
        {
            var input = Float32Tensor.randn(new long[] { 25 });
            var expected = input.Data<float>().ToArray().Select(MathF.Truncate).ToArray();
            var res = input.trunc();
            Assert.True(res.allclose(Float32Tensor.from(expected)));

            input.trunc_();
            Assert.True(input.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void CeilTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Ceiling).ToArray();
            var input = Float32Tensor.from(data);
            var res = input.ceil();
            Assert.True(res.allclose(Float32Tensor.from(expected)));

            input.ceil_();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void RoundTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(x => MathF.Round(x)).ToArray();
            var input = Float32Tensor.from(data);
            var res = input.round();
            Assert.True(res.allclose(Float32Tensor.from(expected)));

            input.round_();
            Assert.True(res.allclose(Float32Tensor.from(expected)));
        }

        [Fact]
        public void BucketizeTest()
        {
            var boundaries = Int32Tensor.from(new int[] { 1, 3, 5, 7, 9 });
            var tensor = Int32Tensor.from(new int[] { 3, 6, 9, 3, 6, 9 }, 2, 3);
            {
                var res = tensor.bucketize(boundaries, true, false);
                var expected = Int32Tensor.from(new int[] { 1, 3, 4, 1, 3, 4 }, 2, 3);
                Assert.True(res.allclose(expected));
            }
            {
                var res = tensor.bucketize(boundaries, true, true);
                var expected = Int32Tensor.from(new int[] { 2, 3, 5, 2, 3, 5 }, 2, 3);
                Assert.True(res.allclose(expected));
            }
        }

        [Fact]
        public void VanderTest()
        {
            var x = Int32Tensor.from(new int[] { 1, 2, 3, 5 });
            {
                var res = x.vander();
                var expected = Int64Tensor.from(new long[] { 1, 1, 1, 1, 8, 4, 2, 1, 27, 9, 3, 1, 125, 25, 5, 1 }, 4, 4);
                Assert.Equal(expected, res);
            }
            {
                var res = x.vander(3);
                var expected = Int64Tensor.from(new long[] { 1, 1, 1, 4, 2, 1, 9, 3, 1, 25, 5, 1 }, 4, 3);
                Assert.Equal(expected, res);
            }
            {
                var res = x.vander(3, true);
                var expected = Int64Tensor.from(new long[] { 1, 1, 1, 1, 2, 4, 1, 3, 9, 1, 5, 25 }, 4, 3);
                Assert.Equal(expected, res);
            }
        }

        [Fact]
        public void ExpandTest()
        {
            Tensor ones = Float32Tensor.ones(new long[] { 2 });
            Tensor onesExpanded = ones.expand(new long[] { 3, 2 });

            Assert.Equal(onesExpanded.shape, new long[] { 3, 2 });
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    Assert.Equal(1.0, onesExpanded[i, j].ToSingle());
                }
            }
        }

        [Fact]
        public void MovedimTest()
        {
            Tensor input = Float32Tensor.randn(new long[] { 3, 2, 1 });
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
        public void CountNZTest()
        {
            Tensor input = Float32Tensor.from(new float[] { 0, 1, 1, 0, 0, 0, 0, 0, 1 }, 3, 3);
            {
                var res = input.count_nonzero();
                Assert.Equal(Int64Tensor.from(3), res);
            }
            {
                var res = input.count_nonzero(new long[] { 0 });
                Assert.Equal(Int64Tensor.from(new long[] { 0, 1, 2 }), res);
            }
        }

        [Fact]
        public void SortTest1()
        {
            var input = Float64Tensor.from(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expectedValues = Float64Tensor.from(new double[] {
                -1.2631, -1.1289, -0.1321, 0.4370,
                -2.0527, -1.1250,  0.2275, 0.3077,
                -0.5495, -0.1259, -0.0881, 1.0284
            }, 3, 4);

            var expectedIndices = Int64Tensor.from(new long[] {
                2, 3, 0, 1,
                0, 1, 2, 3,
                2, 1, 0, 3
            }, 3, 4);

            var res = input.sort();
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        public void SortTest2()
        {
            var input = Float64Tensor.from(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expectedValues = Float64Tensor.from(new double[] {
                -2.0527, -1.1250, -1.2631, -1.1289,
                -0.1321, -0.1259, -0.5495,  0.3077,
                -0.0881,  0.4370,  0.2275,  1.0284
            }, 3, 4);

            var expectedIndices = Int64Tensor.from(new long[] {
                1, 1, 0, 0,
                0, 2, 2, 1,
                2, 0, 1, 2
            }, 3, 4);

            var res = input.sort(dim: 0);
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        public void SortTest3()
        {
            var input = Float64Tensor.from(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284,
            }, 3, 4);

            var expectedValues = Float64Tensor.from(new double[] {
                0.4370, -0.1321, -1.1289, -1.2631,
                0.3077,  0.2275, -1.1250, -2.0527,
                1.0284, -0.0881, -0.1259, -0.5495,
            }, 3, 4);

            var expectedIndices = Int64Tensor.from(new long[] {
                1, 0, 3, 2,
                3, 2, 1, 0,
                3, 0, 1, 2,
            }, 3, 4);

            var res = input.sort(descending: true);
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        public void SortTest4()
        {
            var input = Float64Tensor.from(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expectedValues = Float64Tensor.from(new double[] {
                -0.0881,  0.4370,  0.2275,  1.0284,
                -0.1321, -0.1259, -0.5495,  0.3077,
                -2.0527, -1.1250, -1.2631, -1.1289,
            }, 3, 4);

            var expectedIndices = Int64Tensor.from(new long[] {
                2, 0, 1, 2,
                0, 2, 2, 1,
                1, 1, 0, 0,
            }, 3, 4);

            var res = input.sort(dim: 0, descending: true);
            Assert.True(res.Values.allclose(expectedValues));
            Assert.Equal(expectedIndices, res.Indices);
        }

        [Fact]
        public void MSortTest()
        {
            var input = Float64Tensor.from(new double[] {
                -0.1321, 0.4370, -1.2631, -1.1289,
                -2.0527, -1.1250,  0.2275,  0.3077,
                -0.0881, -0.1259, -0.5495,  1.0284
            }, 3, 4);

            var expected = Float64Tensor.from(new double[] {
                -2.0527, -1.1250, -1.2631, -1.1289,
                -0.1321, -0.1259, -0.5495,  0.3077,
                -0.0881,  0.4370,  0.2275,  1.0284
            }, 3, 4);

            var res = input.msort();
            Assert.True(res.allclose(expected));
        }

        [Fact]
        public void RepeatTest()
        {
            var input = Int32Tensor.from(new int[] { 1, 2, 3 });
            var expected = Int32Tensor.from(new int[] {
                1,  2,  3,  1,  2,  3,
                1,  2,  3,  1,  2,  3,
                1,  2,  3,  1,  2,  3,
                1,  2,  3,  1,  2,  3
            }).reshape(4, 6);

            var res = input.repeat(new long[] { 4, 2 });
            Assert.Equal(res, expected);
        }

        [Fact]
        public void FlipLRTest()
        {
            var input = Int32Tensor.from(new int[] {
                1,  2,  3,
                1,  2,  3,
                1,  2,  3,
            }).reshape(3, 3);

            var expected = Int32Tensor.from(new int[] {
                3,  2,  1,
                3,  2,  1,
                3,  2,  1,
            }).reshape(3, 3);

            var res = input.fliplr();
            Assert.Equal(res, expected);
        }

        [Fact]
        public void FlipUDTest()
        {
            var input = Int32Tensor.from(new int[] {
                1,  1,  1,
                2,  2,  2,
                3,  3,  3,
            }).reshape(3, 3);

            var expected = Int32Tensor.from(new int[] {
                3,  3,  3,
                2,  2,  2,
                1,  1,  1,
            }).reshape(3, 3);

            var res = input.flipud();
            Assert.Equal(res, expected);
        }

        [Fact]
        public void TopKTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res1 = Float32Tensor.from(data).topk(1);
            var res1_0 = res1.values[0].ToSingle();
            var index1_0 = res1.indexes[0].ToInt64();
            Assert.Equal(3.1f, res1_0);
            Assert.Equal(2L, index1_0);

            var res2 = Float32Tensor.from(data).topk(2, sorted: true);
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
        public void SumTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };

            var res1 = Float32Tensor.from(data).sum();
            var res1_0 = res1.ToSingle();
            Assert.Equal(6.0f, res1_0);

            var res2 = Float32Tensor.from(data).sum(type: ScalarType.Float64);
            var res2_0 = res2.ToDouble();
            Assert.Equal(6.0, res2_0);

            // summing integers gives long unless type is explicitly specified
            var dataInt32 = new int[] { 1, 2, 3 };
            var res3 = Int32Tensor.from(dataInt32).sum();
            Assert.Equal(ScalarType.Int64, res3.dtype);
            var res3_0 = res3.ToInt64();
            Assert.Equal(6L, res3_0);

            // summing integers gives long unless type is explicitly specified
            var res4 = Int32Tensor.from(dataInt32).sum(type: ScalarType.Int32);
            Assert.Equal(ScalarType.Int32, res4.dtype);
            var res4_0 = res4.ToInt32();
            Assert.Equal(6L, res4_0);

        }

        [Fact]
        public void UnbindTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = Float32Tensor.from(data).unbind();
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { }, res[0].shape);
            Assert.Equal(new long[] { }, res[1].shape);
            Assert.Equal(new long[] { }, res[2].shape);
            Assert.Equal(1.1f, res[0].ToSingle());
            Assert.Equal(2.0f, res[1].ToSingle());
            Assert.Equal(3.1f, res[2].ToSingle());
        }

        [Fact]
        public void SplitWithSizeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = Float32Tensor.from(data).split(2);
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
        public void SplitWithSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = Float32Tensor.from(data).split(new long[] { 2, 1 });
            Assert.Equal(2, res.Length);
            Assert.Equal(new long[] { 2 }, res[0].shape);
            Assert.Equal(new long[] { 1 }, res[1].shape);
            Assert.Equal(1.1f, res[0][0].ToSingle());
            Assert.Equal(2.0f, res[0][1].ToSingle());
            Assert.Equal(3.1f, res[1][0].ToSingle());
        }

        [Fact]
        public void TensorSplitWithSizeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = Float32Tensor.from(data).tensor_split(2);
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
        public void TensorSplitWithSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = Float32Tensor.from(data).tensor_split(new long[] { 1, 4 });
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
        public void VSplitWithSizeTest()
        {
            var a = Int32Tensor.arange(64).reshape(4, 4, 4);

            var b = a.vsplit(2);
            Assert.Equal(new long[] { 2, 4, 4 }, b[0].shape);
            Assert.Equal(new long[] { 2, 4, 4 }, b[1].shape);

            Assert.Throws<ArgumentException>(() => a.vsplit(3));
        }

        [Fact]
        public void VSplitWithSizesTest()
        {
            var a = Int32Tensor.arange(80).reshape(5, 4, 4);

            var b = a.vsplit(new long[] { 2, 3 });
            Assert.Equal(new long[] { 2, 4, 4 }, b[0].shape);
            Assert.Equal(new long[] { 1, 4, 4 }, b[1].shape);
            Assert.Equal(new long[] { 2, 4, 4 }, b[2].shape);
        }

        [Fact]
        public void HSplitWithSizeTest()
        {
            var a = Int32Tensor.arange(64).reshape(4, 4, 4);

            var b = a.hsplit(2);
            Assert.Equal(new long[] { 4, 2, 4 }, b[0].shape);
            Assert.Equal(new long[] { 4, 2, 4 }, b[1].shape);

            Assert.Throws<ArgumentException>(() => a.hsplit(3));
        }

        [Fact]
        public void HSplitWithSizesTest()
        {
            var a = Int32Tensor.arange(80).reshape(4, 5, 4);

            var b = a.hsplit(new long[] { 2, 3 });
            Assert.Equal(new long[] { 4, 2, 4 }, b[0].shape);
            Assert.Equal(new long[] { 4, 1, 4 }, b[1].shape);
            Assert.Equal(new long[] { 4, 2, 4 }, b[2].shape);
        }

        [Fact]
        public void DSplitWithSizeTest()
        {
            var a = Int32Tensor.arange(64).reshape(4, 4, 4);

            var b = a.dsplit(2);
            Assert.Equal(new long[] { 4, 4, 2 }, b[0].shape);
            Assert.Equal(new long[] { 4, 4, 2 }, b[1].shape);

            Assert.Throws<ArgumentException>(() => a.hsplit(3));
        }

        [Fact]
        public void DSplitWithSizesTest()
        {
            var a = Int32Tensor.arange(80).reshape(4, 4, 5);

            var b = a.dsplit(new long[] { 2, 3 });
            Assert.Equal(new long[] { 4, 4, 2 }, b[0].shape);
            Assert.Equal(new long[] { 4, 4, 1 }, b[1].shape);
            Assert.Equal(new long[] { 4, 4, 2 }, b[2].shape);
        }

        [Fact]
        public void TensorSplitWithTensorSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = Float32Tensor.from(data).tensor_split(Int64Tensor.from(new long[] { 1, 4 }));
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
        public void TensorChunkTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.2f, 5.3f };

            var res = Float32Tensor.from(data).chunk(2);
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
        public void TensorNonZeroTest()
        {
            var data = new double[] { 0.6, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 1.2, 0.5 };
            using (var t = Float64Tensor.from(data, 3, 4)) {

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
        }

        [Fact]
        public void TakeAlongTest()
        {
            var t = Int32Tensor.from(new int[] { 10, 30, 20, 60, 40, 50 }).reshape(2, 3);
            var max_idx = t.argmax();
            var sort_idx = t.argsort(dimension: 1);

            var x = t.take_along_dim(max_idx);
            var y = t.take_along_dim(sort_idx, dimension: 1);

            Assert.Equal(60, x.DataItem<int>());
            Assert.Equal(new int[] { 10, 20, 30, 40, 50, 60 }, y.Data<int>().ToArray());
        }

        [Fact]
        public void RandomTest()
        {
            var res = Float32Tensor.rand(new long[] { 2 });
            Assert.Equal(new long[] { 2 }, res.shape);

            var res1 = Int16Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res1.shape);

            var res2 = Int32Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res2.shape);

            var res3 = Int64Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res3.shape);

            var res4 = ByteTensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res4.shape);

            var res5 = Int8Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res5.shape);

            var res6 = Float16Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res6.shape);

            var res7 = BFloat16Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res7.shape);

            var res8 = Float32Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res8.shape);

            var res9 = Float64Tensor.randint(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res9.shape);

            //var res7 = ComplexFloat16Tensor.randint(10, new long[] { 200 });
            //Assert.Equal(new long[] { 200 }, res7.Shape);

            var res10 = ComplexFloat32Tensor.randint(100, new long[] { 20, 10 });
            Assert.Equal(new long[] { 20, 10 }, res10.shape);

            var res11 = ComplexFloat64Tensor.randint(10, new long[] { 20, 10 });
            Assert.Equal(new long[] { 20, 10 }, res11.shape);
        }

        [Fact]
        public void SqueezeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            using (var res = Float32Tensor.from(data).expand(new long[] { 1, 1, 3 }).squeeze(0).squeeze(0)) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
            // Test negative dims, too.
            using (var res = Float32Tensor.from(data).expand(new long[] { 1, 1, 3 }).squeeze(-3).squeeze(0)) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
            // And all dims.
            using (var res = Float32Tensor.from(data).expand(new long[] { 1, 1, 3 }).squeeze()) {
                Assert.Equal(new long[] { 3 }, res.shape);
                Assert.Equal(1.1f, res[0].ToSingle());
                Assert.Equal(2.0f, res[1].ToSingle());
                Assert.Equal(3.1f, res[2].ToSingle());
            }
        }

        [Fact]
        public void NarrowTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = Float32Tensor.from(data).narrow(0, 1, 2);
            Assert.Equal(new long[] { 2 }, res.shape);
            Assert.Equal(2.0f, res[0].ToSingle());
            Assert.Equal(3.1f, res[1].ToSingle());
        }

        [Fact]
        public void SliceTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.0f };

            var res = Float32Tensor.from(data).slice(0, 1, 1, 1);
            Assert.Equal(new long[] { 0 }, res.shape);

            var res2 = Float32Tensor.from(data).slice(0, 1, 2, 1);
            Assert.Equal(new long[] { 1 }, res2.shape);
            Assert.Equal(2.0f, res2[0].ToSingle());

            var res3 = Float32Tensor.from(data).slice(0, 1, 4, 2);
            Assert.Equal(new long[] { 2 }, res3.shape);
            Assert.Equal(2.0f, res3[0].ToSingle());
            Assert.Equal(4.0f, res3[1].ToSingle());
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
            var t1t = Float32Tensor.from(t1raw, new long[] { 3, 4, 5 });
            var t2t = Float32Tensor.from(t2raw, new long[] { 2, 4, 3 });

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
            var t1t = Float32Tensor.from(t1raw, new long[] { 2, 3, 4 });
            var t2t = Float32Tensor.from(t2raw, new long[] { 2, 3, 2 });

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
            var t1t = Float32Tensor.from(t1raw, new long[] { 3, 4, 5 }); //.cuda();
            var t2t = Float32Tensor.from(t2raw, new long[] { 2, 4, 3 }); //.cuda();

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
                var data = t3p2d3.Data<float>();
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
        public void CholeskyTest()
        {
            var a = Float64Tensor.randn(new long[] { 3, 2, 2 });
            a = a.matmul(a.swapdims(-2, -1));   // Worked this in to get it tested. Alias for 'transpose'
            var l = linalg.cholesky(a);

            Assert.True(a.allclose(l.matmul(l.swapaxes(-2, -1)))); // Worked this in to get it tested. Alias for 'transpose'
        }

        [Fact]
        public void QRTest()
        {
            var a = Float32Tensor.randn(new long[] { 4, 25, 25 });

            var l = linalg.qr(a);

            Assert.Equal(a.shape, l.Q.shape);
            Assert.Equal(a.shape, l.R.shape);
        }

        [Fact]
        public void SVDTest()
        {
            var a = Float32Tensor.randn(new long[] { 4, 25, 15 });

            var l = linalg.svd(a);

            Assert.Equal(new long[] { 4, 25, 25 }, l.U.shape);
            Assert.Equal(new long[] { 4, 15 }, l.S.shape);
            Assert.Equal(new long[] { 4, 15, 15 }, l.Vh.shape);

            l = linalg.svd(a, fullMatrices: false);

            Assert.Equal(a.shape, l.U.shape);
            Assert.Equal(new long[] { 4, 15 }, l.S.shape);
            Assert.Equal(new long[] { 4, 15, 15 }, l.Vh.shape);
        }


        [Fact]
        public void SVDValsTest()
        {
            var a = Float64Tensor.from(new double[] { -1.3490, -0.1723, 0.7730,
                -1.6118, -0.3385, -0.6490,
                 0.0908, 2.0704, 0.5647,
                -0.6451, 0.1911, 0.7353,
                 0.5247, 0.5160, 0.5110}, 5, 3);

            var l = linalg.svdvals(a);
            Assert.True(l.allclose(Float64Tensor.from(new double[] { 2.5138929972840613, 2.1086555338402455, 1.1064930672223237 }), rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void LSTSQTest()
        {
            var a = Float32Tensor.randn(new long[] { 4, 25, 15 });
            var b = Float32Tensor.randn(new long[] { 4, 25, 10 });

            var l = linalg.lstsq(a, b);

            Assert.Equal(new long[] { 4, 15, 10 }, l.Solution.shape);
            Assert.Equal(0, l.Residuals.shape[0]);
            Assert.Equal(new long[] { 4 }, l.Rank.shape);
            Assert.Equal(new long[] { 4, 15, 10 }, l.Solution.shape);
            Assert.Equal(0, l.SingularValues.shape[0]);
        }

        [Fact]
        public void MatrixPowerTest()
        {
            var a = Float32Tensor.randn(new long[] { 25, 25 });
            var b = a.matrix_power(3);
            Assert.Equal(new long[] { 25, 25 }, b.shape);
        }

        [Fact]
        public void MultiDotTest()
        {
            var a = Float32Tensor.randn(new long[] { 25, 25 });
            var b = Float32Tensor.randn(new long[] { 25, 25 });
            var c = Float32Tensor.randn(new long[] { 25, 25 });
            var d = torch.linalg.multi_dot(new Tensor[] { a, b, c });
            Assert.Equal(new long[] { 25, 25 }, d.shape);
        }

        [Fact]
        public void DeterminantTest()
        {
            {
                var a = Float32Tensor.from(
                    new float[] { 0.9478f, 0.9158f, -1.1295f,
                                  0.9701f, 0.7346f, -1.8044f,
                                 -0.2337f, 0.0557f, 0.6929f }, 3, 3);
                var l = linalg.det(a);
                Assert.True(l.allclose(Float32Tensor.from(0.09335048f)));
            }
            {
                var a = Float32Tensor.from(
                    new float[] { 0.9254f, -0.6213f, -0.5787f, 1.6843f, 0.3242f, -0.9665f,
                                  0.4539f, -0.0887f, 1.1336f, -0.4025f, -0.7089f, 0.9032f }, 3, 2, 2);
                var l = linalg.det(a);
                Assert.True(l.allclose(Float32Tensor.from(new float[] { 1.19910491f, 0.4099378f, 0.7385352f })));
            }
        }


        [Fact]
        public void MatrixNormTest()
        {
            {
                var a = Float32Tensor.arange(9).view(3, 3);

                var b = linalg.matrix_norm(a);
                var c = linalg.matrix_norm(a, ord: -1);

                Assert.Equal(14.282857f, b.DataItem<float>());
                Assert.Equal(9.0f, c.DataItem<float>());
            }
        }


        [Fact]
        public void VectorNormTest()
        {
            {
                var a = Float32Tensor.from(
                    new float[] { -4.0f, -3.0f, -2.0f, -1.0f, 0, 1.0f, 2.0f, 3.0f, 4.0f });

                var b = linalg.vector_norm(a, ord: 3.5);
                var c = linalg.vector_norm(a.view(3, 3), ord: 3.5);

                Assert.Equal(5.4344883f, b.DataItem<float>());
                Assert.Equal(5.4344883f, c.DataItem<float>());
            }
        }

        [Fact]
        public void EighvalsTest32()
        {
            {
                var a = Float32Tensor.from(
                    new float[] { 2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f, -2.7457f, -1.7517f, 1.7166f }, 3, 3);
                var expected = ComplexFloat32Tensor.from(
                    new (float, float)[] { (3.44288778f, 0.0f), (2.17609453f, 0.0f), (-2.128083f, 0.0f) });
                var l = linalg.eigvals(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        public void EighvalsTest64()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var a = Float64Tensor.from(
                    new double[] { 2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f, -2.7457f, -1.7517f, 1.7166f }, 3, 3);
                var expected = ComplexFloat64Tensor.from(
                    new System.Numerics.Complex[] { new System.Numerics.Complex(3.44288778f, 0.0f), new System.Numerics.Complex(2.17609453f, 0.0f), new System.Numerics.Complex(-2.128083f, 0.0f) });
                var l = linalg.eigvals(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        public void EighvalshTest32()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var a = Float32Tensor.from(
                    new float[] {  2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f,
                                  -2.7457f, -1.7517f, 1.7166f,  2.2207f, 2.2207f, -2.0898f }, 3, 2, 2);
                var expected = Float32Tensor.from(
                    new float[] { 2.5797f, 3.46290016f, -4.16046524f, 1.37806475f, -3.11126733f, 2.73806715f }, 3, 2);
                var l = linalg.eigvalsh(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        public void EighvalshTest64()
        {
            {
                var a = Float64Tensor.from(
                    new double[] {  2.8050, -0.3850, -0.3850, 3.2376, -1.0307, -2.7457,
                                  -2.7457, -1.7517, 1.7166,  2.2207, 2.2207, -2.0898 }, 3, 2, 2);
                var expected = Float64Tensor.from(
                    new double[] { 2.5797, 3.46290016, -4.16046524, 1.37806475, -3.11126733, 2.73806715 }, 3, 2);
                var l = linalg.eigvalsh(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        public void LinalgNormTest()
        {
            {
                var a = Float32Tensor.from(
                    new float[] { -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f });
                var b = a.reshape(3, 3);

                Assert.True(linalg.norm(a).allclose(Float32Tensor.from(7.7460f)));
                Assert.True(linalg.norm(b).allclose(Float32Tensor.from(7.7460f)));
                Assert.True(linalg.norm(b, "fro").allclose(Float32Tensor.from(7.7460f)));

                Assert.True(linalg.norm(a, float.PositiveInfinity).allclose(Float32Tensor.from(4.0f)));
                Assert.True(linalg.norm(b, float.PositiveInfinity).allclose(Float32Tensor.from(9.0f)));
                Assert.True(linalg.norm(a, float.NegativeInfinity).allclose(Float32Tensor.from(0.0f)));
                Assert.True(linalg.norm(b, float.NegativeInfinity).allclose(Float32Tensor.from(2.0f)));

                Assert.True(linalg.norm(a, 1).allclose(Float32Tensor.from(20.0f)));
                Assert.True(linalg.norm(b, 1).allclose(Float32Tensor.from(7.0f)));
                Assert.True(linalg.norm(a, -1).allclose(Float32Tensor.from(0.0f)));
                Assert.True(linalg.norm(b, -1).allclose(Float32Tensor.from(6.0f)));

                Assert.True(linalg.norm(a, 2).allclose(Float32Tensor.from(7.7460f)));
                Assert.True(linalg.norm(b, 2).allclose(Float32Tensor.from(7.3485f)));
                Assert.True(linalg.norm(a, 3).allclose(Float32Tensor.from(5.8480f)));
                Assert.True(linalg.norm(a, -2).allclose(Float32Tensor.from(0.0f)));
                Assert.True(linalg.norm(a, -3).allclose(Float32Tensor.from(0.0f)));
            }
        }

        [Fact]
        public void TestSpecialEntropy()
        {
            var a = Float32Tensor.from(new float[] { -0.5f, 1.0f, 0.5f });
            var expected = Float32Tensor.from(
                    new float[] { Single.NegativeInfinity, 0.0f, 0.3465736f });
            var b = torch.special.entr(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        public void TestSpecialErrorFunction()
        {
            var a = Float32Tensor.from(new float[] { 0.0f, -1.0f, 10.0f });
            var expected = Float32Tensor.from(
                    new float[] { 0.0f, -0.8427f, 1.0f });
            var b = torch.special.erf(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        public void TestSpecialComplementaryErrorFunction()
        {
            var a = Float32Tensor.from(new float[] { 0.0f, -1.0f, 10.0f });
            var expected = Float32Tensor.from(
                    new float[] { 1.0f, 1.8427f, 0.0f });
            var b = torch.special.erfc(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        public void TestSpecialInverseErrorFunction()
        {
            var a = Float32Tensor.from(new float[] { 0.0f, 0.5f, -1.0f });
            var expected = Float32Tensor.from(
                    new float[] { 0.0f, 0.476936281f, Single.NegativeInfinity });
            var b = torch.special.erfinv(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        public void TestSpecialExpit()
        {
            var a = Float32Tensor.randn(new long[] { 10 });
            var expected = Float32Tensor.from(a.Data<float>().ToArray().Select(x => 1.0f / (1.0f + MathF.Exp(-x))).ToArray());
            var b = torch.special.expit(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void TestSpecialExpm1()
        {
            var a = Float32Tensor.randn(new long[] { 10 });
            var expected = Float32Tensor.from(a.Data<float>().ToArray().Select(x => MathF.Exp(x) - 1.0f).ToArray());
            var b = torch.special.expm1(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void TestSpecialExp2()
        {
            var a = Float32Tensor.randn(new long[] { 10 });
            var expected = Float32Tensor.from(a.Data<float>().ToArray().Select(x => MathF.Pow(2.0f, x)).ToArray());
            var b = torch.special.exp2(a);
            Assert.True(b.allclose(expected, rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        public void TestSpecialGammaLN()
        {
            var a = Float32Tensor.arange(0.5f, 2f, 0.5f);
            var expected = Float32Tensor.from(new float[] { 0.5723649f, 0.0f, -0.120782226f });
            var b = torch.special.gammaln(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        public void TestSpeciali0e()
        {
            var a = Float32Tensor.arange(0.0f, 5.0f, 1.0f);
            var expected = Float32Tensor.from(new float[] { 1.0f, 0.465759635f, 0.3085083f, 0.243000358f, 0.20700191f });
            var b = torch.special.i0e(a);
            Assert.True(b.allclose(expected));
        }

        [Fact]
        public void NanToNumTest()
        {
            {
                var a = Float32Tensor.from(new float[] { Single.NaN, Single.PositiveInfinity, Single.NegativeInfinity, MathF.PI });

                {
                    var expected = Float32Tensor.from(new float[] { 0.0f, Single.MaxValue, Single.MinValue, MathF.PI });
                    Assert.True(a.nan_to_num().allclose(expected));
                }
                {
                    var expected = Float32Tensor.from(new float[] { 2.0f, Single.MaxValue, Single.MinValue, MathF.PI });
                    Assert.True(a.nan_to_num(nan: 2.0f).allclose(expected));
                }
                {
                    var expected = Float32Tensor.from(new float[] { 2.0f, 3.0f, Single.MinValue, MathF.PI });
                    Assert.True(a.nan_to_num(nan: 2.0f, posinf: 3.0f).allclose(expected));
                }
                {
                    var expected = Float32Tensor.from(new float[] { 2.0f, 3.0f, -13.0f, MathF.PI });
                    Assert.True(a.nan_to_num(nan: 2.0f, posinf: 3.0f, neginf: -13.0f).allclose(expected));
                }
            }
        }

        [Fact]
        public void TensorDiffTest()
        {
            var a = Float32Tensor.from(new float[] { 1, 3, 2 });
            Assert.True(a.diff().allclose(Float32Tensor.from(new float[] { 2, -1 })));
            var b = Float32Tensor.from(new float[] { 4, 5 });
            Assert.True(a.diff(append: b).allclose(Float32Tensor.from(new float[] { 2, -1, 2, 1 })));
            var c = Float32Tensor.from(new float[] { 1, 2, 3, 3, 4, 5 }, 2, 3);
            Assert.True(c.diff(dim: 0).allclose(Float32Tensor.from(new float[] { 2, 2, 2 }, 1, 3)));
            Assert.True(c.diff(dim: 1).allclose(Float32Tensor.from(new float[] { 1, 1, 1, 1 }, 2, 2)));
        }

        [Fact]
        public void RavelTest()
        {
            var expected = Int32Tensor.from(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var a = expected.view(2, 2, 2);
            Assert.Equal(new long[] { 2, 2, 2 }, a.shape);
            Assert.Equal(expected, a.ravel());
        }

        [Fact]
        public void StrideTest_1()
        {
            var x = Int32Tensor.zeros(new long[] { 2, 2, 2 });
            Assert.Equal(4, x.stride(0));
            Assert.Equal(2, x.stride(1));
            Assert.Equal(1, x.stride(2));

            Assert.Equal(new long[] { 4, 2, 1 }, x.stride());
        }

        [Fact]
        public void StrideTest_2()
        {
            var x = Int32Tensor.zeros(new long[] { 2, 2, 2 });
            Assert.Equal(new long[] { 4, 2, 1 }, x.stride());
        }

        [Fact]
        public void Complex32PartsTest()
        {
            var x = ComplexFloat32Tensor.zeros(new long[] { 20 });
            var r1 = x.Real;
            var i1 = x.Imag;

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
        public void Complex64PartsTest()
        {
            var x = ComplexFloat64Tensor.zeros(new long[] { 20 });
            var r1 = x.Real;
            var i1 = x.Imag;

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
        public void Complex32TensorView()
        {
            var x = ComplexFloat32Tensor.zeros(new long[] { 20, 20, 20 });
            var y = Float32Tensor.zeros(new long[] { 20, 20, 20, 2 });

            var vasr = x.view_as_real();
            Assert.Equal(new long[] { 20, 20, 20, 2 }, vasr.shape);
            Assert.Equal(ScalarType.Float32, vasr.dtype);

            var vasc = y.view_as_complex();
            Assert.Equal(ScalarType.ComplexFloat32, vasc.dtype);
            Assert.Equal(x.shape, vasc.shape);

        }

        [Fact]
        public void Complex64TensorView()
        {
            var x = ComplexFloat64Tensor.zeros(new long[] { 20, 20, 20 });
            var y = Float64Tensor.zeros(new long[] { 20, 20, 20, 2 });

            var vasr = x.view_as_real();
            Assert.Equal(new long[] { 20, 20, 20, 2 }, vasr.shape);
            Assert.Equal(ScalarType.Float64, vasr.dtype);

            var vasc = y.view_as_complex();
            Assert.Equal(ScalarType.ComplexFloat64, vasc.dtype);
            Assert.Equal(x.shape, vasc.shape);

        }

        [Fact]
        public void Float32FFT()
        {
            var input = Float32Tensor.arange(4);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        public void Float64FFT()
        {
            var input = Float64Tensor.arange(4);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32HFFT()
        {
            var input = Float32Tensor.arange(4);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float32, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        public void Float64HFFT()
        {
            var input = Float64Tensor.arange(4);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float64, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32RFFT()
        {
            var input = Float32Tensor.arange(4);
            var output = fft.rfft(input);
            Assert.Equal(3, output.shape[0]);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.irfft(output);
            Assert.Equal(ScalarType.Float32, inverted.dtype);
        }

        [Fact]
        public void Float64RFFT()
        {
            var input = Float64Tensor.arange(4);
            var output = fft.rfft(input);
            Assert.Equal(3, output.shape[0]);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.irfft(output);
            Assert.Equal(ScalarType.Float64, inverted.dtype);
        }

        [Fact]
        public void ComplexFloat32FFT()
        {
            var input = ComplexFloat32Tensor.arange(4);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }


        [Fact]
        public void ComplexFloat64FFT()
        {
            var input = ComplexFloat64Tensor.arange(4);
            var output = fft.fft_(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void ComplexFloat32HFFT()
        {
            var input = ComplexFloat32Tensor.arange(4);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float32, output.dtype);

            var inverted = fft.ihfft(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }


        [Fact]
        public void ComplexFloat64HFFT()
        {
            var input = ComplexFloat64Tensor.arange(4);
            var output = fft.hfft(input);
            Assert.Equal(6, output.shape[0]);
            Assert.Equal(ScalarType.Float64, output.dtype);

            var inverted = fft.ihfft(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32FFT2()
        {
            var input = Float32Tensor.rand(new long[] { 5, 5 });
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        public void Float64FFT2()
        {
            var input = Float64Tensor.rand(new long[] { 5, 5 });
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32RFFT2()
        {
            var input = Float32Tensor.rand(new long[] { 5, 5 });
            var output = fft.rfft2(input);
            Assert.Equal(new long[] { 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.irfft2(output);
            Assert.Equal(ScalarType.Float32, inverted.dtype);
        }

        [Fact]
        public void Float64RFFT2()
        {
            var input = Float64Tensor.rand(new long[] { 5, 5 });
            var output = fft.rfft2(input);
            Assert.Equal(new long[] { 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.irfft2(output);
            Assert.Equal(ScalarType.Float64, inverted.dtype);
        }

        [Fact]
        public void ComplexFloat32FFT2()
        {
            var input = ComplexFloat32Tensor.rand(new long[] { 5, 5 });
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        public void ComplexFloat64FFT2()
        {
            var input = ComplexFloat64Tensor.rand(new long[] { 5, 5 });
            var output = fft.fft2(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifft2(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32FFTN()
        {
            var input = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        public void Float64FFTN()
        {
            var input = Float64Tensor.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32RFFTN()
        {
            var input = Float32Tensor.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.rfftn(input);
            Assert.Equal(new long[] { 5, 5, 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.irfftn(output);
            Assert.Equal(ScalarType.Float32, inverted.dtype);
        }

        [Fact]
        public void Float64RFFTN()
        {
            var input = Float64Tensor.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.rfftn(input);
            Assert.Equal(new long[] { 5, 5, 5, 3 }, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.irfftn(output);
            Assert.Equal(ScalarType.Float64, inverted.dtype);
        }

        [Fact]
        public void ComplexFloat32FFTN()
        {
            var input = ComplexFloat32Tensor.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat32, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat32, inverted.dtype);
        }

        [Fact]
        public void ComplexFloat64FFTN()
        {
            var input = ComplexFloat64Tensor.rand(new long[] { 5, 5, 5, 5 });
            var output = fft.fftn(input);
            Assert.Equal(input.shape, output.shape);
            Assert.Equal(ScalarType.ComplexFloat64, output.dtype);

            var inverted = fft.ifftn(output);
            Assert.Equal(ScalarType.ComplexFloat64, inverted.dtype);
        }

        [Fact]
        public void Float32FFTFrequency()
        {
            var x = Float32Tensor.fftfreq(5);
            Assert.Equal(ScalarType.Float32, x.dtype);
            Assert.Equal(1, x.dim());
            Assert.Equal(5, x.shape[0]);
        }

        [Fact]
        public void Float64FFTFrequency()
        {
            var x = Float64Tensor.fftfreq(5);
            Assert.Equal(ScalarType.Float64, x.dtype);
            Assert.Equal(1, x.dim());
            Assert.Equal(5, x.shape[0]);
        }

        [Fact]
        public void Float32FFTShift()
        {
            var x = Float32Tensor.fftfreq(4);
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
        public void Float64FFTShift()
        {
            var x = Float64Tensor.fftfreq(4);
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
        public void CropTensor()
        {
            {
                var input = Float32Tensor.rand(25, 25);
                var cropped = input.crop(5, 5, 15, 13);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 13; i++) {
                    // Test the diagonal only.
                    Assert.Equal(input[5 + i, 5 + i], cropped[i, i]);
                }
            }
            {
                var input = Float32Tensor.rand(3, 25, 25);
                var cropped = input.crop(5, 5, 15, 13);

                Assert.Equal(new long[] { 3, 15, 13 }, cropped.shape);
                for (int c = 0; c < 3; c++)
                    for (int i = 0; i < 13; i++) {
                        // Test the diagonal only.
                        Assert.Equal(input[c, 5 + i, 5 + i], cropped[c, i, i]);
                    }
            }
            {
                var input = Float32Tensor.rand(16, 3, 25, 25);
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
        public void CroppedTensorWithPadding()
        {
            {
                var input = Float32Tensor.rand(25, 25);
                var cropped = input.crop(-1, -1, 15, 13);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 12; i++) {
                    // Test the diagonal only, and skip the padded corner.
                    Assert.Equal(input[i, i], cropped[i + 1, i + 1]);
                }
            }
            {
                var input = Float32Tensor.rand(25, 25);
                var cropped = input.crop(11, 13, 15, 13);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 12; i++) {
                    // Test the diagonal only, and skip the padded corner.
                    Assert.Equal(input[11 + i, 13 + i], cropped[i, i]);
                }
            }
        }

        [Fact]
        public void RandomResizeCropTensor()
        {
            {
                var input = Float32Tensor.rand(4, 3, 100, 100);
                var cropped = torchvision.transforms.RandomResizedCrop(100, 0.1, 0.5).forward(input);

                Assert.Equal(new long[] { 4, 3, 100, 100 }, cropped.shape);
            }
        }

        [Fact]
        public void CenterCropTensor()
        {
            {
                var input = Float32Tensor.rand(25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).forward(input);

                Assert.Equal(new long[] { 15, 13 }, cropped.shape);
                for (int i = 0; i < 13; i++) {
                    // Test the diagonal only.
                    Assert.Equal(input[5 + i, 6 + i], cropped[i, i]);
                }
            }
            {
                var input = Float32Tensor.rand(3, 25, 25);
                var cropped = torchvision.transforms.CenterCrop(15, 13).forward(input);

                Assert.Equal(new long[] { 3, 15, 13 }, cropped.shape);
                for (int c = 0; c < 3; c++)
                    for (int i = 0; i < 13; i++) {
                        // Test the diagonal only.
                        Assert.Equal(input[c, 5 + i, 6 + i], cropped[c, i, i]);
                    }
            }
            {
                var input = Float32Tensor.rand(16, 3, 25, 25);
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
        public void ResizeTensorDown()
        {
            {
                var input = Float32Tensor.rand(16, 3, 25, 25);
                var resized = torchvision.transforms.Resize(15).forward(input);

                Assert.Equal(new long[] { 16, 3, 15, 15 }, resized.shape);
            }
            {
                var input = Int32Tensor.randint(255, new long[] { 16, 3, 25, 25 });
                var resized = torchvision.transforms.Resize(15).forward(input);

                Assert.Equal(new long[] { 16, 3, 15, 15 }, resized.shape);
            }
        }

        [Fact]
        public void ResizeTensorUp()
        {
            {
                var input = Float32Tensor.rand(16, 3, 25, 25);
                var resized = torchvision.transforms.Resize(50).forward(input);

                Assert.Equal(new long[] { 16, 3, 50, 50 }, resized.shape);
            }
            {
                var input = Int32Tensor.randint(255, new long[] { 16, 3, 25, 25 });
                var resized = torchvision.transforms.Resize(50).forward(input);

                Assert.Equal(new long[] { 16, 3, 50, 50 }, resized.shape);
            }
        }

        [Fact]
        public void GrayscaleTensor()
        {
            {
                var input = Float32Tensor.rand(16, 3, 25, 25);
                var gray = torchvision.transforms.Grayscale().forward(input);

                Assert.Equal(new long[] { 16, 1, 25, 25 }, gray.shape);
            }
            {
                var input = Float32Tensor.rand(16, 3, 25, 25);
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
        public void InvertTensor()
        {
            {
                using (var input = Float32Tensor.rand(25)) {
                    var iData = input.Data<float>().ToArray();

                    var poster = torchvision.transforms.Invert().forward(input);
                    var pData = poster.Data<float>().ToArray();

                    Assert.Equal(new long[] { 25 }, poster.shape);
                    for (int i = 0; i < poster.shape[0]; i++) {
                        Assert.Equal(1.0f - iData[i], pData[i]);
                    }
                }
            }
        }

        [Fact]
        public void PosterizeTensor()
        {
            {
                using (var input = ByteTensor.randint(255, new long[] { 16, 3, 25, 25 })) {
                    var poster = torchvision.transforms.Posterize(4).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);

                }
            }
            {
                using (var input = ByteTensor.randint(255, new long[] { 25 })) {
                    var poster = torchvision.transforms.Posterize(4).forward(input);

                    Assert.Equal(new long[] { 25 }, poster.shape);
                    Assert.All(poster.Data<byte>().ToArray(), b => Assert.Equal(0, b & 0xf));
                }
            }
        }

        [Fact]
        public void AdjustSharpnessTensor()
        {
            {
                using (var input = ByteTensor.randint(255, new long[] { 16, 3, 25, 25 })) {
                    var poster = torchvision.transforms.AdjustSharpness(1.5).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
            {
                using (var input = ByteTensor.randint(255, new long[] { 16, 3, 25, 25 })) {
                    var poster = torchvision.transforms.AdjustSharpness(0.5).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
        }

        [Fact]
        public void AdjustHueTensor()
        {
            {
                using (var input = torch.stack(new Tensor[] { torch.zeros(1, 2, 2), torch.ones(1, 2, 2), torch.zeros(1, 2, 2) }, dimension: -3)) {
                    var poster = torchvision.transforms.AdjustHue(0).forward(input);

                    var istr = input.ToString(true);
                    var pstr = poster.ToString(true);

                    Assert.Equal(new long[] { 1, 3, 2, 2 }, poster.shape);
                    Assert.True(poster.allclose(input));
                }
            }
        }

        [Fact]
        public void RotateTensor()
        {
            {
                using (var input = torch.rand(16, 3, 25, 50)) {
                    var poster = torchvision.transforms.Rotate(90).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 50 }, poster.shape);
                }
            }
        }

        [Fact]
        public void AutocontrastTensor()
        {
            {
                using (var input = ByteTensor.randint(255, new long[] { 16, 3, 25, 25 })) {
                    var poster = torchvision.transforms.AutoContrast().forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
            {
                using (var input = ByteTensor.randint(255, new long[] { 16, 3, 25, 25 })) {
                    var poster = torchvision.transforms.AutoContrast().forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
        }

        [Fact]
        public void PerspectiveTest()
        {
            {
                using (var input = torch.ones(1, 3, 8, 8)) {

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

                    var pStr = poster.ToString(true);

                    Assert.Equal(new long[] { 1, 3, 8, 8 }, poster.shape);
                }
            }
        }

        [Fact]
        public void GaussianBlurTest()
        {
            {
                using (var input = ByteTensor.randint(255, new long[] { 16, 3, 25, 25 })) {
                    var poster = torchvision.transforms.GaussianBlur(4).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);

                }
            }
            {
                using (var input = Float32Tensor.rand(16, 3, 25, 25)) {
                    // Test even-number kernel size.
                    var poster = torchvision.transforms.GaussianBlur(4).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
            {
                using (var input = Float32Tensor.rand(16, 3, 25, 25)) {
                    // Test odd-number kernel size.
                    var poster = torchvision.transforms.GaussianBlur(5).forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
            {
                using (var input = Float32Tensor.rand(16, 3, 25, 25)) {
                    var random = torchvision.transforms.Randomize(torchvision.transforms.GaussianBlur(4), 0.5);
                    var poster = random.forward(input);

                    Assert.Equal(new long[] { 16, 3, 25, 25 }, poster.shape);
                }
            }
        }

        [Fact]
        public void SolarizeTensor()
        {
            {
                using (var input = Float32Tensor.rand(25)) {
                    var poster = torchvision.transforms.Solarize(0.55).forward(input);

                    Assert.Equal(new long[] { 25 }, poster.shape);
                    Assert.All(poster.Data<float>().ToArray(), f => Assert.True(f < 0.55f));
                }
            }
        }



        [Fact]
        public void HorizontalFlipTest()
        {
            var input = Int32Tensor.from(new int[] {
                1,  2,  3,
                1,  2,  3,
                1,  2,  3,
            }).reshape(3, 3);

            var expected = Int32Tensor.from(new int[] {
                3,  2,  1,
                3,  2,  1,
                3,  2,  1,
            }).reshape(3, 3);

            var res = torchvision.transforms.HorizontalFlip().forward(input);
            Assert.Equal(res, expected);
        }

        [Fact]
        public void VerticalFlipTest()
        {
            var input = Int32Tensor.from(new int[] {
                1,  1,  1,
                2,  2,  2,
                3,  3,  3,
            }).reshape(3, 3);

            var expected = Int32Tensor.from(new int[] {
                3,  3,  3,
                2,  2,  2,
                1,  1,  1,
            }).reshape(3, 3);

            var res = torchvision.transforms.VerticalFlip().forward(input);
            Assert.Equal(res, expected);
        }
    }
}
