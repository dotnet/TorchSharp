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
    [Collection("Sequential")]
    public class PointwiseTensorMath
    {
        [Fact]
        [TestOf(nameof(Tensor))]
        public void TestArithmeticOperatorsFloat16()
        {
            // Float16 arange_cuda not available on cuda in LibTorch 1.8.0
            // Float16 arange_cpu not available on cuda in LibTorch 1.8.0
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = torch.ones(new long[] { 10, 10 }, float16, device: device);
                    var c2 = torch.ones(new long[] { 10, 10 }, float16, device: device);
                    var c3 = torch.ones(new long[] { 10, 10 }, float16, device: device);
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
        [TestOf(nameof(Tensor))]
        public void TestArithmeticOperatorsBFloat16()
        {
            // BFloat16 arange_cuda not available on cuda in LibTorch 1.8.0
            // BFloat16 arange_cpu not available on cuda in LibTorch 1.8.0
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = torch.ones(new long[] { 10, 10 }, bfloat16, device: device);
                    var c2 = torch.ones(new long[] { 10, 10 }, bfloat16, device: device);
                    var c3 = torch.ones(new long[] { 10, 10 }, bfloat16, device: device);
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
        [TestOf(nameof(Tensor))]
        public void TestArithmeticOperatorsFloat32()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = torch.arange(0, 10, float32, device: device).expand(new long[] { 10, 10 });
                    var c2 = torch.arange(10, 0, -1, float32, device: device).expand(new long[] { 10, 10 });
                    var c3 = torch.ones(new long[] { 10, 10 }, float32, device: device);
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
        [TestOf(nameof(Tensor))]
        public void TestArithmeticOperatorsFloat64()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = torch.arange(0, 10, float64, device: device).expand(new long[] { 10, 10 });
                    var c2 = torch.arange(10, 0, -1, float64, device: device).expand(new long[] { 10, 10 });
                    var c3 = torch.ones(new long[] { 10, 10 }, float64, device: device);
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
        [TestOf(nameof(Tensor))]
        public void TestArithmeticOperatorsComplexFloat64()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = torch.arange(0, 10, complex128, device: device).expand(new long[] { 10, 10 });
                    var c2 = torch.arange(10, 0, -1, complex128, device: device).expand(new long[] { 10, 10 });
                    var c3 = torch.ones(new long[] { 10, 10 }, complex128, device: device);
                    Func<Tensor, long, long, System.Numerics.Complex> getFunc = (tt, i, j) => tt[i, j].ToComplexFloat64();
                    // scalar-tensor operators
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a + 0.5, a => a + 0.5);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => 0.5 + a, a => 0.5 + a);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a - 0.5, a => a - 0.5);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a * 0.5, a => a * 0.5);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => 0.5 * a, a => 0.5 * a);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a / 0.5, a => a / 0.5);

                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a.add(0.5), a => a + 0.5);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a.sub(0.5), a => a - 0.5);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a.mul(0.5), a => a * 0.5);
                    TestOneTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, getFunc, getFunc, a => a.div(0.5), a => a / 0.5);

                    TestOneTensorInPlace<System.Numerics.Complex>(c1, c2, getFunc, a => a.add_(0.5), a => a + 0.5);
                    TestOneTensorInPlace<System.Numerics.Complex>(c1, c2, getFunc, a => a.sub_(0.5), a => a - 0.5);
                    TestOneTensorInPlace<System.Numerics.Complex>(c1, c2, getFunc, a => a.mul_(0.5), a => a * 0.5);
                    TestOneTensorInPlace<System.Numerics.Complex>(c1, c2, getFunc, a => a.div_(0.5), a => a / 0.5);

                    // tensor-tensor operators
                    TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a + b, (a, b) => a + b);
                    TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a - b, (a, b) => a - b);
                    TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a * b, (a, b) => a * b);
                    // Rounding errors make this test volatile
                    //TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a / b, (a, b) => a / b);

                    TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a.add(b), (a, b) => a + b);
                    TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a.sub(b), (a, b) => a - b);
                    TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a.mul(b), (a, b) => a * b);
                    // Rounding errors make this test volatile
                    //TestTwoTensor<System.Numerics.Complex, System.Numerics.Complex>(c1, c2, c3, getFunc, getFunc, (a, b) => a.div(b), (a, b) => a / b);

                    TestTwoTensorInPlace<System.Numerics.Complex>(c1, c2, c3, getFunc, (a, b) => a.add_(b), (a, b) => a + b);
                    TestTwoTensorInPlace<System.Numerics.Complex>(c1, c2, c3, getFunc, (a, b) => a.sub_(b), (a, b) => a - b);
                    TestTwoTensorInPlace<System.Numerics.Complex>(c1, c2, c3, getFunc, (a, b) => a.mul_(b), (a, b) => a * b);
                    // Rounding errors make this test volatile
                    //TestTwoTensorInPlace<System.Numerics.Complex>(c1, c2, c3, getFunc, (a, b) => a.div_(b), (a, b) => a / b);
                }
            }
        }

        [Fact]
        [TestOf(nameof(Tensor))]
        public void TestComparisonOperatorsFloat32()
        {
            foreach (var device in new Device[] { torch.CPU, torch.CUDA }) {
                if (device.type != DeviceType.CUDA || torch.cuda.is_available()) {
                    var c1 = torch.arange(0, 10, float32, device: device).expand(new long[] { 10, 10 });
                    var c2 = torch.arange(10, 0, -1, float32, device: device).expand(new long[] { 10, 10 });
                    var c3 = torch.ones(new long[] { 10, 10 }, float32, device: device);
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
            Func<Tin, Tin, Tin> scalarFunc) where Tin : unmanaged
        {

            var x = c1 * c3;
            var xClone = x.clone();
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            if (x.device_type == DeviceType.CPU) {
                var xData = x.data<Tin>();
                var yData = y.data<Tin>();
                var zData = z.data<Tin>();

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
        [TestOf(nameof(Tensor.eq))]
        [TestOf(nameof(Tensor.ne))]
        [TestOf(nameof(Tensor.lt))]
        [TestOf(nameof(Tensor.gt))]
        [TestOf(nameof(Tensor.le))]
        public void TestComparison()
        {
            var A = torch.tensor(new float[] { 1.2f, 3.4f, 1.4f, 3.3f }).reshape(2, 2);
            var B = torch.tensor(new float[] { 1.3f, 3.3f });
            Assert.Equal(new bool[] { false, false, false, true }, A.eq(B).data<bool>().ToArray());
            Assert.Equal(new bool[] { false, false, false, true }, torch.eq(A, B).data<bool>().ToArray());
            Assert.Equal(new bool[] { true, true, true, false }, A.ne(B).data<bool>().ToArray());
            Assert.Equal(new bool[] { true, true, true, false }, torch.ne(A, B).data<bool>().ToArray());
            Assert.Equal(new bool[] { true, false, false, false }, A.lt(B).data<bool>().ToArray());
            Assert.Equal(new bool[] { true, false, false, false }, torch.lt(A, B).data<bool>().ToArray());
            Assert.Equal(new bool[] { true, false, false, true }, A.le(B).data<bool>().ToArray());
            Assert.Equal(new bool[] { true, false, false, true }, torch.le(A, B).data<bool>().ToArray());
            Assert.Equal(new bool[] { false, true, true, false }, A.gt(B).data<bool>().ToArray());
            Assert.Equal(new bool[] { false, true, true, false }, torch.gt(A, B).data<bool>().ToArray());
            Assert.Equal(new bool[] { false, true, true, true }, A.ge(B).data<bool>().ToArray());
            Assert.Equal(new bool[] { false, true, true, true }, torch.ge(A, B).data<bool>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.frexp))]
        public void TestFrexp()
        {
            var x = torch.arange(9, float32);
            var r = x.frexp();

            Assert.Equal(new float[] { 0.0000f, 0.5000f, 0.5000f, 0.7500f, 0.5000f, 0.6250f, 0.7500f, 0.8750f, 0.5000f }, r.Mantissa.data<float>().ToArray());
            Assert.Equal(new int[] { 0, 1, 2, 2, 3, 3, 3, 3, 4 }, r.Exponent.data<int>().ToArray());
        }

        [Fact]
        [TestOf(nameof(Tensor.deg2rad))]
        public void Deg2RadTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(angl => (angl * MathF.PI) / 180.0f).ToArray();
            var res = torch.tensor(data).deg2rad();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.clamp))]
        public void ClampTest1()
        {
            var data = torch.rand(3, 3, 3) * 10;
            var cl = data.clamp(1, 5);

            Assert.All(cl.data<float>().ToArray(), d => Assert.True(d >= 1.0f && d <= 5.0f));
        }

        [Fact]
        [TestOf(nameof(Tensor.clamp))]
        public void ClampTest2()
        {
            var data = torch.rand(3, 3, 3) * 10;
            var cl = data.clamp(torch.ones(3, 3, 3), torch.ones(3, 3, 3) * 5);

            Assert.All(cl.data<float>().ToArray(), d => Assert.True(d >= 1.0f && d <= 5.0f));
        }

        [Fact]
        [TestOf(nameof(Tensor.clamp))]
        public void ClampTest3()
        {
            var data = torch.rand(3, 3, 3) * 10;
            var cl = torch.clamp(data, 1, 5);

            Assert.All(cl.data<float>().ToArray(), d => Assert.True(d >= 1.0f && d <= 5.0f));
        }

        [Fact]
        [TestOf(nameof(Tensor.clamp))]
        public void ClampTest4()
        {
            var data = torch.rand(3, 3, 3) * 10;
            var cl = torch.clamp(data, torch.ones(3, 3, 3), torch.ones(3, 3, 3) * 5);

            Assert.All(cl.data<float>().ToArray(), d => Assert.True(d >= 1.0f && d <= 5.0f));
        }

        [Fact]
        [TestOf(nameof(Tensor.rad2deg))]
        public void Rad2DegTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(angl => (angl * 180.0f) / MathF.PI).ToArray();
            var res = torch.tensor(data).rad2deg();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.abs))]
        public void AbsTest()
        {
            var data = torch.arange(-10.0f, 10.0f, 1.0f);
            var expected = data.data<float>().ToArray().Select(MathF.Abs).ToArray();
            var res = data.abs();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.abs))]
        public void AbsTestC32()
        {
            var data = torch.rand(new long[] { 25 }, complex64);
            var expected = data.data<(float R, float I)>().ToArray().Select(c => MathF.Sqrt(c.R * c.R + c.I * c.I)).ToArray();
            var res = data.abs();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.abs))]
        public void AbsTestC64()
        {
            var data = torch.rand(new long[] { 25 }, complex128);
            var expected = data.data<System.Numerics.Complex>().ToArray().Select(c => Math.Sqrt(c.Real * c.Real + c.Imaginary * c.Imaginary)).ToArray<double>();
            var res = data.abs();
            Assert.True(res.allclose(torch.tensor(expected, float64)));
        }

        [Fact]
        [TestOf(nameof(Tensor.angle))]
        public void AngleTestC32()
        {
            var data = torch.randn(new long[] { 25 }, complex64);
            var expected = data.data<(float R, float I)>().ToArray().Select(c => {
                var x = c.R;
                var y = c.I;
                return (x > 0 || y != 0) ? 2 * MathF.Atan(y / (MathF.Sqrt(x * x + y * y) + x)) : (x < 0 && y == 0) ? MathF.PI : 0;
            }).ToArray();
            var res = data.angle();
            Assert.True(res.allclose(torch.tensor(expected), rtol: 1e-03, atol: 1e-05));
        }

        [Fact]
        [TestOf(nameof(Tensor.angle))]
        public void AngleTestC64()
        {
            var data = torch.randn(new long[] { 25 }, complex128);
            var expected = data.data<System.Numerics.Complex>().ToArray().Select(c => {
                var x = c.Real;
                var y = c.Imaginary;
                return (x > 0 || y != 0) ? 2 * Math.Atan(y / (Math.Sqrt(x * x + y * y) + x)) : (x < 0 && y == 0) ? Math.PI : 0;
            }).ToArray<double>();
            var res = data.angle();
            Assert.True(res.allclose(torch.tensor(expected, float64), rtol: 1e-03, atol: 1e-05));
        }

        [Fact]
        [TestOf(nameof(Tensor.sqrt))]
        public void SqrtTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sqrt).ToArray();
            var res = torch.tensor(data).sqrt();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.sin))]
        public void SinTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sin).ToArray();
            var res = torch.tensor(data).sin();
            Assert.True(res.allclose(torch.tensor(expected)));
            res = torch.sin(torch.tensor(data));
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.cos))]
        public void CosTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cos).ToArray();
            var res = torch.tensor(data).cos();
            Assert.True(res.allclose(torch.tensor(expected)));
            res = torch.cos(torch.tensor(data));
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.i0))]
        public void I0Test()
        {
            var data = torch.arange(0, 5, 1, float32);
            var expected = new float[] { 0.99999994f, 1.266066f, 2.27958512f, 4.88079262f, 11.3019209f };
            var res = data.i0();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.hypot))]
        public void HypotTest()
        {
            var a = new float[] { 1.0f, 2.0f, 3.0f };
            var b = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = a.Select(x => MathF.Sqrt(2.0f) * x).ToArray();
            var res = torch.tensor(a).hypot(torch.tensor(b));
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.tan))]
        public void TanTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tan).ToArray();
            var res = torch.tensor(data).tan();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.sinh))]
        public void SinhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sinh).ToArray();
            var res = torch.tensor(data).sinh();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.cosh))]
        public void CoshTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cosh).ToArray();
            var res = torch.tensor(data).cosh();
            var tmp = res.data<Single>();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.tanh))]
        public void TanhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tanh).ToArray();
            var res = torch.tensor(data).tanh();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.asinh))]
        public void ArcSinhTest()
        {
            var data = new float[] { -0.1f, 0.0f, 0.1f };
            var expected = data.Select(MathF.Asinh).ToArray();
            var res = torch.tensor(data).asinh();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.acosh))]
        public void ArcCoshTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Acosh).ToArray();
            var res = torch.tensor(data).acosh();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.atanh))]
        public void ArcTanhTest()
        {
            var data = new float[] { -0.1f, 0.0f, 0.1f };
            var expected = data.Select(MathF.Atanh).ToArray();
            var res = torch.tensor(data).atanh();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.asin))]
        public void AsinTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Asin).ToArray();
            {
                var res = torch.tensor(data).asin();
                Assert.True(res.allclose(torch.tensor(expected)));
            }
            {
                var res = torch.tensor(data).arcsin();
                Assert.True(res.allclose(torch.tensor(expected)));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.acos))]
        public void AcosTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Acos).ToArray();
            {
                var res = torch.tensor(data).acos();
                Assert.True(res.allclose(torch.tensor(expected)));
            }
            {
                var res = torch.tensor(data).arccos();
                Assert.True(res.allclose(torch.tensor(expected)));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.atan))]
        public void AtanTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Atan).ToArray();
            {
                var res = torch.tensor(data).atan();
                Assert.True(res.allclose(torch.tensor(expected)));
            }
            {
                var res = torch.tensor(data).arctan();
                Assert.True(res.allclose(torch.tensor(expected)));
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.log))]
        public void LogTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(x => MathF.Log(x)).ToArray();
            var res = torch.tensor(data).log();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.log10))]
        public void Log10Test()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Log10).ToArray();
            var res = torch.tensor(data).log10();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.log2))]
        public void Log2Test()
        {
            var data = new float[] { 1.0f, 2.0f, 32.0f };
            var expected = data.Select(MathF.Log2).ToArray();
            var res = torch.tensor(data).log2();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.logaddexp))]
        public void LogAddExpTest()
        {
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            var y = new float[] { 4.0f, 5.0f, 6.0f };
            var expected = new float[x.Length];
            for (int i = 0; i < x.Length; i++) {
                expected[i] = MathF.Log(MathF.Exp(x[i]) + MathF.Exp(y[i]));
            }
            var res = torch.tensor(x).logaddexp(torch.tensor(y));
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.logaddexp2))]
        public void LogAddExp2Test()
        {
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            var y = new float[] { 4.0f, 5.0f, 6.0f };
            var expected = new float[x.Length];
            for (int i = 0; i < x.Length; i++) {
                expected[i] = MathF.Log(MathF.Pow(2.0f, x[i]) + MathF.Pow(2.0f, y[i]), 2.0f);
            }
            var res = torch.tensor(x).logaddexp2(torch.tensor(y));
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.reciprocal))]
        public void ReciprocalTest()
        {
            var x = torch.ones(new long[] { 10, 10 });
            x.fill_(4.0f);
            var y = x.reciprocal();

            Assert.All(x.data<float>().ToArray(), a => Assert.Equal(4.0f, a));
            Assert.All(y.data<float>().ToArray(), a => Assert.Equal(0.25f, a));

            x.reciprocal_();
            Assert.All(x.data<float>().ToArray(), a => Assert.Equal(0.25f, a));
        }

        [Fact]
        [TestOf(nameof(Tensor.exp2))]
        public void Exp2Test()
        {
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = new float[] { 2.0f, 4.0f, 8.0f };
            var res = torch.tensor(x).exp2();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.floor))]
        public void FloorTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Floor).ToArray();
            var input = torch.tensor(data);
            var res = input.floor();
            Assert.True(res.allclose(torch.tensor(expected)));

            input.floor_();
            Assert.True(input.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.floor_divide))]
        public void FloorDivideTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(d => MathF.Floor(d / 2)).ToArray();
            var input = torch.tensor(data);
            var res = input.floor_divide(2.0f);
            Assert.True(res.allclose(torch.tensor(expected)));

            input.floor_divide_(2.0f);
            Assert.True(input.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.trunc))]
        public void TruncTest()
        {
            var input = torch.randn(new long[] { 25 });
            var expected = input.data<float>().ToArray().Select(MathF.Truncate).ToArray();
            var res = input.trunc();
            Assert.True(res.allclose(torch.tensor(expected)));

            input.trunc_();
            Assert.True(input.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.ceil))]
        public void CeilTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Ceiling).ToArray();
            var input = torch.tensor(data);
            var res = input.ceil();
            Assert.True(res.allclose(torch.tensor(expected)));

            input.ceil_();
            Assert.True(res.allclose(torch.tensor(expected)));
        }

        [Fact]
        [TestOf(nameof(Tensor.round))]
        public void RoundTest()
        {
            var rnd = new Random();
            var data = Enumerable.Range(1, 100).Select(i => (float)rnd.NextDouble() * 10000).ToArray();

            {
                var expected = data.Select(x => MathF.Round(x)).ToArray();
                var input = torch.tensor(data);
                var res = input.round();
                Assert.True(res.allclose(torch.tensor(expected)));

                input.round_();
                Assert.True(input.allclose(torch.tensor(expected)));
            }
            {
                var expected = data.Select(x => MathF.Round(x * 10.0f) / 10.0f).ToArray();
                var input = torch.tensor(data);
                var res = input.round(1);
                Assert.True(res.allclose(torch.tensor(expected)));

                input.round_(1);
                Assert.True(input.allclose(torch.tensor(expected)));
            }
            {
                var expected = data.Select(x => MathF.Round(x * 100.0f) / 100.0f).ToArray();
                var input = torch.tensor(data);
                var res = input.round(2);
                Assert.True(res.allclose(torch.tensor(expected)));

                input.round_(2);
                Assert.True(input.allclose(torch.tensor(expected)));
            }
            {
                var expected = data.Select(x => MathF.Round(x * 0.1f) / 0.1f).ToArray();
                var input = torch.tensor(data);
                var res = input.round(-1);
                Assert.True(res.allclose(torch.tensor(expected)));

                input.round_(-1);
                Assert.True(input.allclose(torch.tensor(expected)));
            }
            {
                var expected = data.Select(x => MathF.Round(x * 0.01f) / 0.01f).ToArray();
                var input = torch.tensor(data);
                var res = input.round(-2);
                Assert.True(res.allclose(torch.tensor(expected)));

                input.round_(-2);
                Assert.True(input.allclose(torch.tensor(expected)));
            }
        }

        [Fact]
        [TestOf(nameof(torch.round))]
        [TestOf(nameof(torch.round_))]
        public void RoundTestWithDecimals()
        {
            const long n = 7L;
            var i = eye(n); // identity matrix
            var a = rand(new[] { n, n });
            var b = linalg.inv(a);

            // check non-inline version
            var r0 = round(matmul(a, b), 2L);
            var r1 = round(matmul(b, a), 3L);
            Assert.True(i.allclose(r0), "round() failed");
            Assert.True(i.allclose(r1), "round() failed");

            // check inline version
            var r0_ = matmul(a, b).round_(2L);
            var r1_ = matmul(b, a).round_(3L);
            Assert.True(i.allclose(r0_), "round_() failed");
            Assert.True(i.allclose(r1_), "round_() failed");
        }
    }
}
