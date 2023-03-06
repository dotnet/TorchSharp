// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class LinearAlgebra
    {
        [Fact]
        [TestOf(nameof(tensordot))]
        public void TestTensorDot()
        {
            var a = arange(60).reshape(3, 4, 5);
            var b = arange(24).reshape(4, 3, 2);
            var res = tensordot(a, b, new []{ 1L, 0L }, new []{ 0L, 1L });

            var expected = from_array(new long[,]
            {
                {4400, 4730},
                {4532, 4874},
                {4664, 5018},
                {4796, 5162},
                {4928, 5306}
            });

            Assert.True(allclose(res, expected));
        }

        [Fact]
        [TestOf(nameof(lu))]
        public void TestLUSolve()
        {
            var A = randn(2, 3, 3);
            var b = randn(2, 3, 1);

            {
                var (A_LU, pivots, infos) = lu(A);

                Assert.NotNull(A_LU);
                Assert.NotNull(pivots);
                Assert.Null(infos);

                Assert.Equal(new long[] { 2, 3, 3 }, A_LU.shape);
                Assert.Equal(new long[] { 2, 3 }, pivots.shape);

                var x = lu_solve(b, A_LU, pivots);
                Assert.Equal(new long[] { 2, 3, 1 }, x.shape);

                var y = norm(bmm(A, x) - b);
                Assert.Empty(y.shape);
            }

            {
                var (A_LU, pivots, infos) = lu(A, get_infos: true);

                Assert.NotNull(A_LU);
                Assert.NotNull(pivots);
                Assert.NotNull(infos);

                Assert.Equal(new long[] { 2, 3, 3 }, A_LU.shape);
                Assert.Equal(new long[] { 2, 3 }, pivots.shape);
                Assert.Equal(new long[] { 2 }, infos.shape);

                var x = lu_solve(b, A_LU, pivots);
                Assert.Equal(new long[] { 2, 3, 1 }, x.shape);

                var y = norm(bmm(A, x) - b);
                Assert.Empty(y.shape);
            }
        }

        [Fact]
        [TestOf(nameof(lu_unpack))]
        public void TestLUUnpack()
        {
            var A = randn(2, 3, 3);

            {
                var (A_LU, pivots, infos) = lu(A);

                Assert.NotNull(A_LU);
                Assert.NotNull(pivots);
                Assert.Null(infos);

                var (P, A_L, A_U) = lu_unpack(A_LU, pivots);

                Assert.NotNull(P);
                Assert.NotNull(A_L);
                Assert.NotNull(A_U);

                Assert.Equal(new long[] { 2, 3, 3 }, P.shape);
                Assert.Equal(new long[] { 2, 3, 3 }, A_L!.shape);
                Assert.Equal(new long[] { 2, 3, 3 }, A_U!.shape);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.mul))]
        public void TestMul()
        {
            var x = ones(new long[] { 100, 100 });

            var y = x.mul(0.5f.ToScalar());

            var ydata = y.data<float>();
            var xdata = x.data<float>();

            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    Assert.Equal(ydata[i + j], xdata[i + j] * 0.5f);
                }
            }
        }

        void TestMmGen(Device device)
        {
            {
                var x1 = ones(new long[] { 1, 2 }, device: device);
                var x2 = ones(new long[] { 2, 1 }, device: device);

                var y = x1.mm(x2).to(DeviceType.CPU);

                var ydata = y.data<float>();

                Assert.Equal(2.0f, ydata[0]);
            }
            //System.Runtime.InteropServices.ExternalException : addmm for CUDA tensors only supports floating - point types.Try converting the tensors with.float() at C:\w\b\windows\pytorch\aten\src\THC / generic / THCTensorMathBlas.cu:453
            if (device.type == DeviceType.CPU) {
                var x1 = ones(new long[] { 1, 2 }, int64, device: device);
                var x2 = ones(new long[] { 2, 1 }, int64, device: device);

                var y = x1.mm(x2).to(DeviceType.CPU);

                var ydata = y.data<long>();

                Assert.Equal(2L, ydata[0]);
            }
        }

        [Fact]
        [TestOf(nameof(CPU))]
        public void TestMmCpu()
        {
            TestMmGen(CPU);
        }

        [Fact]
        [TestOf(nameof(CUDA))]
        public void TestMmCuda()
        {
            if (cuda.is_available()) {
                TestMmGen(CUDA);
            }
        }

        void TestMVGen(Device device)
        {
            {
                var mat1 = ones(new long[] { 4, 3 }, device: device);
                var vec1 = ones(new long[] { 3 }, device: device);

                var y = mat1.mv(vec1).to(DeviceType.CPU);

                Assert.Equal(4, y.shape[0]);
            }
        }

        void TestAddMVGen(Device device)
        {
            {
                var x1 = ones(new long[] { 4 }, device: device);
                var mat1 = ones(new long[] { 4, 3 }, device: device);
                var vec1 = ones(new long[] { 3 }, device: device);

                var y = x1.addmv(mat1, vec1).to(DeviceType.CPU);

                Assert.Equal(4, y.shape[0]);
            }
        }

        [Fact]
        [TestOf(nameof(CPU))]
        public void TestMVCpu()
        {
            TestMVGen(CPU);
        }

        [Fact]
        [TestOf(nameof(CUDA))]
        public void TestMVCuda()
        {
            if (cuda.is_available()) {
                TestMVGen(CUDA);
            }
        }

        [Fact]
        public void TestAddMVCpu()
        {
            TestAddMVGen(CPU);
        }

        [Fact]
        [TestOf(nameof(CUDA))]
        public void TestAddMVCuda()
        {
            if (cuda.is_available()) {
                TestAddMVGen(CUDA);
            }
        }

        void TestAddRGen(Device device)
        {
            {
                var x1 = ones(new long[] { 4, 3 }, device: device);
                var vec1 = ones(new long[] { 4 }, device: device);
                var vec2 = ones(new long[] { 3 }, device: device);

                var y = x1.addr(vec1, vec2).to(DeviceType.CPU);

                Assert.Equal(new long[] { 4, 3 }, y.shape);
            }
        }

        [Fact]
        [TestOf(nameof(CPU))]
        public void TestAddRCpu()
        {
            TestAddRGen(CPU);
        }

        [Fact]
        [TestOf(nameof(CUDA))]
        public void TestAddRCuda()
        {
            if (cuda.is_available()) {
                TestAddRGen(CUDA);
            }
        }

        [Fact]
        [TestOf(nameof(Tensor.vdot))]
        public void VdotTest()
        {
            var a = new float[] { 1.0f, 2.0f, 3.0f };
            var b = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = tensor(a.Zip(b).Select(x => x.First * x.Second).Sum());
            var res = tensor(a).vdot(tensor(b));
            Assert.True(res.allclose(expected));
        }

        [Fact]
        [TestOf(nameof(Tensor.vander))]
        public void VanderTest()
        {
            var x = tensor(new int[] { 1, 2, 3, 5 });
            {
                var res = x.vander();
                var expected = tensor(new long[] { 1, 1, 1, 1, 8, 4, 2, 1, 27, 9, 3, 1, 125, 25, 5, 1 }, 4, 4);
                Assert.Equal(expected, res);
            }
            {
                var res = x.vander(3);
                var expected = tensor(new long[] { 1, 1, 1, 4, 2, 1, 9, 3, 1, 25, 5, 1 }, 4, 3);
                Assert.Equal(expected, res);
            }
            {
                var res = x.vander(3, true);
                var expected = tensor(new long[] { 1, 1, 1, 1, 2, 4, 1, 3, 9, 1, 5, 25 }, 4, 3);
                Assert.Equal(expected, res);
            }
        }

        [Fact]
        [TestOf(nameof(linalg.vander))]
        public void LinalgVanderTest()
        {
            var x = tensor(new int[] { 1, 2, 3, 5 });
            {
                var res = linalg.vander(x);
                var expected = tensor(new long[] { 1, 1, 1, 1, 1, 2, 4, 8, 1, 3, 9, 27, 1, 5, 25, 125 }, 4, 4);
                Assert.Equal(expected, res);
            }
            {
                var res = linalg.vander(x, 3);
                var expected = tensor(new long[] { 1, 1, 1, 1, 2, 4, 1, 3, 9, 1, 5, 25 }, 4, 3);
                Assert.Equal(expected, res);
            }
        }

        [Fact]
        [TestOf(nameof(linalg.cholesky))]
        public void CholeskyTest()
        {
            var a = randn(new long[] { 3, 2, 2 }, float64);
            a = a.matmul(a.swapdims(-2, -1));   // Worked this in to get it tested. Alias for 'transpose'
            var l = linalg.cholesky(a);

            Assert.True(a.allclose(l.matmul(l.swapaxes(-2, -1)))); // Worked this in to get it tested. Alias for 'transpose'
        }

        [Fact]
        [TestOf(nameof(linalg.cholesky_ex))]
        public void CholeskyExTest()
        {
            var a = randn(new long[] { 3, 2, 2 }, float64);
            a = a.matmul(a.swapdims(-2, -1));   // Worked this in to get it tested. Alias for 'transpose'
            var (l, info) = linalg.cholesky_ex(a);

            Assert.True(a.allclose(l.matmul(l.swapaxes(-2, -1))));
        }

        [Fact]
        [TestOf(nameof(linalg.inv))]
        public void InvTest()
        {
            var a = randn(new long[] { 3, 2, 2 }, float64);
            var l = linalg.inv(a);

            Assert.Equal(a.shape, l.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.inv_ex))]
        public void InvExTest()
        {
            var a = randn(new long[] { 3, 2, 2 }, float64);
            var (l, info) = linalg.inv_ex(a);

            Assert.Equal(a.shape, l.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.cond))]
        public void CondTestF64()
        {
            {
                var a = randn(new long[] { 3, 3, 3 }, float64);
                // The following mostly checks that the runtime interop doesn't blow up.
                _ = linalg.cond(a);
                _ = linalg.cond(a, "fro");
                _ = linalg.cond(a, "nuc");
                _ = linalg.cond(a, 1);
                _ = linalg.cond(a, -1);
                _ = linalg.cond(a, 2);
                _ = linalg.cond(a, -2);
                _ = linalg.cond(a, Double.PositiveInfinity);
                _ = linalg.cond(a, Double.NegativeInfinity);
            }
        }

        [Fact]
        [TestOf(nameof(linalg.cond))]
        public void CondTestCF64()
        {
            {
                var a = randn(new long[] { 3, 3, 3 }, complex128);
                // The following mostly checks that the runtime interop doesn't blow up.
                _ = linalg.cond(a);
                _ = linalg.cond(a, "fro");
                _ = linalg.cond(a, "nuc");
                _ = linalg.cond(a, 1);
                _ = linalg.cond(a, -1);
                _ = linalg.cond(a, 2);
                _ = linalg.cond(a, -2);
                _ = linalg.cond(a, Double.PositiveInfinity);
                _ = linalg.cond(a, Double.NegativeInfinity);
            }
        }

        [Fact]
        [TestOf(nameof(linalg.qr))]
        public void QRTest()
        {
            var a = randn(new long[] { 4, 25, 25 });

            var l = linalg.qr(a);

            Assert.Equal(a.shape, l.Q.shape);
            Assert.Equal(a.shape, l.R.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.solve))]
        public void SolveTest()
        {
            var A = randn(3, 3);
            var b = randn(3);
            var x = linalg.solve(A, b);
            Assert.True(A.matmul(x).allclose(b, rtol: 1e-03, atol: 1e-06));
        }

        [Fact]
        [TestOf(nameof(linalg.svd))]
        public void SVDTest()
        {
            var a = randn(new long[] { 4, 25, 15 });

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
        [TestOf(nameof(linalg.svdvals))]
        public void SVDValsTest()
        {
            var a = tensor(new double[] { -1.3490, -0.1723, 0.7730,
                -1.6118, -0.3385, -0.6490,
                 0.0908, 2.0704, 0.5647,
                -0.6451, 0.1911, 0.7353,
                 0.5247, 0.5160, 0.5110}, 5, 3);

            var l = linalg.svdvals(a);
            Assert.True(l.allclose(tensor(new double[] { 2.5138929972840613, 2.1086555338402455, 1.1064930672223237 }), rtol: 1e-04, atol: 1e-07));
        }

        [Fact]
        [TestOf(nameof(linalg.lstsq))]
        public void LSTSQTest()
        {
            var a = randn(new long[] { 4, 25, 15 });
            var b = randn(new long[] { 4, 25, 10 });

            var l = linalg.lstsq(a, b);

            Assert.Equal(new long[] { 4, 15, 10 }, l.Solution.shape);
            Assert.Equal(0, l.Residuals.shape[0]);
            Assert.Equal(new long[] { 4 }, l.Rank.shape);
            Assert.Equal(new long[] { 4, 15, 10 }, l.Solution.shape);
            Assert.Equal(0, l.SingularValues.shape[0]);
        }

        [Fact]
        [TestOf(nameof(linalg.lu))]
        public void LUTest()
        {
            var A = randn(2, 3, 3);
            var A_factor = linalg.lu(A);
            // For right now, pretty much just checking that it's not blowing up.
            Assert.Multiple(
                () => Assert.NotNull(A_factor.P),
                () => Assert.NotNull(A_factor.L),
                () => Assert.NotNull(A_factor.U)
            );
        }

        [Fact]
        [TestOf(nameof(linalg.lu_factor))]
        public void LUFactorTest()
        {
            var A = randn(2, 3, 3);
            var A_factor = linalg.lu_factor(A);
            // For right now, pretty much just checking that it's not blowing up.
            Assert.Multiple(
                () => Assert.NotNull(A_factor.LU),
                () => Assert.NotNull(A_factor.Pivots)
            );
        }

        [Fact]
        [TestOf(nameof(linalg.ldl_factor))]
        public void LDLFactorTest()
        {
            var A = randn(2, 3, 3);
            var A_factor = linalg.ldl_factor(A);
            // For right now, pretty much just checking that it's not blowing up.
            Assert.Multiple(
                () => Assert.NotNull(A_factor.LU),
                () => Assert.NotNull(A_factor.Pivots)
            );
        }

        [Fact]
        [TestOf(nameof(linalg.ldl_factor))]
        public void LDLFactorExTest()
        {
            var A = randn(2, 3, 3);
            var A_factor = linalg.ldl_factor_ex(A);
            // For right now, pretty much just checking that it's not blowing up.
            Assert.Multiple(
                () => Assert.NotNull(A_factor.LU),
                () => Assert.NotNull(A_factor.Pivots),
                () => Assert.NotNull(A_factor.Info)
            );
        }

        [Fact]
        [TestOf(nameof(Tensor.matrix_power))]
        public void MatrixPowerTest()
        {
            var a = randn(new long[] { 25, 25 });
            var b = a.matrix_power(3);
            Assert.Equal(new long[] { 25, 25 }, b.shape);
        }

        [Fact]
        [TestOf(nameof(Tensor.matrix_exp))]
        public void MatrixExpTest1()
        {
            var a = randn(new long[] { 25, 25 });
            var b = a.matrix_exp();
            Assert.Equal(new long[] { 25, 25 }, b.shape);

            var c = matrix_exp(a);
            Assert.Equal(new long[] { 25, 25 }, c.shape);
        }

        [Fact]
        [TestOf(nameof(matrix_exp))]
        public void MatrixExpTest2()
        {
            var a = randn(new long[] { 16, 25, 25 });
            var b = a.matrix_exp();
            Assert.Equal(new long[] { 16, 25, 25 }, b.shape);
            var c = matrix_exp(a);
            Assert.Equal(new long[] { 16, 25, 25 }, c.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.matrix_rank))]
        public void MatrixRankTest()
        {
            var mr1 = linalg.matrix_rank(randn(4, 3, 2));
            Assert.Equal(new long[] { 4 }, mr1.shape);

            var mr2 = linalg.matrix_rank(randn(2, 4, 3, 2));
            Assert.Equal(new long[] { 2, 4 }, mr2.shape);

            // Really just testing that it doesn't blow up in interop for the following lines:

            mr2 = linalg.matrix_rank(randn(2, 4, 3, 2), atol: 1.0);
            Assert.Equal(new long[] { 2, 4 }, mr2.shape);

            mr2 = linalg.matrix_rank(randn(2, 4, 3, 2), atol: 1.0, rtol: 0.0);
            Assert.Equal(new long[] { 2, 4 }, mr2.shape);

            mr2 = linalg.matrix_rank(randn(2, 4, 3, 2), atol: tensor(1.0));
            Assert.Equal(new long[] { 2, 4 }, mr2.shape);

            mr2 = linalg.matrix_rank(randn(2, 4, 3, 2), atol: tensor(1.0), rtol: tensor(0.0));
            Assert.Equal(new long[] { 2, 4 }, mr2.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.multi_dot))]
        public void MultiDotTest()
        {
            var a = randn(new long[] { 25, 25 });
            var b = randn(new long[] { 25, 25 });
            var c = randn(new long[] { 25, 25 });
            var d = linalg.multi_dot(new Tensor[] { a, b, c });
            Assert.Equal(new long[] { 25, 25 }, d.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.det))]
        public void DeterminantTest()
        {
            {
                var a = tensor(
                    new float[] { 0.9478f, 0.9158f, -1.1295f,
                                  0.9701f, 0.7346f, -1.8044f,
                                 -0.2337f, 0.0557f, 0.6929f }, 3, 3);
                var l = linalg.det(a);
                Assert.True(l.allclose(tensor(0.09335048f)));
            }
            {
                var a = tensor(
                    new float[] { 0.9254f, -0.6213f, -0.5787f, 1.6843f, 0.3242f, -0.9665f,
                                  0.4539f, -0.0887f, 1.1336f, -0.4025f, -0.7089f, 0.9032f }, 3, 2, 2);
                var l = linalg.det(a);
                Assert.True(l.allclose(tensor(new float[] { 1.19910491f, 0.4099378f, 0.7385352f })));
            }
        }

        [Fact]
        [TestOf(nameof(linalg.matrix_norm))]
        public void MatrixNormTest()
        {
            {
                var a = arange(9, float32).view(3, 3);

                var b = linalg.matrix_norm(a);
                var c = linalg.matrix_norm(a, ord: -1);

                Assert.Equal(14.282857f, b.item<float>());
                Assert.Equal(9.0f, c.item<float>());
            }
        }

        [Fact]
        [TestOf(nameof(linalg.vector_norm))]
        public void VectorNormTest()
        {
            {
                var a = tensor(
                    new float[] { -4.0f, -3.0f, -2.0f, -1.0f, 0, 1.0f, 2.0f, 3.0f, 4.0f });

                var b = linalg.vector_norm(a, ord: 3.5);
                var c = linalg.vector_norm(a.view(3, 3), ord: 3.5);

                Assert.Equal(5.4344883f, b.item<float>());
                Assert.Equal(5.4344883f, c.item<float>());
            }
        }

        [Fact]
        [TestOf(nameof(linalg.pinv))]
        public void PinvTest()
        {
            var mr1 = linalg.pinv(randn(4, 3, 5));
            Assert.Equal(new long[] { 4, 5, 3 }, mr1.shape);

            // Really just testing that it doesn't blow up in interop for the following lines:

            mr1 = linalg.pinv(randn(4, 3, 5), atol: 1.0);
            Assert.Equal(new long[] { 4, 5, 3 }, mr1.shape);

            mr1 = linalg.pinv(randn(4, 3, 5), atol: 1.0, rtol: 0.0);
            Assert.Equal(new long[] { 4, 5, 3 }, mr1.shape);

            mr1 = linalg.pinv(randn(4, 3, 5), atol: tensor(1.0));
            Assert.Equal(new long[] { 4, 5, 3 }, mr1.shape);

            mr1 = linalg.pinv(randn(4, 3, 5), atol: tensor(1.0), rtol: tensor(0.0));
            Assert.Equal(new long[] { 4, 5, 3 }, mr1.shape);
        }

        [Fact]
        [TestOf(nameof(linalg.eig))]
        public void EigTest32()
        {
            {
                var a = tensor(
                    new float[] { 2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f, -2.7457f, -1.7517f, 1.7166f }, 3, 3);

                var expected = tensor(
                    new (float, float)[] { (3.44288778f, 0.0f), (2.17609453f, 0.0f), (-2.128083f, 0.0f) });

                {
                    var (values, vectors) = linalg.eig(a);
                    Assert.NotNull(vectors);
                    Assert.True(values.allclose(expected));
                }
            }
        }

        [Fact]
        [TestOf(nameof(linalg.eigvals))]
        public void EighvalsTest32()
        {
            {
                var a = tensor(
                    new float[] { 2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f, -2.7457f, -1.7517f, 1.7166f }, 3, 3);
                var expected = tensor(
                    new (float, float)[] { (3.44288778f, 0.0f), (2.17609453f, 0.0f), (-2.128083f, 0.0f) });
                var l = linalg.eigvals(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(linalg.eigvals))]
        public void EighvalsTest64()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var a = tensor(
                    new double[] { 2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f, -2.7457f, -1.7517f, 1.7166f }, 3, 3);
                var expected = tensor(
                    new System.Numerics.Complex[] { new System.Numerics.Complex(3.44288778f, 0.0f), new System.Numerics.Complex(2.17609453f, 0.0f), new System.Numerics.Complex(-2.128083f, 0.0f) });
                var l = linalg.eigvals(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(linalg.eigvalsh))]
        public void EighvalshTest32()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var a = tensor(
                    new float[] {  2.8050f, -0.3850f, -0.3850f, 3.2376f, -1.0307f, -2.7457f,
                                  -2.7457f, -1.7517f, 1.7166f,  2.2207f, 2.2207f, -2.0898f }, 3, 2, 2);
                var expected = tensor(
                    new float[] { 2.5797f, 3.46290016f, -4.16046524f, 1.37806475f, -3.11126733f, 2.73806715f }, 3, 2);
                var l = linalg.eigvalsh(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(linalg.eigvalsh))]
        public void EighvalshTest64()
        {
            {
                var a = tensor(
                    new double[] {  2.8050, -0.3850, -0.3850, 3.2376, -1.0307, -2.7457,
                                  -2.7457, -1.7517, 1.7166,  2.2207, 2.2207, -2.0898 }, 3, 2, 2);
                var expected = tensor(
                    new double[] { 2.5797, 3.46290016, -4.16046524, 1.37806475, -3.11126733, 2.73806715 }, 3, 2);
                var l = linalg.eigvalsh(a);
                Assert.True(l.allclose(expected));
            }
        }

        [Fact]
        [TestOf(nameof(linalg.norm))]
        public void LinalgNormTest()
        {
            {
                var a = tensor(
                    new float[] { -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f });
                var b = a.reshape(3, 3);

                Assert.True(linalg.norm(a).allclose(tensor(7.7460f)));
                Assert.True(linalg.norm(b).allclose(tensor(7.7460f)));
                Assert.True(linalg.norm(b, "fro").allclose(tensor(7.7460f)));

                Assert.True(linalg.norm(a, float.PositiveInfinity).allclose(tensor(4.0f)));
                Assert.True(linalg.norm(b, float.PositiveInfinity).allclose(tensor(9.0f)));
                Assert.True(linalg.norm(a, float.NegativeInfinity).allclose(tensor(0.0f)));
                Assert.True(linalg.norm(b, float.NegativeInfinity).allclose(tensor(2.0f)));

                Assert.True(linalg.norm(a, 1).allclose(tensor(20.0f)));
                Assert.True(linalg.norm(b, 1).allclose(tensor(7.0f)));
                Assert.True(linalg.norm(a, -1).allclose(tensor(0.0f)));
                Assert.True(linalg.norm(b, -1).allclose(tensor(6.0f)));

                Assert.True(linalg.norm(a, 2).allclose(tensor(7.7460f)));
                Assert.True(linalg.norm(b, 2).allclose(tensor(7.3485f)));
                Assert.True(linalg.norm(a, 3).allclose(tensor(5.8480f)));
                Assert.True(linalg.norm(a, -2).allclose(tensor(0.0f)));
                Assert.True(linalg.norm(a, -3).allclose(tensor(0.0f)));
            }
        }

        [Fact]
        public void TestTrilIndex()
        {
            var a = tril_indices(3, 3);
            var expected = new long[] { 0, 1, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2 };
            Assert.Equal(expected, a.data<long>().ToArray());
        }

        [Fact]
        public void TestTriuIndex()
        {
            var a = triu_indices(3, 3);
            var expected = new long[] { 0, 0, 0, 1, 1, 2, 0, 1, 2, 1, 2, 2 };
            Assert.Equal(expected, a.data<long>().ToArray());
        }
    }
}
