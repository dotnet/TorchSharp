using AtenSharp;
using System;
using Xunit;

namespace AtenSharp.Test
{
    public class UnaryTensorOps
    {
        [Fact]
        public void FloatTensorLog()
        {
            var x1 = FloatTensor.Range(2f, 100f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Log();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Log(x1[i]),(double)x2[i]));
            }
        }

        [Fact]
        public void FloatTensorLog10()
        {
            var x1 = FloatTensor.Range(2f, 100f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Log10();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Log10(x1[i]),(double)x2[i]));
            }
        }

        [Fact]
        public void FloatTensorLog2()
        {
            var x1 = FloatTensor.Range(2f, 100f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Log2();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Log(x1[i],2.0),(double)x2[i]));
            }
        }

        [Fact]
        public void FloatTensorExp()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Exp();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Exp(x1[i]),(double)x2[i]));
            }
        }

        [Fact]
        public void FloatTensorTrigonometrics()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsin = x1.Sin();
            Assert.Equal(x1.Shape.Length,xsin.Shape.Length);
            Assert.Equal(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Sin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cos();
            Assert.Equal(x1.Shape.Length,xcos.Shape.Length);
            Assert.Equal(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Cos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tan();
            Assert.Equal(x1.Shape.Length,xtan.Shape.Length);
            Assert.Equal(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Tan(x1[i]),(double)xtan[i]));
            }
        }

        [Fact]
        public void FloatTensorArcTrigonometrics()
        {
            var x1 = FloatTensor.Range(0.0f, 0.99f, 0.05f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsin = x1.Asin();
            Assert.Equal(x1.Shape.Length,xsin.Shape.Length);
            Assert.Equal(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Asin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Acos();
            Assert.Equal(x1.Shape.Length,xcos.Shape.Length);
            Assert.Equal(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Acos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Atan();
            Assert.Equal(x1.Shape.Length,xtan.Shape.Length);
            Assert.Equal(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Atan(x1[i]),(double)xtan[i]));
            }
        }

        [Fact]
        public void FloatTensorHyperbolics()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsin = x1.Sinh();
            Assert.Equal(x1.Shape.Length,xsin.Shape.Length);
            Assert.Equal(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Sinh(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cosh();
            Assert.Equal(x1.Shape.Length,xcos.Shape.Length);
            Assert.Equal(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Cosh(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tanh();
            Assert.Equal(x1.Shape.Length,xtan.Shape.Length);
            Assert.Equal(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Tanh(x1[i]),(double)xtan[i]));
            }
        }

        [Fact]
        public void FloatTensorPower()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Pow(2.0f);
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(x1[i],2.0f),(double)x2[i]));
            }

            var x3 = x1.TPow(2.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(2.0f, x1[i]),(double)x3[i]));
            }
        }

        [Fact]
        public void FloatTensorUnaries()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsqrt = x1.Sqrt();
            Assert.Equal(x1.Shape.Length,xsqrt.Shape.Length);
            Assert.Equal(x1.Shape[0],xsqrt.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Sqrt(x1[i]),(double)xsqrt[i]));
            }

            x1 = FloatTensor.Range(-15f, 15f, .1f);

            var xceil = x1.Ceil();
            Assert.Equal(x1.Shape.Length,xceil.Shape.Length);
            Assert.Equal(x1.Shape[0],xceil.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Ceiling(x1[i]),(double)xceil[i]));
            }

            var xfloor = x1.Floor();
            Assert.Equal(x1.Shape.Length,xfloor.Shape.Length);
            Assert.Equal(x1.Shape[0],xfloor.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Floor(x1[i]),(double)xfloor[i]));
            }

            var xabs = x1.Abs();
            Assert.Equal(x1.Shape.Length,xabs.Shape.Length);
            Assert.Equal(x1.Shape[0],xabs.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Abs(x1[i]),(double)xabs[i]));
            }

            var xneg = x1.neg();
            Assert.Equal(x1.Shape.Length,xneg.Shape.Length);
            Assert.Equal(x1.Shape[0],xneg.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(-x1[i],(double)xneg[i]));
            }

#if false
            // DISABLED: libtorch appears to round away from zero for N.5, which means that comparing
            //           against Math.Round() will fail.
            x1 = FloatTensor.Range(0f, 15f, .1f);

            var xrnd = x1.Round();
            Assert.Equal(x1.Shape.Length,xrnd.Shape.Length);
            Assert.Equal(x1.Shape[0],xrnd.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Console.WriteLine($"{x1[i]}, {Math.Round(x1[i])}, {(double)xrnd[i]}");
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Round(x1[i]),(double)xrnd[i]));
            }
#endif
        }

        [Fact]
        public void CreateFloatTensorLike()
        {
            var x1 = new FloatTensor (200,200);
            var x2 = x1.OnesLike();
            var x3 = x1.ZerosLike();

            Assert.NotNull(x1);
            Assert.NotNull(x2);
            Assert.NotNull(x3);
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);

            x2.Resize1d(200*200);
            x3.Resize1d(200*200);
            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                Assert.True(1 == x2[i]);
                Assert.Equal(0, x3[i]);
            }
        }

        [Fact]
        public void SignFloat()
        {
            var x1 = FloatTensor.Range(-15f, 15f, 1f);
            var x2 = x1.Sign();

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                var sign = (x1[i] < 0) ? -1f : (x1[i] == 0) ? 0f : 1f;
                Assert.Equal(sign, x2[i]);
            }
        }

        [Fact]
        public void DoubleTensorLog()
        {
            var x1 = DoubleTensor.Range(2f, 100f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Log();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Log(x1[i]),(double)x2[i]));
            }
        }

        [Fact]
        public void DoubleTensorLog10()
        {
            var x1 = DoubleTensor.Range(2f, 100f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Log10();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Log10(x1[i]),(double)x2[i]));
            }
        }

        [Fact]
        public void DoubleTensorLog2()
        {
            var x1 = DoubleTensor.Range(2f, 100f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Log2();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Log(x1[i],2.0),(double)x2[i]));
            }
        }

        [Fact]
        public void DoubleTensorExp()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Exp();
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Exp(x1[i]),(double)x2[i]));
            }
        }

        [Fact]
        public void DoubleTensorTrigonometrics()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsin = x1.Sin();
            Assert.Equal(x1.Shape.Length,xsin.Shape.Length);
            Assert.Equal(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Sin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cos();
            Assert.Equal(x1.Shape.Length,xcos.Shape.Length);
            Assert.Equal(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Cos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tan();
            Assert.Equal(x1.Shape.Length,xtan.Shape.Length);
            Assert.Equal(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Tan(x1[i]),(double)xtan[i]));
            }
        }

        [Fact]
        public void DoubleTensorArcTrigonometrics()
        {
            var x1 = DoubleTensor.Range(0.0f, 0.99f, 0.05f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsin = x1.Asin();
            Assert.Equal(x1.Shape.Length,xsin.Shape.Length);
            Assert.Equal(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Asin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Acos();
            Assert.Equal(x1.Shape.Length,xcos.Shape.Length);
            Assert.Equal(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Acos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Atan();
            Assert.Equal(x1.Shape.Length,xtan.Shape.Length);
            Assert.Equal(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Atan(x1[i]),(double)xtan[i]));
            }
        }

        [Fact]
        public void DoubleTensorHyperbolics()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsin = x1.Sinh();
            Assert.Equal(x1.Shape.Length,xsin.Shape.Length);
            Assert.Equal(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Sinh(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cosh();
            Assert.Equal(x1.Shape.Length,xcos.Shape.Length);
            Assert.Equal(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Cosh(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tanh();
            Assert.Equal(x1.Shape.Length,xtan.Shape.Length);
            Assert.Equal(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Tanh(x1[i]),(double)xtan[i]));
            }
        }

        [Fact]
        public void DoubleTensorPower()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var x2 = x1.Pow(2.0f);
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape[0],x2.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(x1[i],2.0f),x2[i]));
            }

            var x3 = x1.TPow(2.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(2.0f, x1[i]),x3[i]));
            }
        }

        [Fact]
        public void DoubleTensorUnaries()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.NotNull(x1);
            Assert.True(1 == x1.Shape.Length);

            var xsqrt = x1.Sqrt();
            Assert.Equal(x1.Shape.Length, xsqrt.Shape.Length);
            Assert.Equal(x1.Shape[0], xsqrt.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Sqrt(x1[i]), (double)xsqrt[i]));
            }

            x1 = DoubleTensor.Range(-15f, 15f, .1f);

            var xceil = x1.Ceil();
            Assert.Equal(x1.Shape.Length, xceil.Shape.Length);
            Assert.Equal(x1.Shape[0], xceil.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Ceiling(x1[i]), (double)xceil[i]));
            }

            var xfloor = x1.Floor();
            Assert.Equal(x1.Shape.Length, xfloor.Shape.Length);
            Assert.Equal(x1.Shape[0], xfloor.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Floor(x1[i]), (double)xfloor[i]));
            }

            var xabs = x1.Abs();
            Assert.Equal(x1.Shape.Length, xabs.Shape.Length);
            Assert.Equal(x1.Shape[0], xabs.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Abs(x1[i]), (double)xabs[i]));
            }

            var xneg = x1.neg();
            Assert.Equal(x1.Shape.Length, xneg.Shape.Length);
            Assert.Equal(x1.Shape[0], xneg.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(-x1[i], (double)xneg[i]));
            }

#if false
            // DISABLED: libtorch appears to round away from zero for N.5, which means that comparing
            //           against Math.Round() will fail.
            x1 = DoubleTensor.Range(0f, 15f, .1f);

            var xrnd = x1.Round();
            Assert.Equal(x1.Shape.Length,xrnd.Shape.Length);
            Assert.Equal(x1.Shape[0],xrnd.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Console.WriteLine($"{x1[i]}, {Math.Round(x1[i])}, {(double)xrnd[i]}");
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Round(x1[i]),(double)xrnd[i]));
            }
#endif
        }

        [Fact]
        public void CreateDoubleTensorLike()
        {
            var x1 = new DoubleTensor (200,200);
            var x2 = x1.OnesLike();
            var x3 = x1.ZerosLike();

            Assert.NotNull(x1);
            Assert.NotNull(x2);
            Assert.NotNull(x3);
            Assert.Equal(x1.Shape.Length,x2.Shape.Length);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);

            x2.Resize1d(200*200);
            x3.Resize1d(200*200);
            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                Assert.True(1 == x2[i]);
                Assert.Equal(0, x3[i]);
            }
        }

        [Fact]
        public void SignDouble()
        {
            var x1 = DoubleTensor.Range(-15f, 15f, 1f);
            var x2 = x1.Sign();

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                var sign = (x1[i] < 0) ? -1 : (x1[i] == 0) ? 0 : 1;
                Assert.Equal(sign, x2[i]);
            }
        }
    }
}
