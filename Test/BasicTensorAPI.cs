using Microsoft.VisualStudio.TestTools.UnitTesting;
using TorchSharp;
using System;

namespace Test
{
    [TestClass]
    public class BasicTensorAPI
    {
        [TestMethod]
        public void CreateFloatTensor()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10,20);

            var x1Shape = x1.Shape;
            var x2Shape = x2.Shape;

            Assert.AreEqual(1,x1Shape.Length);
            Assert.AreEqual(2,x2Shape.Length);

            Assert.AreEqual(10,x1Shape[0]);
            Assert.AreEqual(10,x2Shape[0]);
            Assert.AreEqual(20,x2Shape[1]);
        }

        [TestMethod]
        public void ReshapeFloatTensor()
        {
            var x2 = new FloatTensor (200);

            Assert.AreEqual(1,x2.Shape.Length);
            Assert.AreEqual(200,x2.Shape[0]);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x2[i] = i*47.11f;
            }

            x2.Resize2d(10,20);
            Assert.AreEqual(2,x2.Shape.Length);
            Assert.AreEqual(10,x2.Shape[0]);
            Assert.AreEqual(20,x2.Shape[1]);

            x2.Resize3d(4,25,2);
            Assert.AreEqual(3,x2.Shape.Length);
            Assert.AreEqual(4,x2.Shape[0]);
            Assert.AreEqual(25,x2.Shape[1]);
            Assert.AreEqual(2,x2.Shape[2]);

            x2.Resize4d(4,5,5,2);
            Assert.AreEqual(4,x2.Shape.Length);
            Assert.AreEqual(4,x2.Shape[0]);
            Assert.AreEqual(5,x2.Shape[1]);
            Assert.AreEqual(5,x2.Shape[2]);
            Assert.AreEqual(2,x2.Shape[3]);

            x2.Resize5d(2,2,5,5,2);
            Assert.AreEqual(5,x2.Shape.Length);
            Assert.AreEqual(2,x2.Shape[0]);
            Assert.AreEqual(2,x2.Shape[1]);
            Assert.AreEqual(5,x2.Shape[2]);
            Assert.AreEqual(5,x2.Shape[3]);
            Assert.AreEqual(2,x2.Shape[4]);

            // Check that the values are retained across resizings.

            x2.Resize1d(200);
            Assert.AreEqual(1,x2.Shape.Length);
            Assert.AreEqual(200,x2.Shape[0]);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                Assert.AreEqual(i*47.11f, x2[i]);
            }
        }
        
        [TestMethod]
        public void CreateFloatTensorFromRange()
        {
            var x1 = FloatTensor.Range(0f, 150f, 5f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            float start = 0f;
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual(start+5f*i, x1[i]);
            }
        }

        [TestMethod]
        public void CreateDoubleTensor()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10,20);

            var x1Shape = x1.Shape;
            var x2Shape = x2.Shape;

            Assert.AreEqual(1,x1Shape.Length);
            Assert.AreEqual(2,x2Shape.Length);

            Assert.AreEqual(10,x1Shape[0]);
            Assert.AreEqual(10,x2Shape[0]);
            Assert.AreEqual(20,x2Shape[1]);
        }

        [TestMethod]
        public void ReshapeDoubleTensor()
        {
            var x2 = new DoubleTensor (200);

            Assert.AreEqual(1,x2.Shape.Length);
            Assert.AreEqual(200,x2.Shape[0]);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x2[i] = i*47.11f;
            }

            x2.Resize2d(10,20);
            Assert.AreEqual(2,x2.Shape.Length);
            Assert.AreEqual(10,x2.Shape[0]);
            Assert.AreEqual(20,x2.Shape[1]);

            x2.Resize3d(4,25,2);
            Assert.AreEqual(3,x2.Shape.Length);
            Assert.AreEqual(4,x2.Shape[0]);
            Assert.AreEqual(25,x2.Shape[1]);
            Assert.AreEqual(2,x2.Shape[2]);

            x2.Resize4d(4,5,5,2);
            Assert.AreEqual(4,x2.Shape.Length);
            Assert.AreEqual(4,x2.Shape[0]);
            Assert.AreEqual(5,x2.Shape[1]);
            Assert.AreEqual(5,x2.Shape[2]);
            Assert.AreEqual(2,x2.Shape[3]);

            x2.Resize5d(2,2,5,5,2);
            Assert.AreEqual(5,x2.Shape.Length);
            Assert.AreEqual(2,x2.Shape[0]);
            Assert.AreEqual(2,x2.Shape[1]);
            Assert.AreEqual(5,x2.Shape[2]);
            Assert.AreEqual(5,x2.Shape[3]);
            Assert.AreEqual(2,x2.Shape[4]);

            // Check that the values are retained across resizings.

            x2.Resize1d(200);
            Assert.AreEqual(1,x2.Shape.Length);
            Assert.AreEqual(200,x2.Shape[0]);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                Assert.AreEqual(i*47.11f, x2[i]);
            }
        }
        
        [TestMethod]
        public void CreateDoubleTensorFromRange()
        {
            var x1 = DoubleTensor.Range(0f, 150f, 5f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            float start = 0f;
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual(start+5f*i, x1[i]);
            }
        }

        
        internal static bool IsApproximatelyEqual(float expected, float actual)
        {
            if (expected < 0.0f)
            {
                var max = expected*.99999f;
                var min = expected*1.00001f;
                return actual >= min && actual <= max;
            }
            else 
            {
                var min = expected*.99999f;
                var max = expected*1.00001f;
                return actual >= min && actual <= max;
            }
        }
        internal static bool IsApproximatelyEqual(double expected, double actual)
        {
            if (expected < 0.0)
            {
                var max = expected*.99999;
                var min = expected*1.00001;
                //Console.WriteLine($"{min}, {max}, {actual}, {expected}");
                return actual >= min && actual <= max;
            }
            else 
            {
                var min = expected*.99999;
                var max = expected*1.00001;
                //Console.WriteLine($"{min}, {max}, {actual}, {expected}");
                return actual >= min && actual <= max;
            }
        }

        [TestMethod]
        public void FloatTensorLog()
        {
            var x1 = FloatTensor.Range(2f, 100f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Log();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Log(x1[i]),(double)x2[i]));
            }
        }
        
        [TestMethod]
        public void FloatTensorLog10()
        {
            var x1 = FloatTensor.Range(2f, 100f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Log10();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Log10(x1[i]),(double)x2[i]));
            }
        }
        
        [TestMethod]
        public void FloatTensorLog2()
        {
            var x1 = FloatTensor.Range(2f, 100f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Log2();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Log(x1[i],2.0),(double)x2[i]));
            }
        }

        [TestMethod]
        public void FloatTensorExp()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Exp();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Exp(x1[i]),(double)x2[i]));
            }
        }
        
        [TestMethod]
        public void FloatTensorTrigonometrics()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsin = x1.Sin();
            Assert.AreEqual(x1.Shape.Length,xsin.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Sin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cos();
            Assert.AreEqual(x1.Shape.Length,xcos.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Cos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tan();
            Assert.AreEqual(x1.Shape.Length,xtan.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Tan(x1[i]),(double)xtan[i]));
            }
        }

        [TestMethod]
        public void FloatTensorArcTrigonometrics()
        {
            var x1 = FloatTensor.Range(0.0f, 0.99f, 0.05f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsin = x1.Asin();
            Assert.AreEqual(x1.Shape.Length,xsin.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Asin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Acos();
            Assert.AreEqual(x1.Shape.Length,xcos.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Acos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Atan();
            Assert.AreEqual(x1.Shape.Length,xtan.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Atan(x1[i]),(double)xtan[i]));
            }
        }

        [TestMethod]
        public void FloatTensorHyperbolics()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsin = x1.Sinh();
            Assert.AreEqual(x1.Shape.Length,xsin.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Sinh(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cosh();
            Assert.AreEqual(x1.Shape.Length,xcos.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Cosh(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tanh();
            Assert.AreEqual(x1.Shape.Length,xtan.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Tanh(x1[i]),(double)xtan[i]));
            }
        }

        [TestMethod]
        public void FloatTensorPower()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Pow(2.0f);
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Pow(x1[i],2.0f),(double)x2[i]));
            }

            var x3 = x1.TPow(2.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Pow(2.0f, x1[i]),(double)x3[i]));
            }
        }

        [TestMethod]
        public void FloatTensorUnaries()
        {
            var x1 = FloatTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsqrt = x1.Sqrt();
            Assert.AreEqual(x1.Shape.Length,xsqrt.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsqrt.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Sqrt(x1[i]),(double)xsqrt[i]));
            }

            x1 = FloatTensor.Range(-15f, 15f, .1f);

            var xceil = x1.Ceil();
            Assert.AreEqual(x1.Shape.Length,xceil.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xceil.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Ceiling(x1[i]),(double)xceil[i]));
            }

            var xfloor = x1.Floor();
            Assert.AreEqual(x1.Shape.Length,xfloor.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xfloor.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Floor(x1[i]),(double)xfloor[i]));
            }

            var xabs = x1.Abs();
            Assert.AreEqual(x1.Shape.Length,xabs.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xabs.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Abs(x1[i]),(double)xabs[i]));
            }

            var xneg = x1.neg();
            Assert.AreEqual(x1.Shape.Length,xneg.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xneg.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(-x1[i],(double)xneg[i]));
            }

#if false            
            // DISABLED: libtorch appears to round away from zero for N.5, which means that comparing
            //           against Math.Round() will fail.
            x1 = FloatTensor.Range(0f, 15f, .1f);

            var xrnd = x1.Round();
            Assert.AreEqual(x1.Shape.Length,xrnd.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xrnd.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Console.WriteLine($"{x1[i]}, {Math.Round(x1[i])}, {(double)xrnd[i]}");
                Assert.IsTrue(IsApproximatelyEqual(Math.Round(x1[i]),(double)xrnd[i]));
            }
#endif
        }

        [TestMethod]
        public void CreateFloatTensorLike()
        {
            var x1 = new FloatTensor (200,200);
            var x2 = x1.OnesLike();
            var x3 = x1.ZerosLike();

            Assert.IsNotNull(x1);
            Assert.IsNotNull(x2);
            Assert.IsNotNull(x3);
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);

            x2.Resize1d(200*200);
            x3.Resize1d(200*200);
            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                Assert.AreEqual(1, x2[i]);
                Assert.AreEqual(0, x3[i]);
            }
        }
    
        [TestMethod]
        public void DoubleTensorLog()
        {
            var x1 = DoubleTensor.Range(2f, 100f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Log();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Log(x1[i]),(double)x2[i]));
            }
        }
        
        [TestMethod]
        public void DoubleTensorLog10()
        {
            var x1 = DoubleTensor.Range(2f, 100f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Log10();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Log10(x1[i]),(double)x2[i]));
            }
        }
        
        [TestMethod]
        public void DoubleTensorLog2()
        {
            var x1 = DoubleTensor.Range(2f, 100f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Log2();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Log(x1[i],2.0),(double)x2[i]));
            }
        }

        [TestMethod]
        public void DoubleTensorExp()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Exp();
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Exp(x1[i]),(double)x2[i]));
            }
        }
        
        [TestMethod]
        public void DoubleTensorTrigonometrics()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsin = x1.Sin();
            Assert.AreEqual(x1.Shape.Length,xsin.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Sin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cos();
            Assert.AreEqual(x1.Shape.Length,xcos.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Cos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tan();
            Assert.AreEqual(x1.Shape.Length,xtan.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Tan(x1[i]),(double)xtan[i]));
            }
        }

        [TestMethod]
        public void DoubleTensorArcTrigonometrics()
        {
            var x1 = DoubleTensor.Range(0.0f, 0.99f, 0.05f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsin = x1.Asin();
            Assert.AreEqual(x1.Shape.Length,xsin.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Asin(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Acos();
            Assert.AreEqual(x1.Shape.Length,xcos.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Acos(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Atan();
            Assert.AreEqual(x1.Shape.Length,xtan.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Atan(x1[i]),(double)xtan[i]));
            }
        }

        [TestMethod]
        public void DoubleTensorHyperbolics()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var xsin = x1.Sinh();
            Assert.AreEqual(x1.Shape.Length,xsin.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xsin.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Sinh(x1[i]),(double)xsin[i]));
            }

            var xcos = x1.Cosh();
            Assert.AreEqual(x1.Shape.Length,xcos.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xcos.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Cosh(x1[i]),(double)xcos[i]));
            }

            var xtan = x1.Tanh();
            Assert.AreEqual(x1.Shape.Length,xtan.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xtan.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Tanh(x1[i]),(double)xtan[i]));
            }
        }

        [TestMethod]
        public void DoubleTensorPower()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1,x1.Shape.Length);

            var x2 = x1.Pow(2.0f);
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x2.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Pow(x1[i],2.0f),x2[i]));
            }

            var x3 = x1.TPow(2.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Pow(2.0f, x1[i]),x3[i]));
            }
        }

        [TestMethod]
        public void DoubleTensorUnaries()
        {
            var x1 = DoubleTensor.Range(2f, 15f, 1f);

            Assert.IsNotNull(x1);
            Assert.AreEqual(1, x1.Shape.Length);

            var xsqrt = x1.Sqrt();
            Assert.AreEqual(x1.Shape.Length, xsqrt.Shape.Length);
            Assert.AreEqual(x1.Shape[0], xsqrt.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Sqrt(x1[i]), (double)xsqrt[i]));
            }

            x1 = DoubleTensor.Range(-15f, 15f, .1f);

            var xceil = x1.Ceil();
            Assert.AreEqual(x1.Shape.Length, xceil.Shape.Length);
            Assert.AreEqual(x1.Shape[0], xceil.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Ceiling(x1[i]), (double)xceil[i]));
            }

            var xfloor = x1.Floor();
            Assert.AreEqual(x1.Shape.Length, xfloor.Shape.Length);
            Assert.AreEqual(x1.Shape[0], xfloor.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Floor(x1[i]), (double)xfloor[i]));
            }

            var xabs = x1.Abs();
            Assert.AreEqual(x1.Shape.Length, xabs.Shape.Length);
            Assert.AreEqual(x1.Shape[0], xabs.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(Math.Abs(x1[i]), (double)xabs[i]));
            }

            var xneg = x1.neg();
            Assert.AreEqual(x1.Shape.Length, xneg.Shape.Length);
            Assert.AreEqual(x1.Shape[0], xneg.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(IsApproximatelyEqual(-x1[i], (double)xneg[i]));
            }

#if false
            // DISABLED: libtorch appears to round away from zero for N.5, which means that comparing
            //           against Math.Round() will fail.
            x1 = DoubleTensor.Range(0f, 15f, .1f);

            var xrnd = x1.Round();
            Assert.AreEqual(x1.Shape.Length,xrnd.Shape.Length);
            Assert.AreEqual(x1.Shape[0],xrnd.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Console.WriteLine($"{x1[i]}, {Math.Round(x1[i])}, {(double)xrnd[i]}");
                Assert.IsTrue(IsApproximatelyEqual(Math.Round(x1[i]),(double)xrnd[i]));
            }
#endif
        }

        [TestMethod]
        public void CreateDoubleTensorLike()
        {
            var x1 = new DoubleTensor (200,200);
            var x2 = x1.OnesLike();
            var x3 = x1.ZerosLike();

            Assert.IsNotNull(x1);
            Assert.IsNotNull(x2);
            Assert.IsNotNull(x3);
            Assert.AreEqual(x1.Shape.Length,x2.Shape.Length);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);

            x2.Resize1d(200*200);
            x3.Resize1d(200*200);
            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                Assert.AreEqual(1, x2[i]);
                Assert.AreEqual(0, x3[i]);
            }
        }
    }
}
