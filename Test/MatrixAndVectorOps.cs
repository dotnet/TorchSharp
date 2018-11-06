using Microsoft.VisualStudio.TestTools.UnitTesting;
using TorchSharp;
using System;
using System.Linq;

namespace Test
{
    [TestClass]
    public class MatrixAndVectorOps
    {
        [TestMethod]
        public void DotProductFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var dot = x1.Dot(x2);

            Assert.AreEqual(715f, dot);
        }

        [TestMethod]
        public void CMulFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var x3 = x1.CMul(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(x1[i]*x2[i],x3[i]));
            }
        }
        
        [TestMethod]
        public void CDivFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var x3 = x1.CDiv(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(x1[i]/x2[i],x3[i]));
            }
        }

        [TestMethod]
        public void LtFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LtTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<x2[i])?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void LeFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LeTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=x2[i])?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void GtFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GtTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void GeFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GeTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void EqFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.EqTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void NeFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.NeTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void LtTFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LtTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void LeTFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LeTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GtTFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GtTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GeTFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GeTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void EqTFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.EqTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void NeTFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.NeTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void CPowFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var x3 = x1.CPow(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(x1[i],x2[i]),x3[i]));
            }
        }

        [TestMethod]
        public void CMaxFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.CMax(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],x2[i]),x3[i]));
            }
        }

        [TestMethod]
        public void CMinFloat()
        {
            var x1 = new FloatTensor (10);
            var x2 = new FloatTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.CMin(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],x2[i]),x3[i]));
            }
        }

        [TestMethod]
        public void CMaxValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMaxValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],0.0f),x3[i]));
            }
        }

        [TestMethod]
        public void CMinValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMinValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],0.0f),x3[i]));
            }
        }      

        [TestMethod]
        public void LtValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void LeValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void GtValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void GeValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void EqValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void NeValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=0.0f)?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void LtTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void LeTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GtTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GeTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void EqTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void NeTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void DotProductDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var dot = x1.Dot(x2);

            Assert.AreEqual(715, dot);
        }

        [TestMethod]
        public void CMulDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var x3 = x1.CMul(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(x1[i]*x2[i],x3[i]));
            }
        }
        
        [TestMethod]
        public void CDivDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var x3 = x1.CDiv(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(x1[i]/x2[i],x3[i]));
            }
        }

        [TestMethod]
        public void CMaxDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.CMax(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],x2[i]),x3[i]));
            }
        }      

        [TestMethod]
        public void CMinDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.CMin(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],x2[i]),x3[i]));
            }
        }

        [TestMethod]
        public void CMaxValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMaxValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],0.0f),x3[i]));
            }
        }

        [TestMethod]
        public void CMinValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMinValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],0.0f),x3[i]));
            }
        }

        [TestMethod]
        public void LtDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LtTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<x2[i])?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void LeDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LeTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=x2[i])?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void GtDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GtTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void GeDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GeTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void EqDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.EqTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void NeDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.NeTensor(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=x2[i])?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void LtTDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LtTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void LeTDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.LeTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GtTDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GtTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GeTDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.GeTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void EqTDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.EqTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void NeTDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
                x2[i] = i+1;
            }

            var x3 = x1.NeTensorT(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=x2[i])?1.0f:0.0f,x3[i]);
            }
        }      

        [TestMethod]
        public void LtValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void LeValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void GtValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void GeValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void EqValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==0.0f)?1:0,x3[i]);
            }
        }      

        [TestMethod]
        public void NeValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValue(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=0.0f)?1:0,x3[i]);
            }
        }

        [TestMethod]
        public void LtTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void LeTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]<=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GtTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void GeTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]>=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void EqTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]==0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void NeTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValueT(0.0f);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual((x1[i]!=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [TestMethod]
        public void CPowDouble()
        {
            var x1 = new DoubleTensor (10);
            var x2 = new DoubleTensor (10);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
                x2[i] = i+1;
            }

            var x3 = x1.CPow(x2);
            Assert.AreEqual(x1.Shape.Length,x3.Shape.Length);
            Assert.AreEqual(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.IsTrue(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(x1[i],x2[i]),x3[i]));
            }
        }
    }
}