using AtenSharp;
using System;
using Xunit;

namespace AtenSharp.Test
{
    public class MatrixAndVectorOps
    {
#region Float Tests
        [Fact]
        public void SumFloat1d()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
            }

            var x3 = x1.Sum(0,true);
            Assert.True(1 == x3.Shape.Length);
            Assert.True(1 == x3.Shape[0]);
            Assert.Equal(100f, x3[0]);

            var x4 = x1.Sum(0,false);
            Assert.True(0 == x4.Shape.Length);
            // 0-dim tensors still have a single value, which can be fetched.
            Assert.Equal(100f, x4[0]);
        }

        [Fact]
        public void SumFloat2d()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Sum(1,true);
            Assert.Equal(2,x3.Shape.Length);
            Assert.True(1 == x3.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(100f, x3[i,0]);
            }

            var x4 = x1.Sum(1,false);
            Assert.True(1 == x4.Shape.Length);
            Assert.Equal(10, x4.Shape[0]);
            Assert.Equal(100f, x4[0]);
        }

        [Fact]
        public void ProdFloat1d()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
            }

            var x3 = x1.Prod(0,true);
            Assert.True(1 == x3.Shape.Length);
            Assert.True(1 == x3.Shape[0]);
            Assert.Equal(654729075f, x3[0]);

            var x4 = x1.Prod(0,false);
            Assert.True(0 == x4.Shape.Length);
            // 0-dim tensors still have a single value, which can be fetched.
            Assert.Equal(654729075f, x4[0]);
        }

        [Fact]
        public void ProdFloat2d()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Prod(1,true);
            Assert.Equal(2,x3.Shape.Length);
            Assert.True(1 == x3.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(654729075f, x3[i,0]);
            }

            var x4 = x1.Prod(1,false);
            Assert.True(1 == x4.Shape.Length);
            Assert.Equal(10, x4.Shape[0]);
            Assert.Equal(654729075f, x4[0]);
        }

        [Fact]
        public void MaxFloat()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Max(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(19f, x3.Item1[i,0]);
                Assert.Equal(9,   x3.Item2[i,0]);
            }

            var x4 = x1.Max(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(19f, x4.Item1[0]);
            Assert.Equal(9, x4.Item2[0]);
        }

        [Fact]
        public void MinFloat()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Min(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(1f, x3.Item1[i,0]);
                Assert.Equal(0,  x3.Item2[i,0]);
            }

            var x4 = x1.Min(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(1f, x4.Item1[0]);
            Assert.Equal(0,  x4.Item2[0]);
        }

        [Fact]
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

            Assert.Equal(715f, dot);
        }

        [Fact]
        public void CrossProductFloat()
        {
            var x1 = new FloatTensor(3);
            var x2 = new FloatTensor(3);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = i + 1;
                x2[i] = 3 - i;
            }

            var cross = x1.CrossProduct(x2);
            Assert.Equal(-4, cross[0]);
            Assert.Equal(8, cross[1]);
            Assert.Equal(-4, cross[0]);
        }
        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(x1[i]*x2[i],x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(x1[i]/x2[i],x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(x1[i],x2[i]),x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],x2[i]),x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],x2[i]),x3[i]));
            }
        }

        [Fact]
        public void CMaxValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMaxValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],0.0f),x3[i]));
            }
        }

        [Fact]
        public void CMinValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMinValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],0.0f),x3[i]));
            }
        }

        [Fact]
        public void LtValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void LeValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void GtValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void GeValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void EqValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void NeValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void LtTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void LeTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void GtTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void GeTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void EqTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void NeTValueFloat()
        {
            var x1 = new FloatTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void ModeFloat()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
                x1[i,5] = 17;
            }

            var x3 = x1.Mode(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(17, x3.Item1[i,0]);
                Assert.Equal(8,  x3.Item2[i,0]);
            }

            var x4 = x1.Mode(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(17, x4.Item1[0]);
            Assert.Equal(8, x4.Item2[0]);
        }

        [Fact]
        public void MedianFloat()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
                x1[i,5] = 17;
            }

            var x3 = x1.Median(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(9, x3.Item1[i,0]);
                Assert.Equal(4,  x3.Item2[i,0]);
            }

            var x4 = x1.Median(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(9, x4.Item1[0]);
            Assert.Equal(4, x4.Item2[0]);
        }

        [Fact]
        public void kthValueFloat()
        {
            var x1 = new FloatTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.KthValue(7,1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(x1[0,6], x3.Item1[i,0]);
                Assert.Equal(6, x3.Item2[i,0]);
            }

            var x4 = x1.KthValue(3,1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(x1[0,2], x4.Item1[0]);
            Assert.Equal(2, x4.Item2[0]);
        }
#endregion

#region Double Tests
        [Fact]
        public void SumDouble1d()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
            }

            var x3 = x1.Sum(0,true);
            Assert.True(1 == x3.Shape.Length);
            Assert.True(1 == x3.Shape[0]);
            Assert.Equal(100f, x3[0]);

            var x4 = x1.Sum(0,false);
            Assert.True(0 == x4.Shape.Length);
            // 0-dim tensors still have a single value, which can be fetched.
            Assert.Equal(100f, x4[0]);
        }

        [Fact]
        public void SumDouble2d()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Sum(1,true);
            Assert.Equal(2,x3.Shape.Length);
            Assert.True(1 == x3.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(100f, x3[i,0]);
            }

            var x4 = x1.Sum(1,false);
            Assert.True(1 == x4.Shape.Length);
            Assert.Equal(10, x4.Shape[0]);
            Assert.Equal(100f, x4[0]);
        }

        [Fact]
        public void ProdDouble1d()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = 2*i+1;
            }

            var x3 = x1.Prod(0,true);
            Assert.True(1 == x3.Shape.Length);
            Assert.True(1 == x3.Shape[0]);
            Assert.Equal(654729075, x3[0]);

            var x4 = x1.Prod(0,false);
            Assert.True(0 == x4.Shape.Length);
            // 0-dim tensors still have a single value, which can be fetched.
            Assert.Equal(654729075, x4[0]);
        }

        [Fact]
        public void ProdDouble2d()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Prod(1,true);
            Assert.Equal(2,x3.Shape.Length);
            Assert.True(1 == x3.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(654729075, x3[i,0]);
            }

            var x4 = x1.Prod(1,false);
            Assert.True(1 == x4.Shape.Length);
            Assert.Equal(10, x4.Shape[0]);
            Assert.Equal(654729075, x4[0]);
        }

        [Fact]
        public void MaxDouble()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Max(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(19, x3.Item1[i,0]);
                Assert.Equal(9,   x3.Item2[i,0]);
            }

            var x4 = x1.Max(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(19, x4.Item1[0]);
            Assert.Equal(9, x4.Item2[0]);
        }

        [Fact]
        public void MinDouble()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.Min(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(1 ==  x3.Item1[i,0]);
                Assert.Equal(0,  x3.Item2[i,0]);
            }

            var x4 = x1.Min(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.True(1 ==  x4.Item1[0]);
            Assert.Equal(0,  x4.Item2[0]);
        }

        [Fact]
        public void ModeDouble()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
                x1[i,5] = 17;
            }

            var x3 = x1.Mode(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(17, x3.Item1[i,0]);
                Assert.Equal(8,  x3.Item2[i,0]);
            }

            var x4 = x1.Mode(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(17, x4.Item1[0]);
            Assert.Equal(8, x4.Item2[0]);
        }

        [Fact]
        public void MedianDouble()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
                x1[i,5] = 17;
            }

            var x3 = x1.Median(1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(9, x3.Item1[i,0]);
                Assert.Equal(4,  x3.Item2[i,0]);
            }

            var x4 = x1.Median(1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(9, x4.Item1[0]);
            Assert.Equal(4, x4.Item2[0]);
        }

        [Fact]
        public void kthValueDouble()
        {
            var x1 = new DoubleTensor (10,10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                for (var j = 0; j < x1.Shape[0]; ++j)
                {
                     x1[i,j] = 2*j+1;
                }
            }

            var x3 = x1.KthValue(7,1,true);
            Assert.Equal(2,x3.Item1.Shape.Length);
            Assert.True(1 == x3.Item1.Shape[1]);
            Assert.Equal(2,x3.Item2.Shape.Length);
            Assert.True(1 == x3.Item2.Shape[1]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal(x1[0,6], x3.Item1[i,0]);
                Assert.Equal(6, x3.Item2[i,0]);
            }

            var x4 = x1.KthValue(3,1,false);
            Assert.True(1 ==  x4.Item1.Shape.Length);
            Assert.Equal(10, x4.Item1.Shape[0]);
            Assert.True(1 ==  x4.Item2.Shape.Length);
            Assert.Equal(10, x4.Item2.Shape[0]);
            Assert.Equal(x1[0,2], x4.Item1[0]);
            Assert.Equal(2, x4.Item2[0]);
        }

        [Fact]
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

            Assert.Equal(715, dot);
        }

        [Fact]
        public void CrossProductDouble()
        {

            var x1 = new DoubleTensor(3);
            var x2 = new DoubleTensor(3);

            for (var i = 0; i < x2.Shape[0]; ++i)
            {
                x1[i] = i + 1;
                x2[i] = 3 - i;
            }

            var cross = x1.CrossProduct(x2);
            Assert.Equal(-4, cross[0]);
            Assert.Equal(8, cross[1]);
            Assert.Equal(-4, cross[0]);
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(x1[i]*x2[i],x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(x1[i]/x2[i],x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],x2[i]),x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],x2[i]),x3[i]));
            }
        }

        [Fact]
        public void CMaxValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMaxValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Max(x1[i],0.0f),x3[i]));
            }
        }

        [Fact]
        public void CMinValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.CMinValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Min(x1[i],0.0f),x3[i]));
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=x2[i])?1:0,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=x2[i])?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void LtValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void LeValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void GtValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void GeValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void EqValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void NeValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValue(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=0.0f)?1:0,x3[i]);
            }
        }

        [Fact]
        public void LtTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LtValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void LeTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.LeValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]<=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void GtTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GtValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void GeTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.GeValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]>=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void EqTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.EqValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]==0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
        public void NeTValueDouble()
        {
            var x1 = new DoubleTensor (10);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = -2*i+10;
            }

            var x3 = x1.NeValueT(0.0f);
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.Equal((x1[i]!=0.0f)?1.0f:0.0f,x3[i]);
            }
        }

        [Fact]
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
            Assert.Equal(x1.Shape.Length,x3.Shape.Length);
            Assert.Equal(x1.Shape[0],x3.Shape[0]);
            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.True(BasicTensorAPI.IsApproximatelyEqual(Math.Pow(x1[i],x2[i]),x3[i]));
            }
        }
#endregion
    }
}