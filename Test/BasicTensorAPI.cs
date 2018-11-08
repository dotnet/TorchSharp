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
        public void GetFloatTensorData()
        {
            const int size = 10;
            var storage0 = new TorchSharp.FloatTensor.FloatStorage(2*size);
            
            var x1 = new FloatTensor(size);
            var x2 = FloatTensor.NewWithStorage1d(storage0,IntPtr.Zero,size,1);      
            var x3 = x2.NewWithStorage1d((IntPtr)size,size,1);      

            Assert.AreNotEqual(IntPtr.Zero,x1.Data);
            Assert.AreNotEqual(IntPtr.Zero,x2.Data);
            Assert.AreNotEqual(IntPtr.Zero,x3.Data);

            Assert.AreNotEqual(IntPtr.Zero,x1.Storage);
            Assert.AreNotEqual(IntPtr.Zero,x2.Storage);
            Assert.AreNotEqual(IntPtr.Zero,x3.Storage);

            Assert.AreNotEqual(storage0,x1.Storage);
            Assert.AreEqual(storage0,x2.Storage);
            Assert.AreEqual(storage0,x3.Storage);
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
        public void DiagFloat()
        {
            var x1 = new FloatTensor (9);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = i+1;
            }

            x1.Resize2d(3,3);

            var diag0 = x1.Diagonal(0);
            var diag1 = x1.Diagonal(1);
            
            Assert.AreEqual(1, diag0.Shape.Length);
            Assert.AreEqual(3, diag0.Shape[0]);
            Assert.AreEqual(1, diag1.Shape.Length);
            Assert.AreEqual(2, diag1.Shape[0]);

            Assert.AreEqual(1, diag0[0]);
            Assert.AreEqual(5, diag0[1]);
            Assert.AreEqual(9, diag0[2]);

            Assert.AreEqual(2, diag1[0]);
            Assert.AreEqual(6, diag1[1]);
        }

        [TestMethod]
        public void ConcatenateFloat()
        {
            var x1 = new FloatTensor (9);
            var x2 = new FloatTensor (9);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = i+1;
                x2[i] = i+1+x1.Shape[0];
            }

            var x3 = x1.Concatenate(x2,0);

            Assert.AreEqual(1,  x3.Shape.Length);
            Assert.AreEqual(18, x3.Shape[0]);

            for (var i = 0; i < x3.Shape[0]; ++i)
            {
                Assert.AreEqual(i+1,x3[i]);
            }
        }

        [TestMethod]
        public void IdentityFloat()
        {
            var x1 = FloatTensor.Eye(10,10);

            for (int i = 0; i < 10; ++i)
            {
                for (int j = 0; j < 10; ++j)
                {
                    if (i == j)
                        Assert.AreEqual(1,x1[i,j]);
                    else
                        Assert.AreEqual(0,x1[i,j]);
                }
            }
        }

        [TestMethod]
        public void RangeFloat()
        {
            var x1 = FloatTensor.Range(0f, 100f, 1f);

            for (int i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual(i, x1[i]);
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
        public void DiagDouble()
        {
            var x1 = new DoubleTensor (9);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = i+1;
            }

            x1.Resize2d(3,3);

            var diag0 = x1.Diagonal(0);
            var diag1 = x1.Diagonal(1);
            
            Assert.AreEqual(1, diag0.Shape.Length);
            Assert.AreEqual(3, diag0.Shape[0]);
            Assert.AreEqual(1, diag1.Shape.Length);
            Assert.AreEqual(2, diag1.Shape[0]);

            Assert.AreEqual(1, diag0[0]);
            Assert.AreEqual(5, diag0[1]);
            Assert.AreEqual(9, diag0[2]);

            Assert.AreEqual(2, diag1[0]);
            Assert.AreEqual(6, diag1[1]);
        }

        [TestMethod]
        public void ConcatenateDouble()
        {
            var x1 = new DoubleTensor (9);
            var x2 = new DoubleTensor (9);

            for (var i = 0; i < x1.Shape[0]; ++i)
            {
                x1[i] = i+1;
                x2[i] = i+1+x1.Shape[0];
            }

            var x3 = x1.Concatenate(x2,0);

            Assert.AreEqual(1,  x3.Shape.Length);
            Assert.AreEqual(18, x3.Shape[0]);

            for (var i = 0; i < x3.Shape[0]; ++i)
            {
                Assert.AreEqual(i+1,x3[i]);
            }
        }

        [TestMethod]
        public void IdentityDouble()
        {
            var x1 = DoubleTensor.Eye(10,10);

            for (int i = 0; i < 10; ++i)
            {
                for (int j = 0; j < 10; ++j)
                {
                    if (i == j)
                        Assert.AreEqual(1,x1[i,j]);
                    else
                        Assert.AreEqual(0,x1[i,j]);
                }
            }
        }

        [TestMethod]
        public void RangeDouble()
        {
            var x1 = DoubleTensor.Range(0f, 100f, 1f);

            for (int i = 0; i < x1.Shape[0]; ++i)
            {
                Assert.AreEqual(i, x1[i]);
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
    }
}
