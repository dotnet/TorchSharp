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
            var x2 = FloatTensor.NewWithStorage1d(storage0,UIntPtr.Zero,size,1);      
            var x3 = x2.NewWithStorage1d((UIntPtr)size,size,1);      

            Assert.AreNotEqual(IntPtr.Zero,x1.Data);
            Assert.AreNotEqual(IntPtr.Zero,x2.Data);
            Assert.AreNotEqual(IntPtr.Zero,x3.Data);

            Assert.AreNotEqual(IntPtr.Zero,x1.Storage);
            Assert.AreNotEqual(IntPtr.Zero,x2.Storage);
            Assert.AreNotEqual(IntPtr.Zero,x3.Storage);
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
        public void NewFloatTensorWithStorage1D()
        {
            var x1 = new FloatTensor(9);
            var x2 = x1.NewWithStorage1d((UIntPtr)0, 9, 1);

            for (int i = 0; i < x1.Shape[0]; i++)
            {
                Assert.AreEqual(x1[i], x2[i]);
            }
        }

        [TestMethod]
        public void TestReshapeFloat1D()
        {
            var x = new FloatTensor(10);

            for (int i = 0; i < x.Shape[0]; i++)
            {
                x[i] = i;
            }

            var y = x.NewWithStorage1d((UIntPtr)0, 10, 1);

            for (int i = 0; i < x.Shape[0]; i++)
            {
                Assert.AreEqual(y[i], i);
                Assert.AreEqual(x[i], i);
            }
        }

        [TestMethod]
        public void TestReshapeFloat1DPointToTheSameStorage()
        {
            var x = new FloatTensor(10);
            for (int i = 0; i < x.Shape[0]; i++)
            {
                x[i] = i;
            }

            var y = x.NewWithStorage1d((UIntPtr)0, 10, 1);

            y[5] = 0;

            for (int i = 0; i < x.Shape[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshapeFloat2D()
        {
            var x = new FloatTensor(5, 10);
            var y = x.NewWithStorage2d((UIntPtr)0, 10, 1, 5, 10 );

            Equals(x.Shape, new int[] { 5, 10 });
            Equals(y.Shape, new int[] { 10, 5 });
        }

        [TestMethod]
        public void TestReshape2FloatD2()
        {
            var x = new FloatTensor(5, 10);
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    x[i, j] = i + j;
                }
            }
            var y = x.NewWithStorage2d((UIntPtr)0, 10, 5, 5, 1);
            
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshapeFloat2DPointToTheSameStorage()
        {
            var x = new FloatTensor(5, 10);

            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    float tmp = i + j;
                    x[i, j] = tmp;
                }
            }

            var y = x.NewWithStorage2d((UIntPtr)0, 10, 5, 5, 1);

            x[4, 9] = 0;
            y[3, 4] = 0;

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshapeFloat3DPointToTheSameStorage()
        {
            var x = new FloatTensor(10, 10, 10);

            x.Fill(1);

            var y = x.NewWithStorage1d((UIntPtr)0, 1000, 1);

            y[5] = 0;

            int count = 0;

            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    for (int k = 0; k < x.Shape[2]; k++)
                    {
                        Assert.AreEqual(x[i, j, k], y[count++]);
                    }
                }
            }
        }

        [TestMethod]
        public void TestReshapeFloat4DPointToTheSameStorage()
        {
            var x = new FloatTensor(10, 10, 10, 5);

            x.Fill(1);

            var y = x.NewWithStorage1d((UIntPtr)0, 5000, 1);

            y[567] = 0;

            int count = 0;

            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    for (int k = 0; k < x.Shape[2]; k++)
                    {
                        for (int l = 0; l < x.Shape[3]; l++)
                        {
                            Assert.AreEqual(x[i, j, k, l], y[count++]);
                        }
                    }
                }
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
