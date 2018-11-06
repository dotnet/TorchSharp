using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Numerics.Tensors;
using Torch.SNT;

namespace Torch.SNT
{
    [TestClass]
    public class TorchTensorUnitTestGenerator
    {
        [TestMethod]
        public void TestCreation0D()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => FloatTorchTensor.Create());
        }

        [TestMethod]
        public void TestCreation1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStrides2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreation3D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStrides3D()
        {
            var x = FloatTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreation4D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStrides4D()
        {
            var x = FloatTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestFill1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill(30.0f);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30.0f);
            }
        }

        [TestMethod]
        public void TestFill2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill(30.0f);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30.0f);
                }
            }
        }

        [TestMethod]
        public void TestFillBySet1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = 30.0f;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30.0f);
            }
        }

        [TestMethod]
        public void TestFillBySet2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = 30.0f;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30.0f);
                }
            }
        }

        [TestMethod]
        public void TestFillEquivalance2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    float tmp = i + j;
                    x[i, j] = tmp;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], x.TorchSharpTensor[i, j]);
                }
            }
        }

        [TestMethod]
        public void TestClone1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill(1f);

            Tensor<float> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], y[i]);
            }

            y[5] = 0f;

            Assert.AreNotEqual(x[5], y[5]);
        }

        [TestMethod]
        public void TestClone2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill(1f);

            Tensor<float> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = 0f;

            Assert.AreNotEqual(x[5, 5], y[5, 5]);
        }

        [TestMethod]
        public void TestCloneEmpty1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill(1f);

            Tensor<float> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], 0f);
            }
        }

        [TestMethod]
        public void TestCloneEmpty2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill(1f);

            Tensor<float> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.AreEqual(y[i, j], 0f);
                }
            }
        }

        [TestMethod]
        public void TestReshape1DFail()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill(1f);

            Assert.ThrowsException<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [TestMethod]
        public void TestReshape1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = i;
            }

            Tensor<float> y = x.Reshape(new int[] { 10 });

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], i);
                Assert.AreEqual(x[i], i);
            }
        }

        [TestMethod]
        public void TestReshape1DPointToTheSameStorage()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = i;
            }

            Tensor<float> y = x.Reshape(new int[] { 10 });

            y[5] = 0;

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshape2D()
        {
            var x = FloatTorchTensor.Create(5, 10);

            var y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }

            Equals(x.Dimensions.ToArray(), new int[] { 5, 10 });
            Equals(x.Strides.ToArray(), new int[] { 1, 10 });
            Equals(y.Dimensions.ToArray(), new int[] { 10, 5 });
            Equals(y.Strides.ToArray(), new int[] { 1, 5 });
        }

        [TestMethod]
        public void TestReshape2D2()
        {
            var x = FloatTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    float tmp = i + j;
                    x[i, j] = tmp;
                }
            }

            Tensor<float> y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshape2DPointToTheSameStorage()
        {
            var x = FloatTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    float tmp = i + j;
                    x[i, j] = tmp;
                }
            }

            Tensor<float> y = x.Reshape(new int[] { 10, 5 });

            x[4, 9] = 0;
            y[3, 4] = 0;

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }
    }
}
