using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Numerics.Tensors;
using Torch.SNT;

namespace Torch.SNT 
{
    [TestClass]
    public class ShortTorchTensorUnitTestGenerator
    {
        [TestMethod]
        public void TestCreationShort0D()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => ShortTorchTensor.Create());
        }

        [TestMethod]
        public void TestCreationShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreationShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesShort2D()
        {
            var x = ShortTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationShort3D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesShort3D()
        {
            var x = ShortTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationShort4D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesShort4D()
        {
            var x = ShortTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestFillShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);
            x.Fill((short)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }

        [TestMethod]
        public void TestFillBySetShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (short)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], (short)30);
            }
        }

        [TestMethod]
        public void TestFillBySetShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = (short)30;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], (short)30);
                }
            }
        }

        [TestMethod]
        public void TestFillEquivalanceShort2D()
        {
            var x = ShortTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    short tmp = (short)(i + j);
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
        public void TestCloneShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)1);

            Tensor<short> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], y[i]);
            }

            y[5] = (short)0;

            Assert.AreNotEqual(x[5], y[5]);
        }

        [TestMethod]
        public void TestCloneShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);
            x.Fill((short)1);

            Tensor<short> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (short)0;

            Assert.AreNotEqual(x[5, 5], y[5, 5]);
        }

        [TestMethod]
        public void TestCloneEmptyShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)1);

            Tensor<short> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (short)0);
            }
        }

        [TestMethod]
        public void TestCloneEmptyShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);
            x.Fill((short)1);

            Tensor<short> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.AreEqual(y[i, j], (short)0);
                }
            }
        }

        [TestMethod]
        public void TestReshapeShort1DFail()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)1);

            Assert.ThrowsException<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [TestMethod]
        public void TestReshapeShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (short)i;
            }

            Tensor<short> y = x.Reshape(new int[] { 10 });

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (short)i);
                Assert.AreEqual(x[i], (short)i);
            }
        }

        [TestMethod]
        public void TestReshapeShort1DPointToTheSameStorage()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (short)i;
            }

            Tensor<short> y = x.Reshape(new int[] { 10 });

            y[5] = (short)0;

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshapeShort2D()
        {
            var x = ShortTorchTensor.Create(5, 10);

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
        public void TestReshape2ShortD2()
        {
            var x = ShortTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    short tmp = (short)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<short> y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshapeShort2DPointToTheSameStorage()
        {
            var x = ShortTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    short tmp = (short)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<short> y = x.Reshape(new int[] { 10, 5 });

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

        [TestMethod]
        public void TestDanglingMemoryShort()
        {
            Memory<short> buffer;

            using (var x = ShortTorchTensor.Create(10))
            {
                x.Fill(33);
                buffer = x.Buffer;
            }

            var arr = buffer.ToArray();

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(33, arr[i]);
            }
        }
    }
    [TestClass]
    public class IntTorchTensorUnitTestGenerator
    {
        [TestMethod]
        public void TestCreationInt0D()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => IntTorchTensor.Create());
        }

        [TestMethod]
        public void TestCreationInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreationInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesInt2D()
        {
            var x = IntTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationInt3D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesInt3D()
        {
            var x = IntTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationInt4D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesInt4D()
        {
            var x = IntTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestFillInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);
            x.Fill((int)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }

        [TestMethod]
        public void TestFillBySetInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (int)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], (int)30);
            }
        }

        [TestMethod]
        public void TestFillBySetInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = (int)30;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], (int)30);
                }
            }
        }

        [TestMethod]
        public void TestFillEquivalanceInt2D()
        {
            var x = IntTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    int tmp = (int)(i + j);
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
        public void TestCloneInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)1);

            Tensor<int> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], y[i]);
            }

            y[5] = (int)0;

            Assert.AreNotEqual(x[5], y[5]);
        }

        [TestMethod]
        public void TestCloneInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);
            x.Fill((int)1);

            Tensor<int> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (int)0;

            Assert.AreNotEqual(x[5, 5], y[5, 5]);
        }

        [TestMethod]
        public void TestCloneEmptyInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)1);

            Tensor<int> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (int)0);
            }
        }

        [TestMethod]
        public void TestCloneEmptyInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);
            x.Fill((int)1);

            Tensor<int> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.AreEqual(y[i, j], (int)0);
                }
            }
        }

        [TestMethod]
        public void TestReshapeInt1DFail()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)1);

            Assert.ThrowsException<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [TestMethod]
        public void TestReshapeInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (int)i;
            }

            Tensor<int> y = x.Reshape(new int[] { 10 });

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (int)i);
                Assert.AreEqual(x[i], (int)i);
            }
        }

        [TestMethod]
        public void TestReshapeInt1DPointToTheSameStorage()
        {
            Tensor<int> x = IntTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (int)i;
            }

            Tensor<int> y = x.Reshape(new int[] { 10 });

            y[5] = (int)0;

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshapeInt2D()
        {
            var x = IntTorchTensor.Create(5, 10);

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
        public void TestReshape2IntD2()
        {
            var x = IntTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    int tmp = (int)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<int> y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshapeInt2DPointToTheSameStorage()
        {
            var x = IntTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    int tmp = (int)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<int> y = x.Reshape(new int[] { 10, 5 });

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

        [TestMethod]
        public void TestDanglingMemoryInt()
        {
            Memory<int> buffer;

            using (var x = IntTorchTensor.Create(10))
            {
                x.Fill(33);
                buffer = x.Buffer;
            }

            var arr = buffer.ToArray();

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(33, arr[i]);
            }
        }
    }
    [TestClass]
    public class LongTorchTensorUnitTestGenerator
    {
        [TestMethod]
        public void TestCreationLong0D()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => LongTorchTensor.Create());
        }

        [TestMethod]
        public void TestCreationLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreationLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesLong2D()
        {
            var x = LongTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationLong3D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesLong3D()
        {
            var x = LongTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationLong4D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesLong4D()
        {
            var x = LongTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestFillLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);
            x.Fill((long)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }

        [TestMethod]
        public void TestFillBySetLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (long)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], (long)30);
            }
        }

        [TestMethod]
        public void TestFillBySetLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = (long)30;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], (long)30);
                }
            }
        }

        [TestMethod]
        public void TestFillEquivalanceLong2D()
        {
            var x = LongTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    long tmp = (long)(i + j);
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
        public void TestCloneLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)1);

            Tensor<long> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], y[i]);
            }

            y[5] = (long)0;

            Assert.AreNotEqual(x[5], y[5]);
        }

        [TestMethod]
        public void TestCloneLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);
            x.Fill((long)1);

            Tensor<long> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (long)0;

            Assert.AreNotEqual(x[5, 5], y[5, 5]);
        }

        [TestMethod]
        public void TestCloneEmptyLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)1);

            Tensor<long> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (long)0);
            }
        }

        [TestMethod]
        public void TestCloneEmptyLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);
            x.Fill((long)1);

            Tensor<long> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.AreEqual(y[i, j], (long)0);
                }
            }
        }

        [TestMethod]
        public void TestReshapeLong1DFail()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)1);

            Assert.ThrowsException<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [TestMethod]
        public void TestReshapeLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (long)i;
            }

            Tensor<long> y = x.Reshape(new int[] { 10 });

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (long)i);
                Assert.AreEqual(x[i], (long)i);
            }
        }

        [TestMethod]
        public void TestReshapeLong1DPointToTheSameStorage()
        {
            Tensor<long> x = LongTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (long)i;
            }

            Tensor<long> y = x.Reshape(new int[] { 10 });

            y[5] = (long)0;

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshapeLong2D()
        {
            var x = LongTorchTensor.Create(5, 10);

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
        public void TestReshape2LongD2()
        {
            var x = LongTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    long tmp = (long)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<long> y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshapeLong2DPointToTheSameStorage()
        {
            var x = LongTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    long tmp = (long)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<long> y = x.Reshape(new int[] { 10, 5 });

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

        [TestMethod]
        public void TestDanglingMemoryLong()
        {
            Memory<long> buffer;

            using (var x = LongTorchTensor.Create(10))
            {
                x.Fill(33);
                buffer = x.Buffer;
            }

            var arr = buffer.ToArray();

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(33, arr[i]);
            }
        }
    }
    [TestClass]
    public class DoubleTorchTensorUnitTestGenerator
    {
        [TestMethod]
        public void TestCreationDouble0D()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => DoubleTorchTensor.Create());
        }

        [TestMethod]
        public void TestCreationDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreationDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesDouble2D()
        {
            var x = DoubleTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationDouble3D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesDouble3D()
        {
            var x = DoubleTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationDouble4D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesDouble4D()
        {
            var x = DoubleTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestFillDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);
            x.Fill((double)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }

        [TestMethod]
        public void TestFillBySetDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (double)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], (double)30);
            }
        }

        [TestMethod]
        public void TestFillBySetDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = (double)30;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], (double)30);
                }
            }
        }

        [TestMethod]
        public void TestFillEquivalanceDouble2D()
        {
            var x = DoubleTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    double tmp = (double)(i + j);
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
        public void TestCloneDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)1);

            Tensor<double> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], y[i]);
            }

            y[5] = (double)0;

            Assert.AreNotEqual(x[5], y[5]);
        }

        [TestMethod]
        public void TestCloneDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);
            x.Fill((double)1);

            Tensor<double> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (double)0;

            Assert.AreNotEqual(x[5, 5], y[5, 5]);
        }

        [TestMethod]
        public void TestCloneEmptyDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)1);

            Tensor<double> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (double)0);
            }
        }

        [TestMethod]
        public void TestCloneEmptyDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);
            x.Fill((double)1);

            Tensor<double> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.AreEqual(y[i, j], (double)0);
                }
            }
        }

        [TestMethod]
        public void TestReshapeDouble1DFail()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)1);

            Assert.ThrowsException<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [TestMethod]
        public void TestReshapeDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (double)i;
            }

            Tensor<double> y = x.Reshape(new int[] { 10 });

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (double)i);
                Assert.AreEqual(x[i], (double)i);
            }
        }

        [TestMethod]
        public void TestReshapeDouble1DPointToTheSameStorage()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (double)i;
            }

            Tensor<double> y = x.Reshape(new int[] { 10 });

            y[5] = (double)0;

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshapeDouble2D()
        {
            var x = DoubleTorchTensor.Create(5, 10);

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
        public void TestReshape2DoubleD2()
        {
            var x = DoubleTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    double tmp = (double)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<double> y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.AreEqual(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [TestMethod]
        public void TestReshapeDouble2DPointToTheSameStorage()
        {
            var x = DoubleTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    double tmp = (double)(i + j);
                    x[i, j] = tmp;
                }
            }

            Tensor<double> y = x.Reshape(new int[] { 10, 5 });

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

        [TestMethod]
        public void TestDanglingMemoryDouble()
        {
            Memory<double> buffer;

            using (var x = DoubleTorchTensor.Create(10))
            {
                x.Fill(33);
                buffer = x.Buffer;
            }

            var arr = buffer.ToArray();

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(33, arr[i]);
            }
        }
    }
    [TestClass]
    public class FloatTorchTensorUnitTestGenerator
    {
        [TestMethod]
        public void TestCreationFloat0D()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => FloatTorchTensor.Create());
        }

        [TestMethod]
        public void TestCreationFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreationFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesFloat2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationFloat3D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesFloat3D()
        {
            var x = FloatTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestCreationFloat4D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestShapeAndStridesFloat4D()
        {
            var x = FloatTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.AreEqual(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [TestMethod]
        public void TestFillFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill((float)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }

        [TestMethod]
        public void TestFillBySetFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (float)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], (float)30);
            }
        }

        [TestMethod]
        public void TestFillBySetFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = (float)30;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], (float)30);
                }
            }
        }

        [TestMethod]
        public void TestFillEquivalanceFloat2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    float tmp = (float)(i + j);
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
        public void TestCloneFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)1);

            Tensor<float> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], y[i]);
            }

            y[5] = (float)0;

            Assert.AreNotEqual(x[5], y[5]);
        }

        [TestMethod]
        public void TestCloneFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill((float)1);

            Tensor<float> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (float)0;

            Assert.AreNotEqual(x[5, 5], y[5, 5]);
        }

        [TestMethod]
        public void TestCloneEmptyFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)1);

            Tensor<float> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (float)0);
            }
        }

        [TestMethod]
        public void TestCloneEmptyFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill((float)1);

            Tensor<float> y = x.CloneEmpty();

            CollectionAssert.AreEqual(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.AreEqual(y[i, j], (float)0);
                }
            }
        }

        [TestMethod]
        public void TestReshapeFloat1DFail()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)1);

            Assert.ThrowsException<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [TestMethod]
        public void TestReshapeFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (float)i;
            }

            Tensor<float> y = x.Reshape(new int[] { 10 });

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], (float)i);
                Assert.AreEqual(x[i], (float)i);
            }
        }

        [TestMethod]
        public void TestReshapeFloat1DPointToTheSameStorage()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (float)i;
            }

            Tensor<float> y = x.Reshape(new int[] { 10 });

            y[5] = (float)0;

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(y[i], x[i]);
            }
        }

        [TestMethod]
        public void TestReshapeFloat2D()
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
        public void TestReshape2FloatD2()
        {
            var x = FloatTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    float tmp = (float)(i + j);
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
        public void TestReshapeFloat2DPointToTheSameStorage()
        {
            var x = FloatTorchTensor.Create(5, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    float tmp = (float)(i + j);
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

        [TestMethod]
        public void TestDanglingMemoryFloat()
        {
            Memory<float> buffer;

            using (var x = FloatTorchTensor.Create(10))
            {
                x.Fill(33);
                buffer = x.Buffer;
            }

            var arr = buffer.ToArray();

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(33, arr[i]);
            }
        }
    }
}
