using System;
using System.Numerics.Tensors;
using Xunit;

namespace Torch.SNT
{
    public class ShortTorchTensorUnitTestGenerator
    {
        [Fact]
        public void TestCreationShort0D()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => ShortTorchTensor.Create());
        }

        [Fact]
        public void TestCreationShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestCreationShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesShort2D()
        {
            var x = ShortTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationShort3D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10, 3);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesShort3D()
        {
            var x = ShortTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationShort4D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10, 3, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesShort4D()
        {
            var x = ShortTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestFillShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(30, x[i]);
            }
        }

        [Fact]
        public void TestFillShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);
            x.Fill((short)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(30, x[i, j]);
                }
            }
        }

        [Fact]
        public void TestFillBySetShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (short)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], (short)30);
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], (short)30);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], x.TorchSharpTensor[i, j]);
                }
            }
        }

        [Fact]
        public void TestCloneShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)1);

            Tensor<short> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], y[i]);
            }

            y[5] = (short)0;

            Assert.NotEqual(x[5], y[5]);
        }

        [Fact]
        public void TestCloneShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);
            x.Fill((short)1);

            Tensor<short> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (short)0;

            Assert.NotEqual(x[5, 5], y[5, 5]);
        }

        [Fact]
        public void TestCloneEmptyShort1D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)1);

            Tensor<short> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(y[i], (short)0);
            }
        }

        [Fact]
        public void TestCloneEmptyShort2D()
        {
            Tensor<short> x = ShortTorchTensor.Create(10, 10);
            x.Fill((short)1);

            Tensor<short> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.Equal(y[i, j], (short)0);
                }
            }
        }

        [Fact]
        public void TestReshapeShort1DFail()
        {
            Tensor<short> x = ShortTorchTensor.Create(10);
            x.Fill((short)1);

            Assert.Throws<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [Fact]
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
                Assert.Equal(y[i], (short)i);
                Assert.Equal(x[i], (short)i);
            }
        }

        [Fact]
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
                Assert.Equal(y[i], x[i]);
            }
        }

        [Fact]
        public void TestReshapeShort2D()
        {
            var x = ShortTorchTensor.Create(5, 10);

            var y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }

            Equals(x.Dimensions.ToArray(), new int[] { 5, 10 });
            Equals(x.Strides.ToArray(), new int[] { 1, 10 });
            Equals(y.Dimensions.ToArray(), new int[] { 10, 5 });
            Equals(y.Strides.ToArray(), new int[] { 1, 5 });
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                Assert.Equal(33, arr[i]);
            }
        }
    }

    public class IntTorchTensorUnitTestGenerator
    {
        [Fact]
        public void TestCreationInt0D()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => IntTorchTensor.Create());
        }

        [Fact]
        public void TestCreationInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestCreationInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesInt2D()
        {
            var x = IntTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationInt3D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10, 3);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesInt3D()
        {
            var x = IntTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationInt4D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10, 3, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesInt4D()
        {
            var x = IntTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestFillInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(30, x[i]);
            }
        }

        [Fact]
        public void TestFillInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);
            x.Fill((int)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(30, x[i, j]);
                }
            }
        }

        [Fact]
        public void TestFillBySetInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (int)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], (int)30);
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], (int)30);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], x.TorchSharpTensor[i, j]);
                }
            }
        }

        [Fact]
        public void TestCloneInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)1);

            Tensor<int> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], y[i]);
            }

            y[5] = (int)0;

            Assert.NotEqual(x[5], y[5]);
        }

        [Fact]
        public void TestCloneInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);
            x.Fill((int)1);

            Tensor<int> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (int)0;

            Assert.NotEqual(x[5, 5], y[5, 5]);
        }

        [Fact]
        public void TestCloneEmptyInt1D()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)1);

            Tensor<int> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(y[i], (int)0);
            }
        }

        [Fact]
        public void TestCloneEmptyInt2D()
        {
            Tensor<int> x = IntTorchTensor.Create(10, 10);
            x.Fill((int)1);

            Tensor<int> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.Equal(y[i, j], (int)0);
                }
            }
        }

        [Fact]
        public void TestReshapeInt1DFail()
        {
            Tensor<int> x = IntTorchTensor.Create(10);
            x.Fill((int)1);

            Assert.Throws<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [Fact]
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
                Assert.Equal(y[i], (int)i);
                Assert.Equal(x[i], (int)i);
            }
        }

        [Fact]
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
                Assert.Equal(y[i], x[i]);
            }
        }

        [Fact]
        public void TestReshapeInt2D()
        {
            var x = IntTorchTensor.Create(5, 10);

            var y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }

            Equals(x.Dimensions.ToArray(), new int[] { 5, 10 });
            Equals(x.Strides.ToArray(), new int[] { 1, 10 });
            Equals(y.Dimensions.ToArray(), new int[] { 10, 5 });
            Equals(y.Strides.ToArray(), new int[] { 1, 5 });
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                Assert.Equal(33, arr[i]);
            }
        }
    }

    public class LongTorchTensorUnitTestGenerator
    {
        [Fact]
        public void TestCreationLong0D()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => LongTorchTensor.Create());
        }

        [Fact]
        public void TestCreationLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestCreationLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesLong2D()
        {
            var x = LongTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationLong3D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10, 3);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesLong3D()
        {
            var x = LongTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationLong4D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10, 3, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesLong4D()
        {
            var x = LongTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestFillLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(30, x[i]);
            }
        }

        [Fact]
        public void TestFillLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);
            x.Fill((long)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(30, x[i, j]);
                }
            }
        }

        [Fact]
        public void TestFillBySetLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (long)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], (long)30);
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], (long)30);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], x.TorchSharpTensor[i, j]);
                }
            }
        }

        [Fact]
        public void TestCloneLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)1);

            Tensor<long> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], y[i]);
            }

            y[5] = (long)0;

            Assert.NotEqual(x[5], y[5]);
        }

        [Fact]
        public void TestCloneLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);
            x.Fill((long)1);

            Tensor<long> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (long)0;

            Assert.NotEqual(x[5, 5], y[5, 5]);
        }

        [Fact]
        public void TestCloneEmptyLong1D()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)1);

            Tensor<long> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(y[i], (long)0);
            }
        }

        [Fact]
        public void TestCloneEmptyLong2D()
        {
            Tensor<long> x = LongTorchTensor.Create(10, 10);
            x.Fill((long)1);

            Tensor<long> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.Equal(y[i, j], (long)0);
                }
            }
        }

        [Fact]
        public void TestReshapeLong1DFail()
        {
            Tensor<long> x = LongTorchTensor.Create(10);
            x.Fill((long)1);

            Assert.Throws<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [Fact]
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
                Assert.Equal(y[i], (long)i);
                Assert.Equal(x[i], (long)i);
            }
        }

        [Fact]
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
                Assert.Equal(y[i], x[i]);
            }
        }

        [Fact]
        public void TestReshapeLong2D()
        {
            var x = LongTorchTensor.Create(5, 10);

            var y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }

            Equals(x.Dimensions.ToArray(), new int[] { 5, 10 });
            Equals(x.Strides.ToArray(), new int[] { 1, 10 });
            Equals(y.Dimensions.ToArray(), new int[] { 10, 5 });
            Equals(y.Strides.ToArray(), new int[] { 1, 5 });
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                Assert.Equal(33, arr[i]);
            }
        }
    }

    public class DoubleTorchTensorUnitTestGenerator
    {
        [Fact]
        public void TestCreationDouble0D()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => DoubleTorchTensor.Create());
        }

        [Fact]
        public void TestCreationDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestCreationDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesDouble2D()
        {
            var x = DoubleTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationDouble3D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10, 3);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesDouble3D()
        {
            var x = DoubleTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationDouble4D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10, 3, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesDouble4D()
        {
            var x = DoubleTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestFillDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(30, x[i]);
            }
        }

        [Fact]
        public void TestFillDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);
            x.Fill((double)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(30, x[i, j]);
                }
            }
        }

        [Fact]
        public void TestFillBySetDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (double)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], (double)30);
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], (double)30);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], x.TorchSharpTensor[i, j]);
                }
            }
        }

        [Fact]
        public void TestCloneDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)1);

            Tensor<double> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], y[i]);
            }

            y[5] = (double)0;

            Assert.NotEqual(x[5], y[5]);
        }

        [Fact]
        public void TestCloneDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);
            x.Fill((double)1);

            Tensor<double> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (double)0;

            Assert.NotEqual(x[5, 5], y[5, 5]);
        }

        [Fact]
        public void TestCloneEmptyDouble1D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)1);

            Tensor<double> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(y[i], (double)0);
            }
        }

        [Fact]
        public void TestCloneEmptyDouble2D()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10, 10);
            x.Fill((double)1);

            Tensor<double> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.Equal(y[i, j], (double)0);
                }
            }
        }

        [Fact]
        public void TestReshapeDouble1DFail()
        {
            Tensor<double> x = DoubleTorchTensor.Create(10);
            x.Fill((double)1);

            Assert.Throws<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [Fact]
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
                Assert.Equal(y[i], (double)i);
                Assert.Equal(x[i], (double)i);
            }
        }

        [Fact]
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
                Assert.Equal(y[i], x[i]);
            }
        }

        [Fact]
        public void TestReshapeDouble2D()
        {
            var x = DoubleTorchTensor.Create(5, 10);

            var y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }

            Equals(x.Dimensions.ToArray(), new int[] { 5, 10 });
            Equals(x.Strides.ToArray(), new int[] { 1, 10 });
            Equals(y.Dimensions.ToArray(), new int[] { 10, 5 });
            Equals(y.Strides.ToArray(), new int[] { 1, 5 });
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                Assert.Equal(33, arr[i]);
            }
        }
    }

    public class FloatTorchTensorUnitTestGenerator
    {
        [Fact]
        public void TestCreationFloat0D()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => FloatTorchTensor.Create());
        }

        [Fact]
        public void TestCreationFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestCreationFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesFloat2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationFloat3D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesFloat3D()
        {
            var x = FloatTorchTensor.Create(10, 10, 3);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestCreationFloat4D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3, 10);

            Assert.NotNull(x);
        }

        [Fact]
        public void TestShapeAndStridesFloat4D()
        {
            var x = FloatTorchTensor.Create(10, 10, 3, 10);

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }
        }

        [Fact]
        public void TestFillFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(30, x[i]);
            }
        }

        [Fact]
        public void TestFillFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill((float)30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(30, x[i, j]);
                }
            }
        }

        [Fact]
        public void TestFillBySetFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = (float)30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], (float)30);
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], (float)30);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], x.TorchSharpTensor[i, j]);
                }
            }
        }

        [Fact]
        public void TestCloneFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)1);

            Tensor<float> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(x[i], y[i]);
            }

            y[5] = (float)0;

            Assert.NotEqual(x[5], y[5]);
        }

        [Fact]
        public void TestCloneFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill((float)1);

            Tensor<float> y = x.Clone();

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.Equal(x[i, j], y[i, j]);
                }
            }

            y[5, 5] = (float)0;

            Assert.NotEqual(x[5, 5], y[5, 5]);
        }

        [Fact]
        public void TestCloneEmptyFloat1D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)1);

            Tensor<float> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.Equal(y[i], (float)0);
            }
        }

        [Fact]
        public void TestCloneEmptyFloat2D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10);
            x.Fill((float)1);

            Tensor<float> y = x.CloneEmpty();

            Assert.Equal(y.Dimensions.ToArray(), x.Dimensions.ToArray());

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[0]; j++)
                {
                    Assert.Equal(y[i, j], (float)0);
                }
            }
        }

        [Fact]
        public void TestReshapeFloat1DFail()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill((float)1);

            Assert.Throws<ArgumentException>(() => x.Reshape(new int[] { 9 }));
        }

        [Fact]
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
                Assert.Equal(y[i], (float)i);
                Assert.Equal(x[i], (float)i);
            }
        }

        [Fact]
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
                Assert.Equal(y[i], x[i]);
            }
        }

        [Fact]
        public void TestReshapeFloat2D()
        {
            var x = FloatTorchTensor.Create(5, 10);

            var y = x.Reshape(new int[] { 10, 5 });

            for (int i = 0; i < 2; i++)
            {
                Assert.Equal(x.Dimensions[0], (int)x.TorchSharpTensor.GetTensorDimension(0));
                Assert.Equal(x.Strides[0], (int)x.TorchSharpTensor.GetTensorStride(0));
            }

            Equals(x.Dimensions.ToArray(), new int[] { 5, 10 });
            Equals(x.Strides.ToArray(), new int[] { 1, 10 });
            Equals(y.Dimensions.ToArray(), new int[] { 10, 5 });
            Equals(y.Strides.ToArray(), new int[] { 1, 5 });
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                    Assert.Equal(x[i, j], y[i * 2 + j / 5, j % 5]);
                }
            }
        }

        [Fact]
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
                Assert.Equal(33, arr[i]);
            }
        }
    }
}
