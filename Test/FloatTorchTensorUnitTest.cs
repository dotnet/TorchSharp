using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using TorchSharp;

namespace TorchSharp.Tests
{
    [TestClass]
    public class FloatTorchTensorUnitTest
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
        public void TestCreation3D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation4D()
        {
            Tensor<float> x = FloatTorchTensor.Create(10, 10, 3, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestFill1d()
        {
            Tensor<float> x = FloatTorchTensor.Create(10);
            x.Fill(30.0f);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30.0f);
            }
        }

        [TestMethod]
        public void TestFill2d()
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

        internal class FloatTorchTensorMockup : DenseTensor<float>
        {
            private readonly FloatTensor inner;

            public static unsafe FloatTorchTensorMockup Create(params int[] sizes)
            {
                var memLen = 0;
                var shape = sizes;

                if (sizes.Length == 0)
                {
                    shape = new int[] { 0 };
                }
                else
                {
                    memLen = sizes.Aggregate((a, b) => a * b);
                }

                var inner = new FloatTensor(sizes.Select(x => (long)x).ToArray());
                var mem = new NativeMemory<float>((void*)inner.Data, memLen);

                return new FloatTorchTensorMockup(mem.Memory, shape, inner);
            }

            public FloatTorchTensorMockup(Memory<float> memory, ReadOnlySpan<int> dimensions, FloatTensor inner) : base(memory, dimensions)
            {
                this.inner = inner;
            }

            public float Get(long x0, long x1)
            {
                return inner[x0, x1];
            }
        }

        [TestMethod]
        public void TestFillEquivalance2D()
        {
            var x = FloatTorchTensorMockup.Create(10, 10);
            var rand = new Random();
            
            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    var tmp = (float)rand.NextDouble();
                    x[i, j] = tmp;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], x.Get(i, j));
                }
            }
        }
    }
}
