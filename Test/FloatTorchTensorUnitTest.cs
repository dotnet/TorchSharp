using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TorchSharp;

namespace Test
{
    [TestClass]
    public class FloatTorchTensorUnitTest
    {
        [TestMethod]
        public void TestCreation0D()
        {
            var x = FloatTorchTensor.Create();

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation1D()
        {
            var x = FloatTorchTensor.Create(10);
            
             Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestFill1d()
        {
            var x = FloatTorchTensor.Create(10);
            x.Fill(30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFill2d()
        {
            var x = FloatTorchTensor.Create(10, 10);
            x.Fill(30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }

        [TestMethod]
        public void TestFillBySet1D()
        {
            var x = FloatTorchTensor.Create(10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                x[i] = 30;
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillBySet2D()
        {
            var x = FloatTorchTensor.Create(10, 10);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    x[i, j] = 30;
                }
            }

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                for (int j = 0; j < x.Dimensions[1]; j++)
                {
                    Assert.AreEqual(x[i, j], 30);
                }
            }
        }
    }
}
