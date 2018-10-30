using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TorchSharp;

namespace Test
{
    [TestClass]
    public class FloatTorchTensorUnitTest
    {
        [TestMethod]
        public void TestCreation()
        {
            var x = FloatTorchTensor.Create(10);
            
             Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestFill()
        {
            var x = FloatTorchTensor.Create(10);
            x.Fill(30);

            for (int i = 0; i < x.Dimensions[0]; i++)
            {
                Assert.AreEqual(x[i], 30);
            }
        }

        [TestMethod]
        public void TestFillBySet()
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
    }
}
