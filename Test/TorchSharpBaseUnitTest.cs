using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TorchSharp;

namespace Test
{
    [TestClass]
    public class TorchSharpBaseUnitTest
    {
        [TestMethod]
        public void TestCreation0D()
        {
            var x = new FloatTensor();

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestDimesions0D()
        {
            var x = new FloatTensor();

            Assert.AreEqual(x.Dimensions, 1);
            Assert.AreEqual(x.Shape[0], 0);
        }
    }
}
