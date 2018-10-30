using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TorchSharp.Tests
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

        [TestMethod]
        public void TestCreation1D()
        {
            var x = new FloatTensor(10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation2D()
        {
            var x = new FloatTensor(10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation3D()
        {
            var x = new FloatTensor(10, 10, 10);

            Assert.AreNotEqual(x, null);
        }

        [TestMethod]
        public void TestCreation4D()
        {
            var x = new FloatTensor(10, 10, 10, 3);

            Assert.AreNotEqual(x, null);
        }
    }
}
