using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TorchSharp.Test
{
    [TestClass]
    public class TorchSharp
    {
        [TestMethod]
        public void CreateFloatTensorOnes()
        {
            var ones = FloatTensor.Ones( new long[]{ 2, 2} );
            Assert.IsNotNull(ones);
        }

        [TestMethod]
        public void CreateFloatTensorOnesAccessData()
        {
            unsafe
            {
                var ones = FloatTensor.Ones(new long[] { 2, 2 });
                var data = new System.Span<float>(ones.Data.ToPointer(), 4);

                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1.0);
                }
            }
        }

        [TestMethod]
        public void CreateIntTensorOnesAccessData()
        {
            unsafe
            {
                var ones = IntTensor.Ones(new long[] { 2, 2 });
                var data = new System.Span<int>(ones.Data.ToPointer(), 4);

                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1);
                }
            }
        }

        [TestMethod]
        public void ScoreModel()
        {
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });
            Assert.IsNotNull(ones);

            var module = Module.LoadModule(@"E:\Source\Repos\libtorch\model.pt");
            Assert.IsNotNull(module);

            var result = module.Score(ones);
            Assert.IsNotNull(result);
        }
    }
}