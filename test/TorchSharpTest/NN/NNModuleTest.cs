using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.JIT;
using TorchSharp.NN;
using TorchSharp.Tensor;
using Xunit;

namespace TorchSharpTest.NN
{
    public class NNModuleTest
    {
        [Fact]
        public void AvgPool2D_Object_Initialized()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2, 2, 2 });
            var obj = TorchSharp.NN.Module.AvgPool2D(ones, new long[] { 2 }, new long[] { 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
        }

        [Fact]
        public void MaxPool2D_Object_Initialized()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2, 2, 2 });
            var obj = TorchSharp.NN.Module.MaxPool2D(ones, new long[] { 2 }, new long[] { 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
        }

    }
}
