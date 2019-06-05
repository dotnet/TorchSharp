using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public static class Init
    {
        [DllImport("LibTorchSharp")]
        private static extern void THSNN_initUniform(IntPtr src, double low, double high);

        public static void Uniform(TorchTensor tensor, double low = 0, double high = 1)
        {
            THSNN_initUniform(tensor.Handle, low, high);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_initKaimingUniform(IntPtr src, double a);

        public static void KaimingUniform(TorchTensor tensor, double a = 0)
        {
            THSNN_initKaimingUniform(tensor.Handle, a);
        }

        public static (long fanIn, long fanOut) CalculateFanInAndFanOut<T>(TorchTensor tensor)
        {
            var dimensions = tensor.Dimensions;

            if (dimensions < 2)
            {
                throw new ArgumentException("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
            }

            var shape = tensor.Shape;
            // Linear
            if (dimensions == 2)
            {
                return (shape[1], shape[0]);
            }
            else
            {
                var numInputFMaps = tensor.Shape[1];
                var numOutputFMaps = tensor.Shape[0];
                var receptiveFieldSize = tensor[0, 0].NumberOfElements;

                return (numInputFMaps * receptiveFieldSize, numOutputFMaps * receptiveFieldSize);
            }
        }
    }
}
