using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a log softmax module.
    /// </summary>
    public class LogSoftMax : FunctionalModule<LogSoftMax>
    {
        private long _dimension;

        internal LogSoftMax(long dimension) : base()
        {
            _dimension = dimension;
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_logSoftMaxApply(IntPtr tensor, long dimension);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(THSNN_logSoftMaxApply(tensor.Handle, _dimension));
        }
    }
}
