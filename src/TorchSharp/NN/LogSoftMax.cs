// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
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

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_logSoftMaxApply(IntPtr tensor, long dimension);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_logSoftMaxApply(tensor.Handle, _dimension));
        }
    }
}
