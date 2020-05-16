// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a log softmax module.
    /// </summary>
    public class LogSoftMax : Module
    {
        internal LogSoftMax (IntPtr handle, long dimension) : base (handle)
        {
            _dimension = dimension;
        }

        private long _dimension;

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_logSoftMaxApply (IntPtr tensor, long dimension);

        public override TorchTensor Forward (TorchTensor tensor)
        {
            return new TorchTensor (THSNN_logSoftMaxApply (tensor.Handle, _dimension));
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_logSoftMaxModule ();

        static public LogSoftMax LogSoftMax (long dimension)
        {
            var handle = THSNN_logSoftMaxModule ();
            Torch.CheckForErrors ();
            return new LogSoftMax (handle, dimension);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor LogSoftMax (TorchTensor x, long dimension)
        {
            using (var l = Modules.LogSoftMax (dimension)) {
                return l.Forward (x);
            }
        }
    }

}
