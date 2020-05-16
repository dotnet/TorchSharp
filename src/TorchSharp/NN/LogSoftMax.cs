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
        internal LogSoftMax (IntPtr handle) : base (handle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_LogSoftMax_forward (Module.HType handle, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_LogSoftMax_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_LogSoftMax_ctor (long dimension);

        static public LogSoftMax LogSoftMax (long dimension)
        {
            var handle = THSNN_LogSoftMax_ctor (dimension);
            Torch.CheckForErrors ();
            return new LogSoftMax (handle);
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
