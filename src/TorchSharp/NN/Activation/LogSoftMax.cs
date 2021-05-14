// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a log softmax module.
    /// </summary>
    public class LogSoftmax : Module
    {
        internal LogSoftmax (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_LogSoftmax_forward(Module.HType handle, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_LogSoftmax_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_LogSoftmax_ctor (long dimension, out IntPtr pBoxedModule);

        static public LogSoftmax LogSoftmax (long dimension)
        {
            var handle = THSNN_LogSoftmax_ctor(dimension, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new LogSoftmax (handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor LogSoftmax (TorchTensor x, long dimension)
        {
            using (var l = Modules.LogSoftmax (dimension)) {
                return l.forward (x);
            }
        }
    }

}
