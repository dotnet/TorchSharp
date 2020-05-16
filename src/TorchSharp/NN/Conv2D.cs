// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Conv2D : Module
    {
        internal Conv2D (IntPtr handle) : base (handle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Conv2d_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Conv2d_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Conv2d_ctor (long inputChannel, long outputChannel, long kernelSize, long stride, long padding);

        static public Conv2D Conv2D (long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0)
        {
            var res = THSNN_Conv2d_ctor (inputChannel, outputChannel, kernelSize, stride, padding);
            Torch.CheckForErrors ();
            return new Conv2D (res);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Conv2D (TorchTensor x, long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0)
        {
            using (var d = Modules.Conv2D (inputChannel, outputChannel, kernelSize, stride, padding)) {
                return d.Forward (x);
            }
        }
    }

}
