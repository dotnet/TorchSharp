// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Conv2D : ProvidedModule
    {
        internal Conv2D(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_conv2DModuleApply(Module.HType module, IntPtr tensor);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_conv2DModuleApply(handle, tensor.Handle));
        }
    }
}
