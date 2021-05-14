// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module for 2d/3d convolutational layers.
    /// </summary>
    public class Flatten : Module
    {
        internal Flatten (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Flatten_forward (Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Flatten_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Flatten_ctor (long startDim, long endDim, out IntPtr pBoxedModule);

        static public Flatten Flatten (long startDim = 1, long endDim = -1)
        {
            var handle = THSNN_Flatten_ctor (startDim, endDim, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Flatten (handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor Flatten (TorchTensor x, long startDim = 1, long endDim = -1)
        {
            using (var f = Modules.Flatten (startDim, endDim)) {
                return f.forward (x);
            }
        }
    }
}
