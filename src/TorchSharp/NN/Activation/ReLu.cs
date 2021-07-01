// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLU module.
    /// </summary>
    public class ReLU : Module
    {
        internal ReLU (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ReLU_forward (Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_ReLU_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (ReLU).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_ReLU_ctor (bool inplace, out IntPtr pBoxedModule);

        /// <summary>
        /// Rectified Linear Unit
        /// </summary>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public ReLU ReLU(bool inPlace = false)
        {
            var handle = THSNN_ReLU_ctor (inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new ReLU (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// Rectified Linear Unit
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public TorchTensor ReLU(TorchTensor x, bool inPlace = false)
        {
            using (var m = Modules.ReLU(inPlace)) {
                return m.forward (x);
            }
        }
    }

}
