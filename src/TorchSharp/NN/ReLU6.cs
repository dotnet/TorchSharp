// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLU6 module.
    /// </summary>
    public class ReLU6 : Module
    {
        internal ReLU6 (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ReLU6_forward (Module.HType module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_ReLU6_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (ReLU6).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_ReLU6_ctor (bool inplace, out IntPtr pBoxedModule);

        /// <summary>
        /// Rectified Linear Unit
        ///
        /// This ReLU version caps positive values at 6.
        /// </summary>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public ReLU6 ReLU6(bool inPlace = false)
        {
            var handle = THSNN_ReLU6_ctor (inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReLU6 (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// Rectified Linear Unit
        ///
        /// This ReLU version caps positive values at 6.
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public TorchTensor ReLU6(TorchTensor x, bool inPlace = false)
        {
            using (var m = Modules.ReLU6(inPlace)) {
                return m.forward (x);
            }
        }
    }

}
