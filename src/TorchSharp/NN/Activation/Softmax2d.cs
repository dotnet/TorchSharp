// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a Softmax2d module.
    /// </summary>
    public class Softmax2d : torch.nn.Module
    {
        internal Softmax2d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Softmax2d_forward (torch.nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Softmax2d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (Softmax2d).Name;
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Softmax2d_ctor (out IntPtr pBoxedModule);

        /// <summary>
        /// Applies Softmax over features to each spatial location
        /// </summary>
        /// <returns></returns>
        static public Softmax2d Softmax2d()
        {
            var handle = THSNN_Softmax2d_ctor(out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Softmax2d (handle, boxedHandle);
        }
    }
    public static partial class functional
    {
        /// <summary>
        /// Applies Softmax over features to each spatial location
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <returns></returns>
        static public TorchTensor Softmax2d (TorchTensor x)
        {
            using (var m =nn.Softmax2d()) {
                return m.forward (x);
            }
        }
    }

}
