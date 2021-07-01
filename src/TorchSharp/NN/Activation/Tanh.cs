// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a Tanh module.
    /// </summary>
    public class Tanh : torch.nn.Module
    {
        internal Tanh (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Tanh_forward (torch.nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Tanh_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (Tanh).Name;
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Tanh_ctor (out IntPtr pBoxedModule);

        /// <summary>
        /// Tanh activation
        /// </summary>
        /// <returns></returns>
        static public Tanh Tanh()
        {
            var handle = THSNN_Tanh_ctor(out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tanh (handle, boxedHandle);
        }
    }
    public static partial class functional
    {
        /// <summary>
        /// Tanh activation
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <returns></returns>
        static public TorchTensor Tanh (TorchTensor x)
        {
            using (var m =nn.Tanh()) {
                return m.forward (x);
            }
        }
    }

}
