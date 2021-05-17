// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a SiLU module.
    /// </summary>
    public class SiLU : Module
    {
        internal SiLU (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_SiLU_forward (Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_SiLU_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (SiLU).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_SiLU_ctor (out IntPtr pBoxedModule);

        /// <summary>
        /// Sigmoid-Weighted Linear Unit
        /// </summary>
        /// <returns></returns>
        /// <remarks>The native libreary does not take an 'inplace' option, even though the PyTorch documentation mentions the parameter.</remarks>
        static public SiLU SiLU()
        {
            var handle = THSNN_SiLU_ctor (out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new SiLU (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// Sigmoid-Weighted Linear Unit
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <returns></returns>
        static public TorchTensor SiLU(TorchTensor x)
        {
            using (var m = Modules.SiLU()) {
                return m.forward (x);
            }
        }
    }

}
