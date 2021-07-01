// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a Softmin module.
    /// </summary>
    public class Softmin : Module
    {
        internal Softmin (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Softmin_forward (Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Softmin_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (Softmin).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Softmin_ctor (long dim, out IntPtr pBoxedModule);

        /// <summary>
        /// Softmin
        /// </summary>
        /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
        /// <returns></returns>
        static public Softmin Softmin(long dim)
        {
            var handle = THSNN_Softmin_ctor(dim, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Softmin (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// Softmin
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
        /// <returns></returns>
        static public TorchTensor Softmin (TorchTensor x, long dim)
        {
            using (var m = Modules.Softmin(dim)) {
                return m.forward (x);
            }
        }
    }

}
