// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module.
    /// </summary>
    public class Dropout : Module
    {
        internal Dropout (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Dropout_forward (Module.HType module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Dropout_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Dropout_ctor (double probability, bool inPlace, out IntPtr pBoxedModule);

        static public Dropout Dropout (double probability = 0.5, bool inPlace = false)
        {
            var handle = THSNN_Dropout_ctor (probability, inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Dropout(handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor Dropout (TorchTensor x, double probability = 0.5, bool inPlace = false)
        {
            using (var d = Modules.Dropout (probability)) {
                return d.forward (x);
            }
        }
    }

}
