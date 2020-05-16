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
        internal Dropout (IntPtr handle) : base (handle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Dropout_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Dropout_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Dropout_ctor (double probability);

        static public Dropout Dropout (double probability = 0.5)
        {
            var handle = THSNN_Dropout_ctor (probability);
            Torch.CheckForErrors ();
            return new Dropout (handle);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor Dropout (TorchTensor x, double probability = 0.5)
        {
            using (var d = Modules.Dropout (probability)) {
                return d.Forward (x);
            }
        }
    }

}
