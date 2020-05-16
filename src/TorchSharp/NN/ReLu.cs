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
        internal ReLU (IntPtr handle) : base (handle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ReLU_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_ReLU_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
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
        extern static IntPtr THSNN_ReLU_ctor (bool inplace);

        static public ReLU Relu (bool inPlace = false)
        {
            var handle = THSNN_ReLU_ctor (inPlace);
            Torch.CheckForErrors ();
            return new ReLU (handle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Relu (TorchTensor x, bool inPlace = false)
        {
            using (var m = Modules.Relu (inPlace)) {
                return m.Forward (x);
            }
        }
    }

}
