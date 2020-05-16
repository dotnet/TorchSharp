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
        internal ReLU (IntPtr handle, bool inPlace = false) : base (handle)
        {
            _inPlace = inPlace;
        }
        private readonly bool _inPlace;

        public override TorchTensor Forward (TorchTensor tensor)
        {
            return _inPlace ? tensor.ReluInPlace () : tensor.Relu ();
        }

        public override string GetName ()
        {
            return typeof (ReLU).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_reluModule ();

        static public ReLU Relu (bool inPlace = false)
        {
            var handle = THSNN_reluModule ();
            Torch.CheckForErrors ();
            return new ReLU (handle, inPlace);
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
