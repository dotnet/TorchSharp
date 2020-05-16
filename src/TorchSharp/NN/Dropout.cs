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
        private double _probability;
        private bool _isTraining;

        internal Dropout (IntPtr handle, bool isTraining, double probability = 0.5) : base (handle)
        {
            _probability = probability;
            _isTraining = isTraining;
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_dropoutModuleApply (IntPtr tensor, double probability, bool isTraining);

        public override TorchTensor Forward (TorchTensor tensor)
        {
            return new TorchTensor (THSNN_dropoutModuleApply (tensor.Handle, _probability, _isTraining));
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_dropoutModule ();

        static public Dropout Dropout (bool isTraining, double probability = 0.5)
        {
            var handle = THSNN_dropoutModule ();
            Torch.CheckForErrors ();
            return new Dropout (handle, isTraining, probability);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor Dropout (TorchTensor x, bool isTraining, double probability = 0.5)
        {
            using (var d = Modules.Dropout (isTraining, probability)) {
                return d.Forward (x);
            }
        }
    }

}
