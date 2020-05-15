// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLU module.
    /// </summary>
    public class ReLU : FunctionalModule<ReLU>
    {
        private readonly bool _inPlace;

        internal ReLU(bool inPlace = false) : base()
        {
            _inPlace = inPlace;
        }

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return _inPlace ? tensor.ReluInPlace() : tensor.Relu(); 
        }

        public override string GetName()
        {
            return typeof(ReLU).Name;
        }
    }
}
