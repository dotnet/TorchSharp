// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a MaxPool2D module.
    /// </summary>
    public class MaxPool2D : FunctionalModule<MaxPool2D>
    {
        private readonly long[] _kernelSize;
        private readonly long[] _stride;

        internal MaxPool2D(long[] kernelSize, long[] stride) : base()
        {
            _kernelSize = kernelSize;
            _stride = stride?? new long[0];
        }

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return tensor.MaxPool2D(_kernelSize, _stride);
        }
    }
}
