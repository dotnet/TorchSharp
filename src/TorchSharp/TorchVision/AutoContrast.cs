// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

using static TorchSharp.Tensor.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AutoContrast : ITransform
    {
        internal AutoContrast()
        {
        }

        public TorchTensor forward(TorchTensor input)
        {
            var bound = input.IsIntegral() ? 255.0f : 1.0f;
            var dtype = input.IsIntegral() ? ScalarType.Float32 : input.Type;

            var minimum = input.amin(new long[] { -2, -1 }, keepDim: true).to(dtype);
            var maximum = input.amax(new long[] { -2, -1 }, keepDim: true).to(dtype);

            var eq_idxs = (minimum == maximum).nonzero_as_list()[0];
            minimum.index_put_(0, eq_idxs);
            maximum.index_put_(bound, eq_idxs);

            var scale = Float32Tensor.from(bound) / (maximum - minimum);

            return ((input - minimum) * scale).clamp(0, bound).to(input.Type);
        }
    }

    public static partial class transforms
    {
        /// <summary>
        /// Autocontrast the pixels of the given image
        /// </summary>
        /// <returns></returns>
        static public ITransform AutoContrast()
        {
            return new AutoContrast();
        }
    }
}
