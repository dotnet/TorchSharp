// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class Equalize : ITransform
    {
        internal Equalize()
        {
        }

        public Tensor forward(Tensor input)
        {
            return transforms.functional.equalize(input);
        }
    }

    public static partial class transforms
    {
        /// <summary>
        /// Equalize the histogram of an image by applying a non-linear mapping to the input
        /// in order to create a uniform distribution of grayscale values in the output.
        /// </summary>
        /// <returns></returns>
        static public ITransform Equalize()
        {
            return new Equalize();
        }
    }
}
