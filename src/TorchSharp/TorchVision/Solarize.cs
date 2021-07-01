// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;


namespace TorchSharp.torchvision
{
    internal class Solarize : ITransform
    {
        internal Solarize(double threshold)
        {
            this.threshold = threshold;
        }

        public TorchTensor forward(TorchTensor input)
        {
            using (var inverted = transforms.Invert().forward(input))
                return input.where(input < threshold, inverted);
        }

        protected double threshold;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Solarize the image by inverting all pixel values above a threshold.
        /// </summary>
        /// <param name="threshold">All pixels equal or above this value are inverted.</param>
        static public ITransform Solarize(double threshold)
        {
            return new Solarize(threshold);
        }
    }
}
