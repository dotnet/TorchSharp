// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustContrast : Adjustment, ITransform
    {
        internal AdjustContrast(double contrast_factor)
        {
            if (contrast_factor < 0.0)
                throw new ArgumentException($"The sharpness factor ({contrast_factor}) must be non-negative.");
            this.contrast_factor = contrast_factor;
        }

        public Tensor forward(Tensor input)
        {
            if (contrast_factor == 1.0)
                // Special case -- no change.
                return input;

            var dtype = torch.is_floating_point(input) ? input.dtype : torch.float32;
            var mean = torch.mean(transforms.Grayscale().forward(input).to_type(dtype), new long[] { -3, -2, -1 }, keepDimension: true);
            return Blend(input, mean, contrast_factor);
        }

        private double contrast_factor;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Adjust the sharpness of the image. 
        /// </summary>
        /// <param name="contrast_factor">
        /// How much to adjust the contrast. Can be any non-negative number.
        /// 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
        /// </param>
        /// <returns></returns>
        static public ITransform AdjustContrast(double contrast_factor)
        {
            return new AdjustContrast(contrast_factor);
        }
    }
}
