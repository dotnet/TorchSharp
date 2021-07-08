// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustSaturation : Adjustment, ITransform
    {
        internal AdjustSaturation(double saturation_factor)
        {
            if (saturation_factor < 0.0)
                throw new ArgumentException($"The saturation factor ({saturation_factor}) must be non-negative.");
            this.saturation_factor = saturation_factor;
        }

        public Tensor forward(Tensor img)
        {
            if (saturation_factor == 1.0)
                // Special case -- no change.
                return img;

            return Blend(img, transforms.Grayscale().forward(img), saturation_factor);
        }

        private double saturation_factor;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Adjust color saturation of an image.
        /// </summary>
        /// <param name="saturation_factor">
        /// How much to adjust the saturation. 0 will give a black and white image, 1 will give the original image
        /// while 2 will enhance the saturation by a factor of 2.
        /// </param>
        /// <returns></returns>
        static public ITransform AdjustSaturation(double saturation_factor)
        {
            return new AdjustSaturation(saturation_factor);
        }
    }
}
