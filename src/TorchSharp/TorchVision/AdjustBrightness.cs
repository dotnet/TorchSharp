// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustBrightness : ITransform
    {
        internal AdjustBrightness(double brightness_factor)
        {
            if (brightness_factor < 0.0)
                throw new ArgumentException($"The sharpness factor ({brightness_factor}) must be non-negative.");
            this.brightness_factor = brightness_factor;
        }

        public Tensor forward(Tensor input)
        {
            return transforms.functional.adjust_brightness(input, brightness_factor);
        }

        private double brightness_factor;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Adjust the sharpness of the image. 
        /// </summary>
        /// <param name="brightness_factor">
        /// How much to adjust the brightness. Can be any non negative number.
        /// 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
        /// </param>
        /// <returns></returns>
        static public ITransform AdjustBrightness(double brightness_factor)
        {
            return new AdjustBrightness(brightness_factor);
        }
    }
}
