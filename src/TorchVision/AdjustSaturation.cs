// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class AdjustSaturation : ITransform
        {
            internal AdjustSaturation(double saturation_factor)
            {
                if (saturation_factor < 0.0)
                    throw new ArgumentException($"The saturation factor ({saturation_factor}) must be non-negative.");
                this.saturation_factor = saturation_factor;
            }

            public Tensor call(Tensor img)
            {
                return transforms.functional.adjust_saturation(img, saturation_factor);
            }

            private double saturation_factor;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Adjust the color saturation of an image.
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
}