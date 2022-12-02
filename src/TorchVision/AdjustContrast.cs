// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class AdjustContrast : ITransform
        {
            internal AdjustContrast(double contrast_factor)
            {
                if (contrast_factor < 0.0)
                    throw new ArgumentException($"The sharpness factor ({contrast_factor}) must be non-negative.");
                this.contrast_factor = contrast_factor;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.adjust_contrast(input, contrast_factor);
            }

            private double contrast_factor;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Adjust the contrast of the image. 
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
}
