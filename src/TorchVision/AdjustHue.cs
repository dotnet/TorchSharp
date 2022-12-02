// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class AdjustHue : ITransform
        {
            internal AdjustHue(double hue_factor)
            {
                hue_factor %= 1.0;

                this.hue_factor = hue_factor;
            }

            public Tensor call(Tensor img)
            {
                return transforms.functional.adjust_hue(img, hue_factor);
            }

            private double hue_factor;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Adjust hue of an image.
            /// The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel(H).
            /// The image is then converted back to original image mode.
            /// </summary>
            /// <param name="hue_factor">
            /// How much to shift the hue channel. 0 means no shift in hue.
            /// Hue is often defined in degrees, with 360 being a full turn on the color wheel.
            /// In this library, 1.0 is a full turn, which means that 0.5 and -0.5 give complete reversal of
            /// the hue channel in HSV space in positive and negative direction respectively.
            /// </param>
            /// <returns></returns>
            /// <remarks>
            /// Unlike Pytorch, TorchSharp will allow the hue_factor to lie outside the range [-0.5,0.5].
            /// A factor of 0.75 has the same effect as -.25
            /// Note that adjusting the hue is a very expensive operation, and may therefore not be suitable as a method
            /// for data augmentation when training speed is important.
            /// </remarks>
            static public ITransform AdjustHue(double hue_factor)
            {
                return new AdjustHue(hue_factor);
            }
        }
    }
}
