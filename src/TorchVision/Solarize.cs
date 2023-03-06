// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Solarize : ITransform
        {
            internal Solarize(double threshold)
            {
                this.threshold = threshold;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.solarize(input, threshold);
            }

            private double threshold;
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
}