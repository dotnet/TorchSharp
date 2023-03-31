// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Crop : ITransform
        {
            internal Crop(int top, int left, int height, int width)
            {
                this.top = top;
                this.left = left;
                this.height = height;
                this.width = width;
            }

            public Tensor call(Tensor input)
            {
                return input.crop(top, left, height, width);
            }

            private int top, left, height, width;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Crop an image at specified location and output size. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge,
            /// image is padded with 0 and then cropped.
            /// </summary>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="height">The height of the crop box.</param>
            /// <param name="width">The width of the crop box.</param>
            /// <returns></returns>
            static public ITransform Crop(int top, int left, int height, int width)
            {
                return new Crop(top, left, height, width);
            }
        }
    }
}