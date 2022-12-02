// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class ResizedCrop : ITransform
        {
            internal ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth)
            {
                cropper = transforms.Crop(top, left, height, width);
                resizer = transforms.Resize(newHeight, newWidth);
            }

            public Tensor call(Tensor input)
            {
                using var cr = cropper.call(input);
                return resizer.call(cr);
            }

            private ITransform cropper;
            private ITransform resizer;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Crop the given image and resize it to desired size.
            /// </summary>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="height">Height of the crop box.</param>
            /// <param name="width">Width of the crop box.</param>
            /// <param name="newHeight">New height.</param>
            /// <param name="newWidth">New width.</param>
            static public ITransform ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth)
            {
                return new ResizedCrop(top, left, height, width, newHeight, newWidth);
            }

            /// <summary>
            /// Crop the given image and resize it to desired (square) size.
            /// </summary>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="size">Height and width of the crop box.</param>
            /// <param name="newSize">New height and width.</param>
            static public ITransform ResizedCrop(int top, int left, int size, int newSize)
            {
                return new ResizedCrop(top, left, size, size, newSize, -1);
            }
        }
    }
}