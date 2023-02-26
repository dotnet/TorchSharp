// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class CenterCrop : ITransform
        {
            internal CenterCrop(int height, int width)
            {
                this.height = height;
                this.width = width;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.center_crop(input, height, width);
            }

            protected int height, width;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Crop the center of the image.
            /// </summary>
            /// <param name="height">Desired output height of the crop</param>
            /// <param name="width">Desired output width of the crop</param>
            static public ITransform CenterCrop(int height, int width)
            {
                return new CenterCrop(height, width);
            }

            /// <summary>
            /// Crop the center of the image.
            /// </summary>
            /// <param name="size">Desired output size of the crop</param>
            static public ITransform CenterCrop(int size)
            {
                return new CenterCrop(size, size);
            }
        }
    }
}