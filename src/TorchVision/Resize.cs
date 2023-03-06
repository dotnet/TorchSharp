// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Resize : ITransform
        {
            internal Resize(int height, int width, int? maxSize)
            {
                this.height = height;
                this.width = width;
                this.maxSize = maxSize;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.resize(input, height, width, maxSize);
            }

            private int height, width;
            private int? maxSize;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="height">Desired output height</param>
            /// <param name="width">Desired output width</param>
            /// <returns></returns>
            static public ITransform Resize(int height, int width)
            {
                return new Resize(height, width, null);
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="size">Desired output size</param>
            /// <param name="maxSize">Max size</param>
            static public ITransform Resize(int size, int? maxSize = null)
            {
                return new Resize(size, -1, maxSize);
            }
        }
    }
}