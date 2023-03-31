// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Posterize : ITransform
        {
            internal Posterize(int bits)
            {
                this.bits = bits;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.posterize(input, bits);
            }

            protected int bits;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Posterize the image by reducing the number of bits for each color channel. 
            /// </summary>
            /// <param name="bits">Number of high-order bits to keep for each channel.</param>
            /// <returns></returns>
            /// <remarks>The tensor must be an integer tensor.</remarks>
            static public ITransform Posterize(int bits)
            {
                return new Posterize(bits);
            }
        }
    }
}