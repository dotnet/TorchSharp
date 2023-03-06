// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Equalize : ITransform
        {
            internal Equalize()
            {
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.equalize(input);
            }
        }

        public static partial class transforms
        {
            /// <summary>
            /// Equalize the histogram of an image by applying a non-linear mapping to the input
            /// in order to create a uniform distribution of grayscale values in the output.
            /// </summary>
            /// <returns></returns>
            static public ITransform Equalize()
            {
                return new Equalize();
            }
        }
    }
}