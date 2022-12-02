// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class VerticalFlip : ITransform
        {
            internal VerticalFlip()
            {
            }

            public Tensor call(Tensor input)
            {
                return input.flip(-2);
            }
        }

        public static partial class transforms
        {
            /// <summary>
            /// Flip the image vertically.
            /// </summary>
            /// <returns></returns>
            static public ITransform VerticalFlip()
            {
                return new VerticalFlip();
            }
        }
    }
}