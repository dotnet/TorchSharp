// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class HorizontalFlip : ITransform
        {
            internal HorizontalFlip()
            {
            }

            public Tensor call(Tensor input)
            {
                return input.flip(-1);
            }
        }

        public static partial class transforms
        {
            /// <summary>
            /// Flip the image horizontally.
            /// </summary>
            /// <returns></returns>
            static public ITransform HorizontalFlip()
            {
                return new HorizontalFlip();
            }
        }
    }
}