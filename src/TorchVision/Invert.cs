// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Invert : ITransform
        {
            internal Invert()
            {
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.invert(input);
            }
        }

        public static partial class transforms
        {
            /// <summary>
            /// Invert the image colors.
            /// </summary>
            /// <remarks>The code assumes that integer color values lie in the range [0,255], and floating point colors in [0,1[.</remarks>
            static public ITransform Invert()
            {
                return new Invert();
            }
        }
    }
}
