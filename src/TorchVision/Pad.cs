// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Pad : ITransform
        {
            internal Pad(long[] pad, PaddingModes mode = PaddingModes.Constant, double value = 0)
            {
                this.pad = pad;
                this.mode = mode;
                this.value = value;
            }

            public Tensor call(Tensor input)
            {
                return TorchSharp.torch.nn.functional.pad(input, pad, mode, value);
            }

            private long[] pad;
            private PaddingModes mode;
            private double value;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Pad the given image on all sides with the given “pad” value.
            /// </summary>
            /// <param name="padding">
            /// Padding on each border. If a single int is provided this is used to pad all borders.
            /// If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
            /// If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
            /// </param>
            /// <param name="fill">Pixel fill value for constant fill.</param>
            /// <param name="mode">Type of padding.</param>
            static public ITransform Pad(long[] padding, PaddingModes mode = PaddingModes.Constant, double fill = 0)
            {
                return new Pad(padding, mode, fill);
            }
        }
    }
}