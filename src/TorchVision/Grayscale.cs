// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Grayscale : ITransform
        {
            internal Grayscale(int outputChannels = 1)
            {
                if (outputChannels != 1 && outputChannels != 3) throw new ArgumentException("The number of output channels must be 1 or 3.");
                this.outputChannels = outputChannels;
            }

            public Tensor call(Tensor input)
            {
                int cDim = (int)input.Dimensions - 3;
                if (input.shape[cDim] == 1) return input.alias();
                if (input.shape[cDim] != 3)
                    throw new ArgumentException("RGB input to 'Grayscale' transform must have 3 channels.");
                return transforms.functional.rgb_to_grayscale(input, outputChannels);
            }

            protected int outputChannels;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Convert image to grayscale. 
            /// </summary>
            /// <param name="outputChannels">The number of channels in the output image tensor.</param>
            /// <returns></returns>
            static public ITransform Grayscale(int outputChannels = 1)
            {
                return new Grayscale(outputChannels);
            }
        }
    }
}
