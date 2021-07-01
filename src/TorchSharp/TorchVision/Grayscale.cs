// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;


namespace TorchSharp.TorchVision
{
    internal class Grayscale : ITransform
    {
        internal Grayscale(int outputChannels = 1)
        {
            if (outputChannels != 1 && outputChannels != 3) throw new ArgumentException("The number of output channels must be 1 or 3.");
            this.outputChannels = outputChannels;
        }

        public TorchTensor forward(TorchTensor input)
        {
            int cDim = (int)input.Dimensions - 3;
            var rgb = input.unbind(cDim);
            var img = (rgb[0] * 0.2989 + rgb[1] * 0.587 + rgb[2] * 0.114).unsqueeze(cDim);
            return outputChannels == 3 ? img.expand(input.shape) : img;
        }

        protected int outputChannels;
    }

    public static partial class Transforms
    {
        static public ITransform Grayscale(int outputChannels = 1)
        {
            return new Grayscale(outputChannels);
        }
    }
}
