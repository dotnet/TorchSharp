// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


namespace TorchSharp.torchvision
{
    internal class Grayscale : ITransform
    {
        internal Grayscale(int outputChannels = 1)
        {
            if (outputChannels != 1 && outputChannels != 3) throw new ArgumentException("The number of output channels must be 1 or 3.");
            this.outputChannels = outputChannels;
        }

        public Tensor forward(Tensor input)
        {
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
