// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.torchvision
{
    internal class Pad : ITransform
    {
        internal Pad(long[] pad, PaddingModes mode = PaddingModes.Constant, double value = 0)
        {
            this.pad = pad;
            this.mode = mode;
            this.value = value;
        }

        public Tensor forward(Tensor input)
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
        /// <param name="mode"></param>
        static public ITransform Pad(long[] padding, PaddingModes mode = PaddingModes.Constant, double fill = 0)
        {
            return new Pad(padding, mode, fill);
        }
    }
}
