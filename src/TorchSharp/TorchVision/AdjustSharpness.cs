// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustSharpness : ITransform
    {
        internal AdjustSharpness(double sharpness)
        {
            if (sharpness < 0.0)
                throw new ArgumentException($"The sharpness factor ({sharpness}) must be non-negative.");
            this.sharpness = sharpness;
        }

        public Tensor forward(Tensor input)
        {
            return transforms.functional.adjust_sharpness(input, sharpness);
        }

        private double sharpness;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Adjust the sharpness of the image. 
        /// </summary>
        /// <param name="sharpness">
        /// How much to adjust the sharpness. Can be any non negative number.
        /// 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
        /// </param>
        /// <returns></returns>
        static public ITransform AdjustSharpness(double sharpness)
        {
            if (sharpness < 0.0)
                throw new ArgumentException("Negative sharpness factor");
            return new AdjustSharpness(sharpness);
        }
    }
}
