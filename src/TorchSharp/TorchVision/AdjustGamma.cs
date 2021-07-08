// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustGamma : ITransform
    {
        internal AdjustGamma(double gamma, double gain = 1.0)
        {
            if (gamma < 0.0)
                throw new ArgumentException($"The saturation factor ({gamma}) must be non-negative.");
            this.gamma = gamma;
            this.gain = gain;
        }

        public Tensor forward(Tensor img)
        {
            var dtype = img.dtype;
            if (!torch.is_floating_point(img))
                img = transforms.ConvertImageDType(torch.float32).forward(img);

            img = (gain * img.pow(gamma)).clamp(0, 1);

            return transforms.ConvertImageDType(dtype).forward(img); ;
        }

        private double gamma;
        private double gain;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Perform gamma correction on an image.
        /// 
        /// See: https://en.wikipedia.org/wiki/Gamma_correction
        /// </summary>
        /// <param name="gamma">
        /// Non negative real number.
        /// gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
        /// </param>
        /// <param name="gain">The constant multiplier in the gamma correction equation.</param>
        /// <returns></returns>
        static public ITransform AdjustGamma(double gamma, double gain = 1.0)
        {
            return new AdjustGamma(gamma);
        }
    }
}
