// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustHue : Adjustment, ITransform
    {
        internal AdjustHue(double hue_factor)
        {
            hue_factor %= 1.0;

s            this.hue_factor = hue_factor;
        }

        public Tensor forward(Tensor img)
        {
            if (hue_factor == 0.0)
                // Special case -- no change.
                return img;

            if (img.shape.Length < 4 || img.shape[img.shape.Length - 3] == 1)
                return img;

            var orig_dtype = img.dtype;
            if (!torch.is_floating_point(img))
                img = img.to_type(torch.float32) / 255.0;

            var HSV = RGBtoHSV(img);

            HSV.h = (HSV.h + hue_factor) % 1.0;

            var img_hue_adj = HSVtoRGB(HSV.h, HSV.s, HSV.v);

            if (orig_dtype.IsIntegral())
                img_hue_adj = (img_hue_adj * 255.0).to_type(orig_dtype);

            //
            // Something really strange happens in the process -- the image comes out as 'NxCxHxW', but the
            // underlying memory is formatted as if it's 'NxHxWxC'.
            // So, as a workaround, we need to reshape it and permute.
            //
            long[] NHWC = new long[] { img.shape[0], img.shape[2], img.shape[3], img.shape[1] };
            long[] permutation = new long[] { 0, 3, 1, 2 };

            return img_hue_adj.reshape(NHWC).permute(permutation);
        }

        private double hue_factor;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Adjust hue of an image.
        /// The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel(H).
        /// The image is then converted back to original image mode.
        /// </summary>
        /// <param name="hue_factor">
        /// How much to shift the hue channel. 0 means no shift in hue.
        /// Hue is often defined in degrees, with 360 being a full turn on the color wheel.
        /// In this library, 1.0 is a full turn, which means that 0.5 and -0.5 give complete reversal of
        /// the hue channel in HSV space in positive and negative direction respectively.
        /// </param>
        /// <returns></returns>
        /// <remarks>
        /// Unlike Pytorch, TorchSharp will allow the hue_factor to lie outside the range [-0.5,0.5].
        /// A factor of 0.75 has the same image effect as -.25
        /// </remarks>
        static public ITransform AdjustHue(double hue_factor)
        {
            return new AdjustHue(hue_factor);
        }
    }
}
