// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.TensorExtensionMethods;

namespace TorchSharp.torchvision
{
    internal class AdjustSharpness : AffineGridBase, ITransform
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

        private Tensor BlurredDegenerateImage(Tensor input)
        {
            var device = input.device;
            var dtype = input.IsIntegral() ? ScalarType.Float32 : input.dtype;
            var kernel = Float32Tensor.ones(3, 3, device: device);
            kernel[1, 1] = Float32Tensor.from(5.0f);
            kernel /= kernel.sum();
            kernel = kernel.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

            var result_tmp = SqueezeIn(input, new ScalarType[] { ScalarType.Float32, ScalarType.Float64}, out var needCast, out var needSqueeze, out var out_dtype);
            result_tmp = torch.nn.functional.conv2d(result_tmp,kernel, groups: result_tmp.shape[result_tmp.shape.Length - 3]);
            result_tmp = SqueezeOut(result_tmp, needCast, needSqueeze, out_dtype);

            var result = input.clone();
            result.index_put_(result_tmp, TensorIndex.Ellipsis, TensorIndex.Slice(1,-1), TensorIndex.Slice(1, -1));
            return result;
        }

        private double sharpness;

        private Tensor Blend(Tensor img1, Tensor img2, double ratio)
        {
            var bound = img1.IsIntegral() ? 255.0 : 1.0;
            return (img1 * ratio + img2 * (1.0 - ratio)).clamp(0, bound).to(img2.dtype);
        }
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
