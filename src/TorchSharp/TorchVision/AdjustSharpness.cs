// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

using static TorchSharp.Tensor.TensorExtensionMethods;

namespace TorchSharp.TorchVision
{
    internal class AdjustSharpness : ITransform
    {
        internal AdjustSharpness(double sharpness)
        {
            if (sharpness < 0.0)
                throw new ArgumentException($"The sharpness factor ({sharpness}) must be non-negative.");
            this.sharpness = sharpness;
        }

        public TorchTensor forward(TorchTensor input)
        {
            if (input.shape[input.shape.Length - 1] <= 2 || input.shape[input.shape.Length - 2] <= 2)
                return input;

            return Blend(input, BlurredDegenerateImage(input), sharpness);
        }

        private TorchTensor BlurredDegenerateImage(TorchTensor input)
        {
            var device = new Device(input.device);
            var dtype = input.IsIntegral() ? ScalarType.Float32 : input.Type;
            var kernel = Float32Tensor.ones(3, 3, device: device);
            kernel[1, 1] = Float32Tensor.from(5.0f);
            kernel /= kernel.sum();
            kernel = kernel.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

            var result_tmp = SqueezeIn(input, out var needCast, out var needSqueeze, out var out_dtype);
            result_tmp = result_tmp.conv2d(kernel, groups: result_tmp.shape[result_tmp.shape.Length - 3]);
            result_tmp = SqueezeOut(result_tmp, needCast, needSqueeze, out_dtype);

            var result = input.clone();
            result.index_put_(result_tmp, TorchTensorIndex.Ellipsis, TorchTensorIndex.Slice(1,-1), TorchTensorIndex.Slice(1, -1));
            return result;
        }

        protected double sharpness;


        private TorchTensor Blend(TorchTensor img1, TorchTensor img2, double ratio)
        {
            var bound = img1.IsIntegral() ? 255.0 : 1.0;
            return (img1 * ratio + img2 * (1.0 - ratio)).clamp(0, bound).to(img2.Type);
        }

        private TorchTensor SqueezeIn(TorchTensor img, out bool needCast, out bool needSqueeze, out ScalarType dtype)
        {
            needSqueeze = false;

            if (img.Dimensions < 4) {
                img = img.unsqueeze(0);
                needSqueeze = true;
            }

            dtype = img.Type;
            needCast = false;

            if (dtype != ScalarType.Float32 && dtype != ScalarType.Float64) {
                needCast = true;
                img = img.to_type(ScalarType.Float32);
            }

            return img;
        }

        private TorchTensor SqueezeOut(TorchTensor img, bool needCast, bool needSqueeze, ScalarType dtype)
        {
            if (needSqueeze) {
                img = img.squeeze(0);
            }

            if (needCast) {
                if (TensorExtensionMethods.IsIntegral(dtype))
                    img = img.round();

                img = img.to_type(dtype);
            }

            return img;
        }
    }

    public static partial class Transforms
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
            return new AdjustSharpness(sharpness);
        }
    }
}
