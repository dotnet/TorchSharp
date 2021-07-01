// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;


namespace TorchSharp.torchvision
{
    internal class GaussianBlur : ITransform
    {
        internal GaussianBlur(IList<long> kernelSize, float min, float max)
        {
            this.sigma = (min == max) ?
                min :
                (float)(new Random().NextDouble() * (max - min) + min);
            this.kernelSize = kernelSize.ToArray();
        }

        public TorchTensor forward(TorchTensor input)
        {
            var dtype = TensorExtensionMethods.IsIntegral(input.Type) ? ScalarType.Float32 : input.Type;

            var kernel = GetGaussianKernel2d(dtype, input.device);
            kernel = kernel.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

            var img = SqueezeIn(input, out var needCast, out var needSqueeze, out var out_dtype);

            // The padding needs to be adjusted to make sure that the output is the same size as the input.

            var k0d2 = kernelSize[0] / 2;
            var k1d2 = kernelSize[1] / 2;
            var k0sm1 = kernelSize[0] - 1;
            var k1sm1 = kernelSize[1] - 1;

            var padding = new long[] { k0d2, k0sm1 - k0d2, k1d2, k1sm1 - k1d2 };

            img = TorchSharp.torch.nn.functional.Pad(img, padding, PaddingModes.Reflect);
            img = img.conv2d(kernel, groups: img.shape[img.shape.Length - 3]);

            return SqueezeOut(img, needCast, needSqueeze, out_dtype);
        }

        private TorchTensor GetGaussianKernel1d(long size)
        {
            var ksize_half = (size - 1) * 0.5f;
            var x = Float32Tensor.linspace(-ksize_half, ksize_half, size);
            var pdf = -(x / sigma).pow(2) * 0.5f;

            return pdf / pdf.sum();
        }

        private TorchTensor GetGaussianKernel2d(ScalarType dtype, torch.device device)
        {
            var kernel_X = GetGaussianKernel1d(kernelSize[0]).to(dtype, device).index(new TorchTensorIndex[] { TorchTensorIndex.None, TorchTensorIndex.Ellipsis });
            var kernel_Y = GetGaussianKernel1d(kernelSize[1]).to(dtype, device).index(new TorchTensorIndex[] { TorchTensorIndex.Ellipsis, TorchTensorIndex.None });
            return kernel_Y.mm(kernel_X);
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

        protected long[] kernelSize;
        protected float sigma;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Apply a Gaussian blur effect to the image.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        static public ITransform GaussianBlur(IList<long> kernelSize, float sigma)
        {
            return new GaussianBlur(kernelSize, sigma, sigma);
        }

        /// <summary>
        /// Apply a Gaussian blur effect to the image.
        /// </summary>
        static public ITransform GaussianBlur(IList<long> kernelSize, float min = 0.1f, float max = 2.0f)
        {
            return new GaussianBlur(kernelSize, min, max);
        }

        /// <summary>
        /// Apply a Gaussian blur effect to the image.
        /// </summary>
        static public ITransform GaussianBlur(long kernelSize, float sigma)
        {
            return new GaussianBlur(new long[] { kernelSize, kernelSize }, sigma, sigma);
        }

        /// <summary>
        /// Apply a Gaussian blur effect to the image.
        /// </summary>
        static public ITransform GaussianBlur(long kernelSize, float min = 0.1f, float max = 2.0f)
        {
            return new GaussianBlur(new long[] { kernelSize, kernelSize }, min, max);
        }

        /// <summary>
        /// Apply a Gaussian blur effect to the image.
        /// </summary>
        static public ITransform GaussianBlur(long kernelHeight, long kernelWidth, float sigma)
        {
            return new GaussianBlur(new long[] { kernelHeight, kernelWidth }, sigma, sigma);
        }

        /// <summary>
        /// Apply a Gaussian blur effect to the image.
        /// </summary>
        static public ITransform GaussianBlur(long kernelHeight, long kernelWidth, float min = 0.1f, float max = 2.0f)
        {
            return new GaussianBlur(new long[] { kernelHeight, kernelWidth }, min, max);
        }
    }
}
