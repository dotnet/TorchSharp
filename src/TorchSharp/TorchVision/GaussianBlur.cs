// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


namespace TorchSharp.torchvision
{
    internal class GaussianBlur : AffineGridBase, ITransform
    {
        internal GaussianBlur(IList<long> kernelSize, float min, float max)
        {
            this.sigma = (min == max) ?
                min :
                (float)(new Random().NextDouble() * (max - min) + min);
            this.kernelSize = kernelSize.ToArray();
        }

        public Tensor forward(Tensor input)
        {
            var dtype = TensorExtensionMethods.IsIntegral(input.dtype) ? ScalarType.Float32 : input.dtype;

            var kernel = GetGaussianKernel2d(dtype, input.device);
            kernel = kernel.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

            var img = SqueezeIn(input, new ScalarType[] { kernel.dtype }, out var needCast, out var needSqueeze, out var out_dtype);

            // The padding needs to be adjusted to make sure that the output is the same size as the input.

            var k0d2 = kernelSize[0] / 2;
            var k1d2 = kernelSize[1] / 2;
            var k0sm1 = kernelSize[0] - 1;
            var k1sm1 = kernelSize[1] - 1;

            var padding = new long[] { k0d2, k0sm1 - k0d2, k1d2, k1sm1 - k1d2 };

            img = TorchSharp.torch.nn.functional.pad(img, padding, PaddingModes.Reflect);
            img = torch.nn.functional.conv2d(img, kernel, groups: img.shape[img.shape.Length - 3]);

            return SqueezeOut(img, needCast, needSqueeze, out_dtype);
        }

        private Tensor GetGaussianKernel1d(long size)
        {
            var ksize_half = (size - 1) * 0.5f;
            var x = Float32Tensor.linspace(-ksize_half, ksize_half, size);
            var pdf = -(x / sigma).pow(2) * 0.5f;

            return pdf / pdf.sum();
        }

        private Tensor GetGaussianKernel2d(ScalarType dtype, torch.Device device)
        {
            var kernel_X = GetGaussianKernel1d(kernelSize[0]).to(dtype, device)[TensorIndex.None, TensorIndex.Slice()];
            var kernel_Y = GetGaussianKernel1d(kernelSize[1]).to(dtype, device)[TensorIndex.Slice(), TensorIndex.None];
            return kernel_Y.mm(kernel_X);
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
