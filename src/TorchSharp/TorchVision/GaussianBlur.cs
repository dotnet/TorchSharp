// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


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

        public Tensor forward(Tensor input)
        {
            return transforms.functional.gaussian_blur(input, kernelSize, new float[] { sigma });
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
