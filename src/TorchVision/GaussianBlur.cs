// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
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

            public Tensor call(Tensor input)
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
            /// <param name="kernelSize">Gaussian kernel size</param>
            /// <param name="sigma">Gaussian kernel standard deviation</param>
            /// <returns></returns>
            static public ITransform GaussianBlur(IList<long> kernelSize, float sigma)
            {
                return new GaussianBlur(kernelSize, sigma, sigma);
            }

            /// <summary>
            /// Apply a Gaussian blur effect to the image.
            /// </summary>
            /// <param name="kernelSize">Gaussian kernel size</param>
            /// <param name="min">Minimum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            /// <param name="max">Maximum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            static public ITransform GaussianBlur(IList<long> kernelSize, float min = 0.1f, float max = 2.0f)
            {
                return new GaussianBlur(kernelSize, min, max);
            }

            /// <summary>
            /// Apply a Gaussian blur effect to the image.
            /// </summary>
            /// <param name="kernelSize">Gaussian kernel size</param>
            /// <param name="sigma">Gaussian kernel standard deviation</param>
            static public ITransform GaussianBlur(long kernelSize, float sigma)
            {
                return new GaussianBlur(new long[] { kernelSize, kernelSize }, sigma, sigma);
            }

            /// <summary>
            /// Apply a Gaussian blur effect to the image.
            /// </summary>
            /// <param name="kernelSize">Gaussian kernel size</param>
            /// <param name="min">Minimum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            /// <param name="max">Maximum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            static public ITransform GaussianBlur(long kernelSize, float min = 0.1f, float max = 2.0f)
            {
                return new GaussianBlur(new long[] { kernelSize, kernelSize }, min, max);
            }

            /// <summary>
            /// Apply a Gaussian blur effect to the image.
            /// </summary>
            /// <param name="kernelHeight">Gaussian kernel height</param>
            /// <param name="kernelWidth">Gaussian kernel width</param>
            /// <param name="sigma">Gaussian kernel standard deviation</param>
            static public ITransform GaussianBlur(long kernelHeight, long kernelWidth, float sigma)
            {
                return new GaussianBlur(new long[] { kernelHeight, kernelWidth }, sigma, sigma);
            }

            /// <summary>
            /// Apply a Gaussian blur effect to the image.
            /// </summary>
            /// <param name="kernelHeight">Gaussian kernel height</param>
            /// <param name="kernelWidth">Gaussian kernel width</param>
            /// <param name="min">Minimum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            /// <param name="max">Minimum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            static public ITransform GaussianBlur(long kernelHeight, long kernelWidth, float min = 0.1f, float max = 2.0f)
            {
                return new GaussianBlur(new long[] { kernelHeight, kernelWidth }, min, max);
            }
        }
    }
}