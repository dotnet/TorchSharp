// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class GaussianBlur : torch.nn.Module<Tensor,Tensor>, ITransform
        {
            internal GaussianBlur(IList<long> kernelSize, float sigma): base(nameof(GaussianBlur))
            {
                if (kernelSize == null || kernelSize.Count != 2 || kernelSize.Any(x => x <= 0)) {
                    throw new ArgumentException("Invalid kernel size argument.");
                }
                if (sigma <= 0) {
                    throw new ArgumentException("Invalid GaussianBlur arguments: sigma must be positive.");
                }
                this.sigma = sigma;
                this.kernelSize = kernelSize.ToArray();
            }

            internal GaussianBlur(IList<long> kernelSize, float sigma_min, float sigma_max) : base(nameof(GaussianBlur))
            {
                if (kernelSize == null || kernelSize.Count != 2 || kernelSize.Any(x => x <= 0)) {
                    throw new ArgumentException("Invalid kernel size argument.");
                }
                if (sigma_min < 0 || sigma_max < 0 || sigma_min > sigma_max) {
                    throw new ArgumentException("Invalid GaussianBlur arguments: min and max must be positive and min <= max");
                }
                // Leave 'this.sigma' null.
                this.sigma_min = sigma_min;
                this.sigma_max = sigma_max;
                this.kernelSize = kernelSize.ToArray();
            }

            public override Tensor forward(Tensor input)
            {
                var s = sigma.HasValue ? sigma.Value : torch.empty(1).uniform_(sigma_min, sigma_max).item<float>();
                return transforms.functional.gaussian_blur(input, kernelSize, new []{s,s});
            }

            protected long[] kernelSize;
            protected float? sigma;
            protected float sigma_min;
            protected float sigma_max;
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
                return new GaussianBlur(kernelSize, sigma);
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
            /// <param name="kernel_size">Gaussian kernel size</param>
            /// <param name="sigma">Gaussian kernel standard deviation</param>
            static public ITransform GaussianBlur(long kernel_size, float sigma)
            {
                return new GaussianBlur(new long[] { kernel_size, kernel_size }, sigma, sigma);
            }

            /// <summary>
            /// Apply a Gaussian blur effect to the image.
            /// </summary>
            /// <param name="kernel_size">Gaussian kernel size</param>
            /// <param name="min">Minimum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            /// <param name="max">Maximum value of the range for the uniform distribution from which the Gaussian kernel standard deviation will sampled</param>
            static public ITransform GaussianBlur(long kernel_size, float min = 0.1f, float max = 2.0f)
            {
                return new GaussianBlur(new long[] { kernel_size, kernel_size }, min, max);
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