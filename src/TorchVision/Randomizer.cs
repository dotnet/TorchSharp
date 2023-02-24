// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Randomizer : ITransform
        {
            internal Randomizer(ITransform transform, double p = 0.1)
            {
                this.p = p;
                this.transform = transform;
            }

            public Tensor call(Tensor input)
            {
                using (var chance = torch.rand(1))

                    if (chance.item<float>() < p) {
                        return transform.call(input);
                    } else {
                        return input;
                    }
            }

            private ITransform transform;
            private double p;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Randomly apply a transform given a probability.
            /// </summary>
            /// <param name="transform">The transform that will be applied randomly.</param>
            /// <param name="p">The probablity of applying the transform</param>
            /// <returns></returns>
            /// <remarks>This uses the default TorchSharp RNG.</remarks>
            static public ITransform Randomize(ITransform transform, double p = 0.1)
            {
                return new Randomizer(transform, p);
            }

            /// <summary>
            /// Posterize the image randomly with a given probability by reducing the number of bits for each color channel. 
            /// </summary>
            /// <param name="bits">Number of bits to keep for each channel (0-8)</param>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            /// <remarks>The tensor must be an integer tensor</remarks>
            static public ITransform RandomPosterize(int bits, double p = 0.5)
            {
                return new Randomizer(Posterize(bits), p);
            }

            /// <summary>
            /// Equalize the image randomly with a given probability. 
            /// </summary>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            /// <remarks>The tensor must be an integer tensor</remarks>
            static public ITransform Equalize(double p = 0.5)
            {
                return new Randomizer(Equalize(), p);
            }

            /// <summary>
            /// Solarize the image randomly with a given probability by inverting all pixel values above a threshold.
            /// </summary>
            /// <param name="threshold">All pixels equal or above this value are inverted.</param>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomSolarize(double threshold, double p = 0.5)
            {
                return new Randomizer(Solarize(threshold), p);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomVerticalFlip(double p = 0.5)
            {
                return new Randomizer(VerticalFlip(), p);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomHorizontalFlip(double p = 0.5)
            {
                return new Randomizer(HorizontalFlip(), p);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomGrayscale(double p = 0.1)
            {
                return new Randomizer(Grayscale(3), p);
            }

            /// <summary>
            /// Adjust the sharpness of the image randomly with a given probability. 
            /// </summary>
            /// <param name="sharpness">The sharpness factor</param>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomAdjustSharpness(double sharpness, double p = 0.5)
            {
                return new Randomizer(AdjustSharpness(sharpness), p);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomAutoContrast(double p = 0.5)
            {
                return new Randomizer(AutoContrast(), p);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="p">Probability of the transform being applied.</param>
            /// <returns></returns>
            static public ITransform RandomInvert(double p = 0.5)
            {
                return new Randomizer(Invert(), p);
            }

            /// <summary>
            /// Apply randomly a list of transformations with a given probability.
            /// </summary>
            /// <param name="transforms">A list of transforms to compose serially.</param>
            /// <param name="p">Probability of the transforms being applied.</param>
            /// <returns></returns>
            static public ITransform RandomApply(ITransform[] transforms, double p = 0.5)
            {
                return new Randomizer(Compose(transforms), p);
            }
        }
    }
}