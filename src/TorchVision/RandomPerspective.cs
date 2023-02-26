// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class RandomPerspective : ITransform
        {
            public RandomPerspective(double distortion, double p)
            {
                this.distortion = distortion;
                this.p = p;
                this.rnd = new Random();
            }

            public Tensor call(Tensor img)
            {
                if (rnd.NextDouble() <= p) {
                    var _end = img.shape.Length;
                    var w = (int)img.shape[_end - 1];
                    var h = (int)img.shape[_end - 2];

                    var (startpoints, endpoints) = GetParams(w, h);
                    return transforms.functional.perspective(img, startpoints, endpoints);
                }

                return img;
            }

            private int Adjust(double input, double min, double max)
            {
                return (int)(Math.Floor(input * (max - min) + min));
            }

            private (List<IList<int>>, List<IList<int>>) GetParams(int width, int height)
            {
                var half_width = width / 2;
                var half_height = height / 2;

                var randoms = torch.rand(8, ScalarType.Float64).data<double>().ToArray();

                var topleft = new int[] {
                Adjust(randoms[0], 0, distortion * half_width + 1),
                Adjust(randoms[1], 0, distortion * half_height + 1)
            };
                var topright = new int[] {
                Adjust(randoms[2], width - (int)Math.Floor(distortion * half_width) - 1, width),
                Adjust(randoms[3], 0, (int)Math.Floor(distortion * half_height) + 1)
            };
                var botright = new int[] {
                Adjust(randoms[4], width - (int)Math.Floor(distortion * half_width) - 1, width),
                Adjust(randoms[5], height - (int)Math.Floor(distortion * half_height) - 1, height),
            };
                var botleft = new int[] {
                Adjust(randoms[6], 0, (int)Math.Floor(distortion * half_width) + 1),
                Adjust(randoms[7], height - (int)Math.Floor(distortion * half_height) - 1, height),
            };

                var startpoints = new int[][] { new int[] { 0, 0 }, new int[] { width - 1, 0 }, new int[] { width - 1, height - 1 }, new int[] { 0, height - 1 } };
                var endpoints = new int[][] { topleft, topright, botright, botleft }.ToList();

                return (startpoints.Select(x => x.ToList() as IList<int>).ToList(), endpoints.Select(x => x.ToList() as IList<int>).ToList());
            }

            private Random rnd;
            private double p;
            private double distortion;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Performs a random perspective transformation of the given image with a given probability.
            /// </summary>
            /// <param name="distortion">Argument to control the degree of distortion and ranges from 0 to 1. Default is 0.5.</param>
            /// <param name="p">Probability of the image being transformed. Default is 0.5.</param>
            /// <returns></returns>
            /// <remarks>The application and perspectives are all stochastic.</remarks>
            static public ITransform RandomPerspective(double distortion = 0.5, double p = 0.5)
            {
                return new RandomPerspective(distortion, p);
            }
        }
    }
}