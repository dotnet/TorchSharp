// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class RandomRotation : ITransform
        {
            public RandomRotation((double, double) degrees, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                this.degrees = degrees;
                this.interpolation = interpolation;
                this.center = center;
                this.expand = expand;
                this.fill = fill;
            }

            public Tensor call(Tensor input)
            {
                var random = new Random();
                var angle = random.NextDouble() * (degrees.Item2 - degrees.Item1) + degrees.Item1;

                var rotate = torchvision.transforms.Rotate((float)angle, interpolation, expand, center, fill);
                return rotate.call(input);
            }

            private (double, double) degrees;
            private bool expand;
            private (int, int)? center;
            private IList<float> fill;
            private InterpolationMode interpolation;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Rotate the image by a random angle. 
            /// </summary>
            /// <param name="degrees">Range of degrees to select from (-degrees, +degrees).</param>
            /// <param name="interpolation">Desired interpolation enum. Default is `InterpolationMode.NEAREST`.</param>
            /// <param name="expand">If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.</param>
            /// <param name="center">Center of rotation, (x, y). Origin is the upper left corner.</param>
            /// <param name="fill">Pixel fill value for the area outside the rotated. If given a number, the value is used for all bands respectively.</param>
            static public ITransform RandomRotation(double degrees, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                return new RandomRotation((-degrees, degrees), interpolation, expand, center, fill);
            }

            /// <summary>
            ///Rotate the image by a random angle.  
            /// </summary>
            /// <param name="degrees">Range of degrees to select from</param>
            /// <param name="interpolation">Desired interpolation enum. Default is `InterpolationMode.NEAREST`.</param>
            /// <param name="expand">If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.</param>
            /// <param name="center">Center of rotation, (x, y). Origin is the upper left corner.</param>
            /// <param name="fill">Pixel fill value for the area outside the rotated. If given a number, the value is used for all bands respectively.</param>
            static public ITransform RandomRotation((double, double) degrees, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                return new RandomRotation(degrees, interpolation, expand, center, fill);
            }
        }
    }
}