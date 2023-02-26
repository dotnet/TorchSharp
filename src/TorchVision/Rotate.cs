// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Rotate : ITransform
        {
            internal Rotate(float angle, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                this.angle = angle;
                this.fill = fill;
                this.expand = expand;
                this.center = center;
                this.interpolation = interpolation;
            }

            public torch.Tensor call(torch.Tensor img)
            {
                return transforms.functional.rotate(img, angle, interpolation, expand, center, fill);
            }

            private float angle;
            private bool expand;
            private (int, int)? center;
            private IList<float> fill;
            private InterpolationMode interpolation;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Rotate the image by angle, counter-clockwise.
            /// </summary>
            /// <param name="angle">Angle by which to rotate.</param>
            /// <param name="interpolation">Desired interpolation enum. Default is `InterpolationMode.NEAREST`.</param>
            /// <param name="expand">If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.</param>
            /// <param name="center">Center of rotation, (x, y). Origin is the upper left corner.</param>
            /// <param name="fill">Pixel fill value for the area outside the rotated. If given a number, the value is used for all bands respectively.</param>
            static public ITransform Rotate(float angle, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                return new Rotate(angle, interpolation, expand, center, fill);
            }
        }
    }
}