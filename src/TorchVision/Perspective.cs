// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Perspective : ITransform
        {
            internal Perspective(IList<IList<int>> startpoints, IList<IList<int>> endpoints, InterpolationMode interpolation, IList<float> fill = null)
            {
                if (interpolation != InterpolationMode.Nearest && interpolation != InterpolationMode.Bilinear)
                    throw new ArgumentException($"Invalid interpolation mode for 'Perspective': {interpolation}. Use 'nearest' or 'bilinear'.");

                this.startpoints = startpoints;
                this.endpoints = endpoints;
                this.interpolation = interpolation;
                this.fill = fill;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.perspective(input, startpoints, endpoints, interpolation, fill);
            }

            private IList<IList<int>> startpoints;
            private IList<IList<int>> endpoints;
            private InterpolationMode interpolation;
            private IList<float> fill;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Perform perspective transform of the given image.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <returns></returns>
            static public ITransform Perspective(IList<IList<int>> startpoints, IList<IList<int>> endpoints, InterpolationMode interpolation = InterpolationMode.Bilinear, IList<float> fill = null)
            {
                return new Perspective(startpoints, endpoints, interpolation, fill);
            }
        }
    }
}