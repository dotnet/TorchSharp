// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    internal class Resize : AffineGridBase, ITransform
    {
        internal Resize(int height, int width, InterpolateMode mode, int? maxSize, bool antialias)
        {
            if (antialias && mode != InterpolateMode.Bilinear && mode != InterpolateMode.Bicubic)
                throw new ArgumentException("Antialias option is supported for bilinear and bicubic interpolation modes only");

            this.height = height;
            this.width = width;
            this.mode = mode;
            this.antialias = antialias;
            this.max = maxSize; 
        }

        public Tensor forward(Tensor input)
        {
            var hoffset = input.Dimensions - 2;
            var iHeight = input.shape[hoffset];
            var iWidth = input.shape[hoffset + 1];

            if (iHeight == height && iWidth == width)
                return input;

            var h = height;
            var w = width;

            if (w == -1) {
                if (max.HasValue && height > max.Value)
                    throw new ArgumentException($"maxSize = {max} must be strictly greater than the requested size for the smaller edge size = {height}");

                // Only one size was specified -- retain the aspect ratio.
                if (iHeight < iWidth) {
                    h = height;
                    w = (int)Math.Floor(height * ((double)iWidth / (double)iHeight));
                } else if (iWidth < iHeight) {
                    w = height;
                    h = (int)Math.Floor(height * ((double)iHeight / (double)iWidth));
                }
                else {
                    w = height;
                }
            }

            if (mode != InterpolateMode.Nearest) {
                throw new NotImplementedException("Interpolation mode != 'Nearest'");
            }


            var img = SqueezeIn(input, new ScalarType[] { ScalarType.Float32, ScalarType.Float64 }, out var needCast, out var needSqueeze, out var dtype);

            img = torch.nn.functional.interpolate(img, new long[] { h, w }, mode: mode, align_corners: null);

            return SqueezeOut(img, needCast, needSqueeze, dtype);
        }

        private bool antialias;
        private InterpolateMode mode;
        private int height, width;
        private int? max;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Resize the input image to the given size.
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="mode"></param>
        /// <param name="antialias"></param>
        /// <returns></returns>
        static public ITransform Resize(int height, int width, InterpolateMode mode = InterpolateMode.Nearest, bool antialias = false)
        {
            return new Resize(height, width, mode, null, antialias);
        }

        /// <summary>
        /// Resize the input image to the given size.
        /// </summary>
        /// <param name="size"></param>
        /// <param name="mode"></param>
        /// <param name="maxSize"></param>
        /// <param name="antialias"></param>
        /// <returns></returns>
        static public ITransform Resize(int size, InterpolateMode mode = InterpolateMode.Nearest, int? maxSize = null, bool antialias = false)
        {
            return new Resize(size, -1, mode, maxSize, antialias);
        }
    }
}
