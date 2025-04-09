// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Resize : ITransform
        {
            internal Resize(int height, int width, InterpolationMode interpolation, int? maxSize, bool antialias)
            {
                this.height = height;
                this.width = width;
                this.interpolation = interpolation;
                this.maxSize = maxSize;
                this.antialias = antialias;
            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.resize(input, height, width, interpolation, maxSize, antialias);
            }

            private int height, width;
            private InterpolationMode interpolation;
            private int? maxSize;
            private bool antialias;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="height">Desired output height</param>
            /// <param name="width">Desired output width</param>
            /// <param name="interpolation">
            /// Desired interpolation enum defined by TorchSharp.torch.InterpolationMode.
            /// Default is InterpolationMode.Nearest; not InterpolationMode.Bilinear (incompatible to Python's torchvision v0.17 or later for historical reasons).
            /// Only InterpolationMode.Nearest, InterpolationMode.NearestExact, InterpolationMode.Bilinear and InterpolationMode.Bicubic are supported.
            /// </param>
            /// <param name="maxSize">The maximum allowed for the longer edge of the resized image.</param>
            /// <param name="antialias">
            /// Whether to apply antialiasing.
            /// It only affects bilinear or bicubic modes and it is ignored otherwise.
            /// Possible values are:
            /// * true: will apply antialiasing for bilinear or bicubic modes. Other mode aren't affected. This is probably what you want to use.
            /// * false (default, incompatible to Python's torchvision v0.17 or later for historical reasons): will not apply antialiasing on any mode.
            /// </param>
            /// <returns></returns>
            static public ITransform Resize(int height, int width, InterpolationMode interpolation = InterpolationMode.Nearest, int? maxSize = null, bool antialias = false)
            {
                return new Resize(height, width, interpolation, maxSize, antialias);
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="size">Desired output size</param>
            /// <param name="interpolation">
            /// Desired interpolation enum defined by TorchSharp.torch.InterpolationMode.
            /// Default is InterpolationMode.Nearest; not InterpolationMode.Bilinear (incompatible to Python's torchvision v0.17 or later for historical reasons).
            /// Only InterpolationMode.Nearest, InterpolationMode.NearestExact, InterpolationMode.Bilinear and InterpolationMode.Bicubic are supported.
            /// </param>
            /// <param name="maxSize">The maximum allowed for the longer edge of the resized image.</param>
            /// <param name="antialias">
            /// Whether to apply antialiasing.
            /// It only affects bilinear or bicubic modes and it is ignored otherwise.
            /// Possible values are:
            /// * true: will apply antialiasing for bilinear or bicubic modes. Other mode aren't affected. This is probably what you want to use.
            /// * false (default, incompatible to Python's torchvision v0.17 or later for historical reasons): will not apply antialiasing on any mode.
            /// </param>
            static public ITransform Resize(int size, InterpolationMode interpolation = InterpolationMode.Nearest, int? maxSize = null, bool antialias = false)
            {
                return new Resize(size, -1, interpolation, maxSize, antialias);
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="height">Desired output height</param>
            /// <param name="width">Desired output width</param>
            /// <param name="maxSize">The maximum allowed for the longer edge of the resized image.</param>
            /// <returns></returns>
            static public ITransform Resize(int height, int width, int? maxSize = null)
            {
                return new Resize(height, width, InterpolationMode.Nearest, maxSize, false);
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="height">Desired output height</param>
            /// <param name="width">Desired output width</param>
            /// <returns></returns>
            static public ITransform Resize(int height, int width)
            {
                return new Resize(height, width, InterpolationMode.Nearest, null, false);
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="size">Desired output size</param>
            /// <param name="maxSize">The maximum allowed for the longer edge of the resized image.</param>
            static public ITransform Resize(int size, int? maxSize = null)
            {
                return new Resize(size, -1, InterpolationMode.Nearest, maxSize, false);
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="size">Desired output size</param>
            static public ITransform Resize(int size)
            {
                return new Resize(size, -1, InterpolationMode.Nearest, null, false);
            }
        }
    }
}