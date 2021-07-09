using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    internal class RandomResizedCrop : ITransform
    {
        public RandomResizedCrop(int height, int width, double scaleMin, double scaleMax, double ratioMin, double ratioMax)
        {
            this.height = height;
            this.width = width;
            this.scaleMax = scaleMax;
            this.scaleMin = scaleMin;
            this.ratioMax = ratioMax;
            this.ratioMin = ratioMin;
        }

        public Tensor forward(Tensor input)
        {
            var hoffset = input.Dimensions - 2;
            var iHeight = input.shape[hoffset];
            var iWidth = input.shape[hoffset + 1];

            var scale = scaleMax - scaleMin;

            var random = new Random();

            // First, figure out how high and wide the crop should be.

            var randomScale = Math.Min(random.NextDouble() * scale + scaleMin, 1); // Preserves the aspect ratio.

            var h = iHeight * randomScale;
            var w = iWidth * randomScale;

            // Then, place the top and left corner at a random location,
            // so that the crop doesn't go outside the image boundaries.

            var top = (int)Math.Floor((iHeight - h) * random.NextDouble());
            var left = (int)Math.Floor((iWidth - w) * random.NextDouble());

            return new ResizedCrop(top, left, (int)Math.Floor(h), (int)Math.Floor(w), height, width).forward(input);
        }

        private int height, width;
        private double scaleMin, scaleMax;
        private double ratioMin, ratioMax;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Crop a random portion of image and resize it to a given size. 
        /// </summary>
        static public ITransform RandomResizedCrop(int height, int width, double scaleMin = 0.08, double scaleMax = 1.0, double ratioMin = 0.75, double ratioMax = 1.3333333333333)
        {
            return new RandomResizedCrop(height, width, scaleMin, scaleMax, ratioMin, ratioMax);
        }

        /// <summary>
        /// Crop a random portion of image and resize it to a given size. 
        /// </summary>
        static public ITransform RandomResizedCrop(int size, double scaleMin = 0.08, double scaleMax = 1.0, double ratioMin = 0.75, double ratioMax = 1.3333333333333)
        {
            return RandomResizedCrop(size, size, scaleMin, scaleMax, ratioMin, ratioMax);
        }
    }
}
