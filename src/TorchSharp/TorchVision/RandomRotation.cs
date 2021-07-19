using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    internal class RandomRotation : ITransform
    {
        public RandomRotation((double, double) degrees, InterpolationMode mode = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
        {
            this.degrees = degrees;
            this.mode = mode;
            this.center = center;
            this.expand = expand;
            this.fill = fill;
        }

        public Tensor forward(Tensor input)
        {
            var random = new Random();
            var angle = random.NextDouble() * (degrees.Item2 - degrees.Item1) + degrees.Item1;

            var rotate = torchvision.transforms.Rotate((float)angle, mode, expand, center, fill);
            return rotate.forward(input);
        }

        private (double, double) degrees;
        private bool expand;
        private (int, int)? center;
        private IList<float> fill;
        private InterpolationMode mode;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Rotate the image by a random angle. 
        /// </summary>
        static public ITransform RandomRotation(double degrees, InterpolationMode mode = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
        {
            return new RandomRotation((-degrees, degrees), mode, expand, center, fill);
        }

        /// <summary>
        ///Rotate the image by a random angle.  
        /// </summary>
        static public ITransform RandomRotation((double, double) degrees, InterpolationMode mode = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
        {
            return RandomRotation(degrees, mode, expand, center, fill);
        }
    }
}
