using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
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

        public torch.Tensor forward(torch.Tensor img)
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
        static public ITransform Rotate(float angle, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
        {
            return new Rotate(angle, interpolation, expand, center, fill);
        }
    }
}