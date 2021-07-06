using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    internal class Rotate : AffineGridBase, ITransform
    {
        public torch.Tensor forward(torch.Tensor img)
        {
            var center_f = (0.0f, 0.0f);

            if (center.HasValue) {
                var img_size = GetImageSize(img);
                center_f = (1.0f * (center.Value.Item1 - img_size.Item1 * 0.5f), 1.0f * (center.Value.Item2 - img_size.Item2 * 0.5f));
            }

            var matrix = GetInverseAffineMatrix(center_f, -angle, (0.0f, 0.0f), 1.0f, (0.0f, 0.0f));

            return RotateImage(img, matrix, interpolation, expand, fill);
        }

        private Tensor RotateImage(Tensor img, IList<float> matrix, GridSampleMode interpolation, bool expand, IList<float> fill)
        {
            var (w, h) = GetImageSize(img);
            var (ow, oh) = expand ? ComputeOutputSize(matrix, w, h) : (w,h);
            var dtype = torch.is_floating_point(img) ? img.dtype : torch.float32;
            var theta = torch.tensor(matrix, dtype: dtype, device: img.device).reshape(1, 2, 3);
            var grid = GenerateAffineGrid(theta, w, h, ow, oh);

            return ApplyGridTransform(img, grid, interpolation, fill);
        }

        internal Rotate(float angle, GridSampleMode interpolation = GridSampleMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
        {
            this.angle = angle;
            this.fill = fill;
            this.expand = expand;
            this.center = center;
            this.interpolation = interpolation;
        }

        private float angle;
        private bool expand;
        private (int, int)? center;
        private IList<float> fill;
        private GridSampleMode interpolation;
    }

    public static partial class transforms
    {
        static public ITransform Rotate(float angle, GridSampleMode interpolation = GridSampleMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
        {
            return new Rotate(angle, interpolation, expand, center, fill);
        }
    }
}