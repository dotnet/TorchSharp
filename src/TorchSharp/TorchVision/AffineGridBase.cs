using System;
using System.Collections.Generic;
using System.Linq;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public class AffineGridBase
    {
        protected Tensor ApplyGridTransform(Tensor img, Tensor grid, GridSampleMode mode, IList<float> fill = null)
        {
            img = SqueezeIn(img, new ScalarType[] { img.dtype }, out var needCast, out var needSqueeze, out var out_dtype);

            if (img.shape[0] > 1) {
                grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]);
            }

            if (fill != null) {
                var dummy = torch.ones(img.shape[0], 1, img.shape[2], img.shape[3], dtype: img.dtype, device: img.device);
                img = torch.cat(new Tensor[] { img, dummy }, dimension: 1);
            }

            img = nn.functional.grid_sample(img, grid, mode: mode, padding_mode: GridSamplePaddingMode.Zeros, align_corners: false);


            if (fill != null) {
                var mask = img[TensorIndex.Colon, TensorIndex.Slice(-1, null), TensorIndex.Colon, TensorIndex.Colon];
                img = img[TensorIndex.Colon, TensorIndex.Slice(null, -1), TensorIndex.Colon, TensorIndex.Colon];
                mask = mask.expand_as(img);

                var len_fill = fill.Count;
                var fill_img = torch.tensor(fill, dtype: img.dtype, device: img.device).view(1, len_fill, 1, 1).expand_as(img);

                if (mode == GridSampleMode.Nearest) {
                    mask = mask < 0.5;
                    img[mask] = fill_img[mask];
                }
                else {
                    img = img * mask + (-mask + 1.0) * fill_img;
                }
            }

            img = SqueezeOut(img, needCast, needSqueeze, out_dtype);
            return img;
        }

        protected Tensor GenerateAffineGrid(Tensor theta, long w, long h, long ow, long oh)
        {
            var d = 0.5;
            var base_grid = torch.empty(1, oh, ow, 3, dtype: theta.dtype, device: theta.device);
            var x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps: ow, device: theta.device);
            base_grid[TensorIndex.Ellipsis, 0].copy_(x_grid);
            var y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps: oh, device: theta.device).unsqueeze_(-1);
            base_grid[TensorIndex.Ellipsis, 1].copy_(y_grid);
            base_grid[TensorIndex.Ellipsis, 2].fill_(1);

            var rescaled_theta = theta.transpose(1, 2) / torch.tensor(new float[] { 0.5f * w, 0.5f * h }, dtype: theta.dtype, device: theta.device);
            var output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta);
            return output_grid.view(1, oh, ow, 2);
        }

        protected (int, int) ComputeOutputSize(IList<float> matrix, long w, long h)
        {
            var pts = torch.tensor(new float[] { -0.5f * w, -0.5f * h, 1.0f, -0.5f * w, 0.5f * h, 1.0f, 0.5f * w, 0.5f * h, 1.0f, 0.5f * w, -0.5f * h, 1.0f}, 4, 3);
            var theta = torch.tensor(matrix, dtype: torch.float32).reshape(1, 2, 3);
            var new_pts = pts.view(1, 4, 3).bmm(theta.transpose(1, 2)).view(4, 2);

            Tensor min_vals = new_pts.min(dimension: 0).values;
            Tensor max_vals = new_pts.max(dimension: 0).values;

            var tol = 1f-4;
            var cmax = torch.ceil((max_vals / tol).trunc_() * tol);
            var cmin = torch.floor((min_vals / tol).trunc_() * tol);

            var size = cmax - cmin;
            return (size[0].ToInt32(), size[1].ToInt32());
        }

        protected IList<float> GetInverseAffineMatrix((float, float) center, float angle, (float, float) translate, float scale, (float, float) shear)
        {
            // Convert to radians.
            var rot = angle * MathF.PI / 180.0f;
            var sx = shear.Item1 * MathF.PI / 180.0f;
            var sy = shear.Item2 * MathF.PI / 180.0f;

            var (cx, cy) = center;
            var (tx, ty) = translate;

            var a = MathF.Cos(rot - sy) / MathF.Cos(sy);
            var b = - MathF.Cos(rot - sy) * MathF.Tan(sx) / MathF.Cos(sy) - MathF.Sin(rot);
            var c = MathF.Sin(rot - sy) / MathF.Cos(sy);
            var d = - MathF.Sin(rot - sy) * MathF.Tan(sx) / MathF.Cos(sy) + MathF.Cos(rot);

            var matrix = (new float[] { d, -b, 0.0f, -c, a, 0.0f }).Select(x => x / scale).ToArray();

            matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty);
            matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty);

            matrix[2] += cx;
            matrix[5] += cy;

            return matrix;
        }

        protected (long, long) GetImageSize(Tensor img)
        {
            var hOffset = img.shape.Length - 2;
            return (img.shape[hOffset+1], img.shape[hOffset]);
        }

        protected Tensor SqueezeIn(Tensor img, IList<ScalarType> req_dtypes, out bool needCast, out bool needSqueeze, out ScalarType dtype)
        {
            needSqueeze = false;

            if (img.Dimensions < 4) {
                img = img.unsqueeze(0);
                needSqueeze = true;
            }

            dtype = img.dtype;
            needCast = false;

            if (!req_dtypes.Contains(dtype)) {
                needCast = true;
                img = img.to_type(req_dtypes[0]);
            }

            return img;
        }

        protected Tensor SqueezeOut(Tensor img, bool needCast, bool needSqueeze, ScalarType dtype)
        {
            if (needSqueeze) {
                img = img.squeeze(0);
            }

            if (needCast) {
                if (TensorExtensionMethods.IsIntegral(dtype))
                    img = img.round();

                img = img.to_type(dtype);
            }

            return img;
        }


    }
}
