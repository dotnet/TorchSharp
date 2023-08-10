using System;
using System.Collections.Generic;
using System.IO;
using static TorchSharp.torch;
using static TorchSharp.torchvision.io;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static class utils
        {
            public static Tensor make_grid(
                IEnumerable<Tensor> tensors,
                long nrow = 8,
                int padding = 2,
                bool normalize = false,
                (double low, double high)? value_range = null,
                bool scale_each = false,
                double pad_value = 0.0f)
            {
                return make_grid(torch.stack(tensors, dim: 0), nrow, padding, normalize, value_range, scale_each, pad_value);
            }

            public static Tensor make_grid(
                Tensor tensor,
                long nrow = 8,
                int padding = 2,
                bool normalize = false,
                (double low, double high)? value_range = null,
                bool scale_each = false,
                double pad_value = 0.0f)
            {
                using var _ = torch.NewDisposeScope();

                if (tensor.Dimensions == 2) // Single image H x W
                {
                    tensor = tensor.unsqueeze(0);
                }
                if (tensor.Dimensions == 3) // Single image
                {
                    if (tensor.size(0) == 1) // Convert single channel to 3 channel
                    {
                        tensor = torch.cat(new[] { tensor, tensor, tensor }, 0);
                    }
                    tensor = tensor.unsqueeze(0);
                }
                if (tensor.Dimensions == 4 && tensor.size(1) == 1) // Single channel images
                {
                    tensor = torch.cat(new[] { tensor, tensor, tensor }, 1);
                }

                if (normalize == true)
                {
                    tensor = tensor.clone();  // avoid modifying tensor in-place
                    void norm_ip(Tensor img, double low, double high)
                    {
                        img.clamp_(min: low, max: high);
                        img.sub_(low).div_(Math.Max(high - low, 1e-5));
                    }

                    void norm_range(Tensor t, (double low, double high)? range)
                    {
                        if (range.HasValue)
                        {
                            var (low, high) = value_range.Value;
                            norm_ip(t, low, high);
                        }
                        else
                        {
                            norm_ip(t, t.min().ToSingle(), t.max().ToSingle());
                        }
                    }

                    if (scale_each == true)
                    {
                        for (long i = 0; i < tensor.size(0); ++i)
                        {
                            norm_range(tensor[i], value_range);
                        }
                    }
                    else
                    {
                        norm_range(tensor, value_range);
                    }
                }

                if (tensor.size(0) == 1)
                    return tensor.unsqueeze(0);

                var nmaps = tensor.size(0);
                var xmaps = Math.Min(nrow, nmaps);
                var ymaps = (long)Math.Ceiling((double)nmaps / xmaps);
                var width = tensor.size(3) + padding;
                var height = tensor.size(2) + padding;
                var num_channels = tensor.size(1);

                var grid = tensor.new_full(new[] { num_channels, height * ymaps + padding, width * xmaps + padding }, pad_value);
                var k = 0L;
                for (long y = 0; y < ymaps; ++y)
                {
                    for (long x = 0; x < xmaps; ++x)
                    {
                        if (k > nmaps) break;
                        grid.narrow(1, y * height, height - padding).narrow(
                            2, x * width + padding, width - padding
                            ).copy_(tensor[k]);
                        ++k;
                    }
                }

                return grid.MoveToOuterDisposeScope();
            }

            public static void save_image(
                Tensor tensor,
                string filename,
                ImageFormat format,
                long nrow = 8,
                int padding = 2,
                bool normalize = false,
                (double low, double high)? value_range = null,
                bool scale_each = false,
                double pad_value = 0.0f,
                Imager imager = null)
            {
                using var filestream = new FileStream(filename, FileMode.OpenOrCreate);
                save_image(tensor, filestream, format, nrow, padding, normalize, value_range, scale_each, pad_value, imager);
            }

            public static void save_image(
                Tensor tensor,
                Stream filestream,
                ImageFormat format,
                long nrow = 8,
                int padding = 2,
                bool normalize = false,
                (double low, double high)? value_range = null,
                bool scale_each = false,
                double pad_value = 0.0f,
                Imager imager = null)
            {
                using var _ = torch.NewDisposeScope();
                var grid = make_grid(tensor, nrow, padding, normalize, value_range, scale_each, pad_value);
                // Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
                var narr = grid.mul(255).add_(0.5).clamp_(0, 255).to(uint8, CPU);
                (imager ?? DefaultImager).EncodeImage(narr, format, filestream);
            }
        }
    }
}
