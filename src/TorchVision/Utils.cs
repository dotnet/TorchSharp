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
            /// <summary>
            /// Make a grid of images.
            /// </summary>
            /// <param name="tensors">A list of images all of the same size.</param>
            /// <param name="nrow">Number of images displayed in each row of the grid.
            /// The final grid size is (B / nrow, nrow). Default: 8.</param>
            /// <param name="padding">Amount of padding. Default: 2.</param>
            /// <param name="normalize">If true, shift the image to the range (0, 1),
            /// by the min and max values specified by value_range. Default: false.</param>
            /// <param name="value_range">Tuple (min, max) where min and max are numbers,
            /// then these numbers are used to normalize the image. By default, min and max
            /// are computed from the tensor.</param>
            /// <param name="scale_each">If true, scale each image in the batch of
            /// images separately rather than the (min, max) over all images. Default: false.</param>
            /// <param name="pad_value">Value for the padded pixels. Default: 0.</param>
            /// <returns>The tensor containing grid of images.</returns>
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

            /// <summary>
            /// Make a grid of images.
            /// </summary>
            /// <param name="tensor">4D mini-batch Tensor of shape (B x C x H x W).</param>
            /// <param name="nrow">Number of images displayed in each row of the grid.
            /// The final grid size is (B / nrow, nrow). Default: 8.</param>
            /// <param name="padding">Amount of padding. Default: 2.</param>
            /// <param name="normalize">If true, shift the image to the range (0, 1),
            /// by the min and max values specified by value_range. Default: false.</param>
            /// <param name="value_range">Tuple (min, max) where min and max are numbers,
            /// then these numbers are used to normalize the image. By default, min and max
            /// are computed from the tensor.</param>
            /// <param name="scale_each">If true, scale each image in the batch of
            /// images separately rather than the (min, max) over all images. Default: false.</param>
            /// <param name="pad_value">Value for the padded pixels. Default: 0.</param>
            /// <returns>The tensor containing grid of images.</returns>
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

                if (tensor.size(0) == 1) {
                    tensor = tensor.squeeze(0);
                    return tensor.MoveToOuterDisposeScope();
                }

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

            /// <summary>
            /// Save a given Tensor into an image file.
            /// </summary>
            /// <param name="tensor">Image to be saved. If given a mini-batch tensor,
            /// saves the tensor as a grid of images by calling make_grid.</param>
            /// <param name="filename">A file name</param>
            /// <param name="format">The format to use is not determined from the
            /// filename extension this parameter should always be used.</param>
            /// <param name="nrow">Number of images displayed in each row of the grid.
            /// The final grid size is (B / nrow, nrow). Default: 8.</param>
            /// <param name="padding">Amount of padding. Default: 2.</param>
            /// <param name="normalize">If true, shift the image to the range (0, 1),
            /// by the min and max values specified by value_range. Default: false.</param>
            /// <param name="value_range">Tuple (min, max) where min and max are numbers,
            /// then these numbers are used to normalize the image. By default, min and max
            /// are computed from the tensor.</param>
            /// <param name="scale_each">If true, scale each image in the batch of
            /// images separately rather than the (min, max) over all images. Default: false.</param>
            /// <param name="pad_value">Value for the padded pixels. Default: 0.</param>
            /// <param name="imager">Imager to use instead of DefaultImager. Default: null</param>
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

            /// <summary>
            /// Save a given Tensor into an image file.
            /// </summary>
            /// <param name="tensor">Image to be saved. If given a mini-batch tensor,
            /// saves the tensor as a grid of images by calling make_grid.</param>
            /// <param name="filestream">A file stream</param>
            /// <param name="format">The format to use is not determined from the
            /// filename extension this parameter should always be used.</param>
            /// <param name="nrow">Number of images displayed in each row of the grid.
            /// The final grid size is (B / nrow, nrow). Default: 8.</param>
            /// <param name="padding">Amount of padding. Default: 2.</param>
            /// <param name="normalize">If true, shift the image to the range (0, 1),
            /// by the min and max values specified by value_range. Default: false.</param>
            /// <param name="value_range">Tuple (min, max) where min and max are numbers,
            /// then these numbers are used to normalize the image. By default, min and max
            /// are computed from the tensor.</param>
            /// <param name="scale_each">If true, scale each image in the batch of
            /// images separately rather than the (min, max) over all images. Default: false.</param>
            /// <param name="pad_value">Value for the padded pixels. Default: 0.</param>
            /// <param name="imager">Imager to use instead of DefaultImager. Default: null</param>
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
