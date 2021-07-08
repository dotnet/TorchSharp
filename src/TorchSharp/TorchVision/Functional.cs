using System;
using System.Collections.Generic;
using System.Linq;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class transforms
    {
        public static partial class functional
        {

            public static Tensor adjust_brightness(Tensor img, double brightness_factor)
            {
                if (brightness_factor == 1.0)
                    // Special case -- no change.
                    return img;

                return Blend(img, torch.zeros_like(img), brightness_factor);
            }

            public static Tensor adjust_contrast(Tensor img, double contrast_factor)
            {
                if (contrast_factor == 1.0)
                    // Special case -- no change.
                    return img;

                var dtype = torch.is_floating_point(img) ? img.dtype : torch.float32;
                var mean = torch.mean(transforms.functional.rgb_to_grayscale(img).to_type(dtype), new long[] { -3, -2, -1 }, keepDimension: true);
                return Blend(img, mean, contrast_factor);
            }

            public static Tensor adjust_gamma(Tensor img, double gamma, double gain = 1.0)
            {
                var dtype = img.dtype;
                if (!torch.is_floating_point(img))
                    img = transforms.functional.convert_image_dtype(img, torch.float32);

                img = (gain * img.pow(gamma)).clamp(0, 1);

                return convert_image_dtype(img, dtype);
            }

            public static Tensor adjust_hue(Tensor img, double hue_factor)
            {
                if (hue_factor == 0.0)
                    // Special case -- no change.
                    return img;

                if (img.shape.Length < 4 || img.shape[img.shape.Length - 3] == 1)
                    return img;

                var orig_dtype = img.dtype;
                if (!torch.is_floating_point(img))
                    img = img.to_type(torch.float32) / 255.0;

                var HSV = RGBtoHSV(img);

                HSV.h = (HSV.h + hue_factor) % 1.0;

                var img_hue_adj = HSVtoRGB(HSV.h, HSV.s, HSV.v);

                if (orig_dtype.IsIntegral())
                    img_hue_adj = (img_hue_adj * 255.0).to_type(orig_dtype);

                return img_hue_adj;
            }

            public static Tensor adjust_saturation(Tensor img, double saturation_factor)
            {
                if (saturation_factor == 1.0)
                    // Special case -- no change.
                    return img;

                return Blend(img, transforms.functional.rgb_to_grayscale(img), saturation_factor);
            }

            public static Tensor adjust_sharpness(Tensor input, double sharpness)
            {
                if (input.shape[input.shape.Length - 1] <= 2 || input.shape[input.shape.Length - 2] <= 2)
                    return input;

                return Blend(input, BlurredDegenerateImage(input), sharpness);
            }

            public static Tensor autocontrast(Tensor input)
            {
                var bound = input.IsIntegral() ? 255.0f : 1.0f;
                var dtype = input.IsIntegral() ? ScalarType.Float32 : input.dtype;

                var minimum = input.amin(new long[] { -2, -1 }, keepDim: true).to(dtype);
                var maximum = input.amax(new long[] { -2, -1 }, keepDim: true).to(dtype);

                var eq_idxs = (minimum == maximum).nonzero_as_list()[0];
                minimum.index_put_(0, eq_idxs);
                maximum.index_put_(bound, eq_idxs);

                var scale = Float32Tensor.from(bound) / (maximum - minimum);

                return ((input - minimum) * scale).clamp(0, bound).to(input.dtype);
            }

            /// <summary>
            /// Crops the given image at the center. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than
            /// output size along any edge, image is padded with 0 and then center cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="height">The height of the crop box.</param>
            /// <param name="width">The width of the crop box.</param>
            /// <returns></returns>
            public static Tensor center_crop(Tensor input, int height, int width)
            {
                var hoffset = input.Dimensions - 2;
                var iHeight = input.shape[hoffset];
                var iWidth = input.shape[hoffset + 1];

                var top = (int)(iHeight - height) / 2;
                var left = (int)(iWidth - width) / 2;

                return input.crop(top, left, height, width);
            }

            /// <summary>
            /// Crops the given image at the center. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than
            /// output size along any edge, image is padded with 0 and then center cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="size">The size of the crop box.</param>
            /// <returns></returns>
            public static Tensor center_crop(Tensor input, int size) => center_crop(input, size, size);

            public static Tensor convert_image_dtype(Tensor image, ScalarType dtype = ScalarType.Float32)
            {
                if (image.dtype == dtype)
                    return image;

                var output_max = MaxValue(dtype);

                if (torch.is_floating_point(image)) {

                    if (torch.is_floating_point(dtype)) {
                        return image.to_type(dtype);
                    }

                    if ((image.dtype == torch.float32 && (dtype == torch.int32 || dtype == torch.int64)) ||
                        (image.dtype == torch.float64 && dtype == torch.int64)) {
                        throw new ArgumentException($"The cast from {image.dtype} to {dtype} cannot be performed safely.");
                    }

                    var eps = 1e-3;
                    var result = image.mul(output_max + 1.0 - eps);
                    return result.to_type(dtype);

                } else {
                    // Integer to floating point.

                    var input_max = MaxValue(image.dtype);

                    if (torch.is_floating_point(dtype)) {
                        return image.to_type(dtype) / input_max;
                    }

                    if (input_max > output_max) {
                        var factor = (input_max + 1) / (output_max + 1);
                        image = torch.div(image, factor);
                        return image.to_type(dtype);
                    } else {
                        var factor = (output_max + 1) / (input_max + 1);
                        image = image.to_type(dtype);
                        return image * factor;
                    }
                }
            }

            /// <summary>
            /// Crop the given image at specified location and output size. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge,
            /// image is padded with 0 and then cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="height">The height of the crop box.</param>
            /// <param name="width">The width of the crop box.</param>
            /// <returns></returns>
            public static Tensor crop(Tensor input, int top, int left, int height, int width)
            {
                return input.crop(top, left, height, width);
            }

            /// <summary>
            /// Crop the given image at specified location and output size. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge,
            /// image is padded with 0 and then cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="size">The size of the crop box.</param>
            /// <returns></returns>
            public static Tensor crop(Tensor input, int top, int left, int size)
            {
                return crop(input, top, left, size, size);
            }

            public static Tensor equalize(Tensor input)
            {
                throw new NotImplementedException("equalize not yet implemented");
            }

            /// <summary>
            /// Erase the input Tensor Image with given value. 
            /// </summary>
            /// <param name="img">The input tensor</param>
            /// <param name="top">Vertical component of the top left corner of the erased region.</param>
            /// <param name="left">Horizontal component of the top left corner of the erased region.</param>
            /// <param name="height">The height of the erased region.</param>
            /// <param name="width">The width of the erased region.</param>
            /// <param name="value">Erasing value.</param>
            /// <param name="inplace">For in-place operations.</param>
            /// <returns></returns>
            public static Tensor erase(Tensor img, int top, int left, int height, int width, Tensor value, bool inplace = false)
            {
                if (!inplace)
                    img = img.clone();

                img[TensorIndex.Ellipsis, top..(top + height), left..(left + width)] = value;
                return img;
            }

            /// <summary>
            /// Performs Gaussian blurring on the image by given kernel.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. 
            /// </summary>
            /// <returns></returns>
            public static Tensor gaussian_blur(Tensor input, IList<long> kernelSize, IList<float> sigma)
            {
                var dtype = TensorExtensionMethods.IsIntegral(input.dtype) ? ScalarType.Float32 : input.dtype;

                if (kernelSize.Count == 1) {
                    kernelSize = new long[] { kernelSize[0], kernelSize[0] };
                }

                if (sigma == null) {
                    sigma = new float[] {
                        0.3f * ((kernelSize[0] - 1) * 0.5f - 1) + 0.8f,
                        0.3f * ((kernelSize[1] - 1) * 0.5f - 1) + 0.8f,
                    };
                }
                else if (sigma.Count == 1) {
                    sigma = new float[] {
                        sigma[0],
                        sigma[0],
                    };
                }
                var kernel = GetGaussianKernel2d(kernelSize, sigma, dtype, input.device);
                kernel = kernel.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

                var img = SqueezeIn(input, new ScalarType[] { kernel.dtype }, out var needCast, out var needSqueeze, out var out_dtype);

                // The padding needs to be adjusted to make sure that the output is the same size as the input.

                var k0d2 = kernelSize[0] / 2;
                var k1d2 = kernelSize[1] / 2;
                var k0sm1 = kernelSize[0] - 1;
                var k1sm1 = kernelSize[1] - 1;

                var padding = new long[] { k0d2, k1d2, k0sm1 - k0d2, k1sm1 - k1d2 };

                img = TorchSharp.torch.nn.functional.pad(img, padding, PaddingModes.Reflect);
                img = torch.nn.functional.conv2d(img, kernel, groups: img.shape[img.shape.Length - 3]);

                return SqueezeOut(img, needCast, needSqueeze, out_dtype);
            }

            /// <summary>
            /// Performs Gaussian blurring on the image by given kernel.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. 
            /// </summary>
            /// <returns></returns>
            public static Tensor gaussian_blur(Tensor input, long kernelSize, float sigma)
            {
                return gaussian_blur(input, new long[] { kernelSize, kernelSize }, new float[] { sigma });
            }

            /// <summary>
            /// Performs Gaussian blurring on the image by given kernel.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. 
            /// </summary>
            /// <returns></returns>
            public static Tensor gaussian_blur(Tensor input, long kernelHeight, long kernelWidth, float sigma_x, float sigma_y)
            {
                return gaussian_blur(input, new long[] { kernelHeight, kernelWidth }, new float[] { sigma_x, sigma_y });
            }

            public static Tensor hflip(Tensor input) => input.flip(-1);

            public static Tensor invert(Tensor input)
            {
                if (input.IsIntegral()) {
                    return -input + 255;
                } else {
                    return -input + 1.0;
                }
            }

            public static Tensor normalize(Tensor input, double[] means, double[] stdevs, ScalarType dtype = ScalarType.Float32)
            {
                if (means.Length != stdevs.Length)
                    throw new ArgumentException("means and stdevs must be the same length in call to Normalize");
                if (means.Length != input.shape[1])
                    throw new ArgumentException("The number of channels is not equal to the number of means and standard deviations");

                var mean = means.ToTensor(new long[] { 1, means.Length, 1, 1 }).to(input.dtype, input.device);     // Assumes NxCxHxW
                var stdev = stdevs.ToTensor(new long[] { 1, stdevs.Length, 1, 1 }).to(input.dtype, input.device);  // Assumes NxCxHxW

                return (input - mean) / stdev;
            }

            public static Tensor pad(Tensor input, long[] padding, int fill = 0, PaddingModes padding_mode = PaddingModes.Constant)
            {
                return torch.nn.functional.pad(input, padding, padding_mode, fill);
            }

            public static Tensor perspective(Tensor input)
            {
                throw new NotImplementedException("perspective not yet implemented");
            }

            public static Tensor posterize(Tensor input, int bits)
            {
                if (input.dtype != ScalarType.Byte) throw new ArgumentException("Only torch.byte image tensors are supported");
                var mask = -(1 << (8 - bits));
                return input & ByteTensor.from((byte)mask);
            }

            public static Tensor resize(Tensor input, int height, int width, InterpolateMode mode, int? maxSize = null, bool antialias = false)
            {
                if (antialias && mode != InterpolateMode.Bilinear && mode != InterpolateMode.Bicubic)
                    throw new ArgumentException("Antialias option is supported for bilinear and bicubic interpolation modes only");

                var hoffset = input.Dimensions - 2;
                var iHeight = input.shape[hoffset];
                var iWidth = input.shape[hoffset + 1];

                if (iHeight == height && iWidth == width)
                    return input;

                var h = height;
                var w = width;

                if (w == -1) {
                    if (maxSize.HasValue && height > maxSize.Value)
                        throw new ArgumentException($"maxSize = {maxSize} must be strictly greater than the requested size for the smaller edge size = {height}");

                    // Only one size was specified -- retain the aspect ratio.
                    if (iHeight < iWidth) {
                        h = height;
                        w = (int)Math.Floor(height * ((double)iWidth / (double)iHeight));
                    } else if (iWidth < iHeight) {
                        w = height;
                        h = (int)Math.Floor(height * ((double)iHeight / (double)iWidth));
                    } else {
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

            public static Tensor resized_crop(Tensor input, int top, int left, int height, int width, int newHeight, int newWidth, InterpolateMode mode = InterpolateMode.Nearest)
            {
                return resize(crop(input, top, left, height, width), newHeight, newWidth, mode);
            }

            public static Tensor rgb_to_grayscale(Tensor input, int num_output_channels = 1)
            {
                if (num_output_channels != 1 && num_output_channels != 3)
                    throw new ArgumentException("The number of output channels must be 1 or 3.");

                int cDim = (int)input.Dimensions - 3;
                if (input.shape[cDim] == 1)
                    // Already grayscale...
                    return input;

                var rgb = input.unbind(cDim);
                var img = (rgb[0] * 0.2989 + rgb[1] * 0.587 + rgb[2] * 0.114).unsqueeze(cDim);
                return num_output_channels == 3 ? img.expand(input.shape) : img;
            }

            public static Tensor rotate(Tensor img, float angle, GridSampleMode interpolation = GridSampleMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                var center_f = (0.0f, 0.0f);

                if (center.HasValue) {
                    var img_size = GetImageSize(img);
                    center_f = (1.0f * (center.Value.Item1 - img_size.Item1 * 0.5f), 1.0f * (center.Value.Item2 - img_size.Item2 * 0.5f));
                }

                var matrix = GetInverseAffineMatrix(center_f, -angle, (0.0f, 0.0f), 1.0f, (0.0f, 0.0f));

                return RotateImage(img, matrix, interpolation, expand, fill);
            }

            public static Tensor solarize(Tensor input, double threshold)
            {
                using (var inverted = invert(input))
                    return torch.where(input < threshold, input, inverted);
            }

            public static Tensor vflip(Tensor input) => input.flip(-2);

            private static Tensor RotateImage(Tensor img, IList<float> matrix, GridSampleMode interpolation, bool expand, IList<float> fill)
            {
                var (w, h) = GetImageSize(img);
                var (ow, oh) = expand ? ComputeOutputSize(matrix, w, h) : (w, h);
                var dtype = torch.is_floating_point(img) ? img.dtype : torch.float32;
                var theta = torch.tensor(matrix, dtype: dtype, device: img.device).reshape(1, 2, 3);
                var grid = GenerateAffineGrid(theta, w, h, ow, oh);

                return ApplyGridTransform(img, grid, interpolation, fill);
            }

            private static Tensor Blend(Tensor img1, Tensor img2, double ratio)
            {
                var bound = img1.IsIntegral() ? 255.0 : 1.0;
                return (img1 * ratio + img2 * (1.0 - ratio)).clamp(0, bound).to(img2.dtype);
            }

            private static Tensor BlurredDegenerateImage(Tensor input)
            {
                var device = input.device;
                var dtype = input.IsIntegral() ? ScalarType.Float32 : input.dtype;
                var kernel = Float32Tensor.ones(3, 3, device: device);
                kernel[1, 1] = Float32Tensor.from(5.0f);
                kernel /= kernel.sum();
                kernel = kernel.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

                var result_tmp = SqueezeIn(input, new ScalarType[] { ScalarType.Float32, ScalarType.Float64 }, out var needCast, out var needSqueeze, out var out_dtype);
                result_tmp = torch.nn.functional.conv2d(result_tmp, kernel, groups: result_tmp.shape[result_tmp.shape.Length - 3]);
                result_tmp = SqueezeOut(result_tmp, needCast, needSqueeze, out_dtype);

                var result = input.clone();
                result.index_put_(result_tmp, TensorIndex.Ellipsis, TensorIndex.Slice(1, -1), TensorIndex.Slice(1, -1));
                return result;
            }

            private static Tensor GetGaussianKernel1d(long size, float sigma)
            {
                var ksize_half = (size - 1) * 0.5f;
                var x = Float32Tensor.linspace(-ksize_half, ksize_half, size);
                var pdf = -(x / sigma).pow(2) * 0.5f;

                return pdf / pdf.sum();
            }

            private static Tensor GetGaussianKernel2d(IList<long> kernelSize, IList<float> sigma, ScalarType dtype, torch.Device device)
            {
                var kernel_X = GetGaussianKernel1d(kernelSize[0], sigma[0]).to(dtype, device)[TensorIndex.None, TensorIndex.Slice()];
                var kernel_Y = GetGaussianKernel1d(kernelSize[1], sigma[1]).to(dtype, device)[TensorIndex.Slice(), TensorIndex.None];
                return kernel_Y.mm(kernel_X);
            }

            private static (Tensor h, Tensor s, Tensor v) RGBtoHSV(Tensor img)
            {
                var RGB = img.unbind(-3);
                var r = RGB[0];
                var g = RGB[1];
                var b = RGB[2];

                var maxc = torch.max(img, dimension: -3).values;
                var minc = torch.min(img, dimension: -3).values;

                var eqc = maxc == minc;
                var cr = maxc - minc;
                var ones = torch.ones_like(maxc);

                var s = cr / torch.where(eqc, ones, maxc);
                var cr_divisor = torch.where(eqc, ones, cr);

                var rc = (maxc - r) / cr_divisor;
                var gc = (maxc - g) / cr_divisor;
                var bc = (maxc - b) / cr_divisor;

                var hr = (maxc == r) * (bc - gc);
                var hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc);
                var hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc);

                var h = (hr + hg + hb);
                h = torch.fmod((h / 6.0 + 1.0), 1.0);

                return (h, s, maxc);
            }

            private static Tensor HSVtoRGB(Tensor h, Tensor s, Tensor v)
            {
                var h6 = h * 6.0;
                var i = torch.floor(h6);
                var f = h6 - i;
                i = i.to(torch.int32) % 6;

                var p = torch.clamp((v * (1.0 - s)), 0.0, 1.0);
                var q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0);
                var t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0);

                var iunsq = i.unsqueeze(dimension: -3);
                var mask = iunsq == torch.arange(6, device: i.device).view(-1, 1, 1);

                var a1 = torch.stack(new Tensor[] { v, q, p, p, t, v }, dimension: -3);
                var a2 = torch.stack(new Tensor[] { t, v, v, q, p, p }, dimension: -3);
                var a3 = torch.stack(new Tensor[] { p, p, t, v, v, q }, dimension: -3);
                var a4 = torch.stack(new Tensor[] { a1, a2, a3 }, dimension: -4);

                var img = torch.einsum("...ijk,...xijk ->...xjk", mask.to(h.dtype), a4);

                //
                // Something really strange happens here -- the image comes out as 'NxCxHxW', but the
                // underlying memory is strided as if it's 'NxHxWxC'.
                //
                // So, as a workaround, we need to reshape it and permute the dimensions.
                //
                long[] NHWC = new long[] { img.shape[0], img.shape[2], img.shape[3], img.shape[1] };
                long[] permutation = new long[] { 0, 3, 1, 2 };

                return img.reshape(NHWC).permute(permutation);

            }

            private static Tensor ApplyGridTransform(Tensor img, Tensor grid, GridSampleMode mode, IList<float> fill = null)
            {
                img = SqueezeIn(img, new ScalarType[] { grid.dtype }, out var needCast, out var needSqueeze, out var out_dtype);

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
                    } else {
                        img = img * mask + (-mask + 1.0) * fill_img;
                    }
                }

                img = SqueezeOut(img, needCast, needSqueeze, out_dtype);
                return img;
            }

            private static Tensor GenerateAffineGrid(Tensor theta, long w, long h, long ow, long oh)
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

            private static (int, int) ComputeOutputSize(IList<float> matrix, long w, long h)
            {
                var pts = torch.tensor(new float[] { -0.5f * w, -0.5f * h, 1.0f, -0.5f * w, 0.5f * h, 1.0f, 0.5f * w, 0.5f * h, 1.0f, 0.5f * w, -0.5f * h, 1.0f }, 4, 3);
                var theta = torch.tensor(matrix, dtype: torch.float32).reshape(1, 2, 3);
                var new_pts = pts.view(1, 4, 3).bmm(theta.transpose(1, 2)).view(4, 2);

                Tensor min_vals = new_pts.min(dimension: 0).values;
                Tensor max_vals = new_pts.max(dimension: 0).values;

                var tol = 1f - 4;
                var cmax = torch.ceil((max_vals / tol).trunc_() * tol);
                var cmin = torch.floor((min_vals / tol).trunc_() * tol);

                var size = cmax - cmin;
                return (size[0].ToInt32(), size[1].ToInt32());
            }

            private static IList<float> GetInverseAffineMatrix((float, float) center, float angle, (float, float) translate, float scale, (float, float) shear)
            {
                // Convert to radians.
                var rot = angle * MathF.PI / 180.0f;
                var sx = shear.Item1 * MathF.PI / 180.0f;
                var sy = shear.Item2 * MathF.PI / 180.0f;

                var (cx, cy) = center;
                var (tx, ty) = translate;

                var a = MathF.Cos(rot - sy) / MathF.Cos(sy);
                var b = -MathF.Cos(rot - sy) * MathF.Tan(sx) / MathF.Cos(sy) - MathF.Sin(rot);
                var c = MathF.Sin(rot - sy) / MathF.Cos(sy);
                var d = -MathF.Sin(rot - sy) * MathF.Tan(sx) / MathF.Cos(sy) + MathF.Cos(rot);

                var matrix = (new float[] { d, -b, 0.0f, -c, a, 0.0f }).Select(x => x / scale).ToArray();

                matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty);
                matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty);

                matrix[2] += cx;
                matrix[5] += cy;

                return matrix;
            }

            private static (long, long) GetImageSize(Tensor img)
            {
                var hOffset = img.shape.Length - 2;
                return (img.shape[hOffset + 1], img.shape[hOffset]);
            }

            private static Tensor SqueezeIn(Tensor img, IList<ScalarType> req_dtypes, out bool needCast, out bool needSqueeze, out ScalarType dtype)
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

            private static Tensor SqueezeOut(Tensor img, bool needCast, bool needSqueeze, ScalarType dtype)
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

            private static long MaxValue(ScalarType dtype)
            {
                switch (dtype) {
                case ScalarType.Byte:
                    return byte.MaxValue;
                case ScalarType.Int8:
                    return sbyte.MaxValue;
                case ScalarType.Int16:
                    return short.MaxValue;
                case ScalarType.Int32:
                    return int.MaxValue;
                case ScalarType.Int64:
                    return long.MaxValue;
                }

                return 0L;
            }

        }
    }
}
