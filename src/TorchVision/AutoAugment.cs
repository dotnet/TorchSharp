using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using F = TorchSharp.torchvision.transforms.functional;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        internal abstract class AutoAugmentBase
        {
            protected enum opType {
                ShearX,
                ShearY,
                TranslateX,
                TranslateY,
                Rotate,
                Brightness,
                Color,
                Contrast,
                Sharpness,
                Posterize,
                Solarize,
                AutoContrast,
                Equalize,
                Invert,
                Identity
            }

            protected static Tensor apply_op(
                                    Tensor img,
                                    opType op_name,
                                    double magnitude,
                                    InterpolationMode interpolation,
                                    IList<float>? fill)
            {
                switch(op_name) {
                    case opType.ShearX:
                        return F.affine(
                            img,
                            angle: 0.0f,
                            translate: new[] { 0, 0 },
                            scale: 1.0f,
                            shear: new[] { (float)((180.0 / Math.PI) * Math.Atan(magnitude)), 0.0f },
                            interpolation: interpolation,
                            fill: fill?.FirstOrDefault());
                    case opType.ShearY:
                        return F.affine(
                            img,
                            angle: 0.0f,
                            translate: new[] { 0, 0 },
                            scale: 1.0f,
                            shear: new[] { 0.0f, (float)((180.0 / Math.PI) * Math.Atan(magnitude)) },
                            interpolation: interpolation,
                            fill: fill?.FirstOrDefault());
                    case opType.TranslateX:
                        return F.affine(
                            img,
                            angle: 0.0f,
                            translate: new[] { (int)(magnitude), 0 },
                            scale: 1.0f,
                            interpolation: interpolation,
                            shear: new[] { 0.0f, 0.0f },
                            fill: fill?.FirstOrDefault());
                    case opType.TranslateY:
                        return F.affine(
                            img,
                            angle: 0.0f,
                            translate: new[] { 0, (int)(magnitude) },
                            scale: 1.0f,
                            interpolation: interpolation,
                            shear: new[] { 0.0f, 0.0f },
                            fill: fill?.FirstOrDefault());
                    case opType.Rotate:
                        return F.rotate(img, (float)magnitude, interpolation, fill: fill);
                    case opType.Brightness:
                        return F.adjust_brightness(img, 1.0 + magnitude);
                    case opType.Color:
                        return F.adjust_saturation(img, 1.0 + magnitude);
                    case opType.Contrast:
                        return F.adjust_contrast(img, 1.0 + magnitude);
                    case opType.Sharpness:
                        return F.adjust_sharpness(img, 1.0 + magnitude);
                    case opType.Posterize:
                        return F.posterize(img, (int)magnitude);
                    case opType.Solarize:
                        return F.solarize(img, magnitude);
                    case opType.AutoContrast:
                        return F.autocontrast(img);
                    case opType.Equalize:
                        return F.equalize(img);
                    case opType.Invert:
                        return F.invert(img);
                    case opType.Identity:
                        return img; // Pass
                    default:
                        throw new ArgumentException($"The provided operator {op_name} is not recognized.");
                }
            }
        }

        /* Original implementation from:
         * https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#RandAugment */
        internal class RandAugment : AutoAugmentBase, ITransform
        {
            public RandAugment(
                int num_ops = 2,
                int magnitude = 9,
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null)
            {
                this.num_ops = num_ops;
                this.magnitude = magnitude;
                this.num_magnitude_bins = num_magnitude_bins;
                this.interpolation = interpolation;
                this.fill = fill;
            }

            public Tensor call(Tensor img)
            {
                using var _ = torch.NewDisposeScope();
                var (_, height, width) = F.get_dimensions(img);
                var op_meta = augmentation_space(num_magnitude_bins, (height, width));
                for (int i = 0; i < num_ops; ++i) {
                    var op_index = torch.randint(0, op_meta.Count, 1).ToInt32();
                    var op_name = op_meta.Keys.ElementAt(op_index);
                    var (magnitudes, signed) = op_meta[op_name];
                    var magnitude = magnitudes.Dimensions > 0 ? magnitudes[this.magnitude].ToDouble() : 0.0;

                    if (signed && torch.randint(0, 2, 1).ToBoolean())
                        magnitude *= -1.0;

                    img = apply_op(img, op_name, magnitude, interpolation, this.fill);
                }
                return img.MoveToOuterDisposeScope();
            }

            private Dictionary<opType, (Tensor, bool)> augmentation_space(int num_bins, (long height, long width) image_size)
            {
                return new Dictionary<opType, (Tensor, bool)> {
                    { opType.Identity, (torch.tensor(0.0), false) },
                    { opType.ShearX, (torch.linspace(0.0, 0.3, num_bins), true) },
                    { opType.ShearY, (torch.linspace(0.0, 0.3, num_bins), true) },
                    { opType.TranslateX, (torch.linspace(0.0, 150.0 / 331.0 * image_size.height, num_bins), true) },
                    { opType.TranslateY, (torch.linspace(0.0, 150.0 / 331.0 * image_size.width, num_bins), true) },
                    { opType.Rotate, (torch.linspace(0.0, 30.0, num_bins), true) },
                    { opType.Brightness, (torch.linspace(0.0, 0.9, num_bins), true) },
                    { opType.Color, (torch.linspace(0.0, 0.9, num_bins), true) },
                    { opType.Contrast, (torch.linspace(0.0, 0.9, num_bins), true) },
                    { opType.Sharpness, (torch.linspace(0.0, 0.9, num_bins), true) },
                    { opType.Posterize, (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().@int(), false)},
                    { opType.Solarize, (torch.linspace(255.0, 0.0, num_bins), false) },
                    { opType.AutoContrast, (torch.tensor(0.0), false) },
                    { opType.Equalize, (torch.tensor(0.0), false) }
                };
            }

            private readonly int num_ops;
            private readonly int magnitude;
            private readonly int num_magnitude_bins;
            private readonly InterpolationMode interpolation;
            private readonly IList<float>? fill;
        }

        internal class AugMix :AutoAugmentBase, ITransform
        {
            public AugMix(
                int severity = 3,
                int mixture_width = 3,
                int chain_depth = -1,
                double alpha = 1.0,
                bool all_ops = true,
                InterpolationMode interpolation = InterpolationMode.Bilinear,
                IList<float>? fill = null)
            {
                if (severity < 1 || severity > ParameterMax)
                    throw new ArgumentException($"The severity must be between [1, {ParameterMax}]. Got {severity} instead.");

                this.severity = severity;
                this.mixture_width = mixture_width;
                this.chain_depth = chain_depth;
                this.alpha = alpha;
                this.all_ops = all_ops;
                this.interpolation = interpolation;
                this.fill = fill;
            }

            private Dictionary<opType, (Tensor, bool)> augmentation_space(int num_bins, (long height, long width) image_size)
            {
                var s = new Dictionary<opType, (Tensor, bool)> {
                    { opType.ShearX, (torch.linspace(0.0, 0.3, num_bins), true) },
                    { opType.ShearY, (torch.linspace(0.0, 0.3, num_bins), true) },
                    { opType.TranslateX, (torch.linspace(0.0, image_size.width / 3.0, num_bins), true) },
                    { opType.TranslateY, (torch.linspace(0.0, image_size.height / 3.0, num_bins), true) },
                    { opType.Rotate, (torch.linspace(0.0, 30.0, num_bins), true) },
                    { opType.Posterize, (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().@int(), false) },
                    { opType.Solarize, (torch.linspace(255.0, 0.0, num_bins), false) },
                    { opType.AutoContrast, (torch.tensor(0.0), false) },
                    { opType.Equalize, (torch.tensor(0.0), false) }
                };
                if (this.all_ops)
                {
                    s.Add(opType.Brightness, (torch.linspace(0.0, 0.9, num_bins), true));
                    s.Add(opType.Color, (torch.linspace(0.0, 0.9, num_bins), true));
                    s.Add(opType.Contrast, (torch.linspace(0.0, 0.9, num_bins), true));
                    s.Add(opType.Sharpness, (torch.linspace(0.0, 0.9, num_bins), true));
                }
                return s;
            }

            public Tensor call(Tensor img)
            {
                using var _ = torch.NewDisposeScope();

                var (channels, height, width) = F.get_dimensions(img);
                var op_meta = augmentation_space(ParameterMax, (height, width));

                var orig_dims = img.shape;
                var batch_view = Enumerable.Repeat(1L, (int)Math.Max(4 - img.ndim, 0)).Concat(orig_dims).ToArray();
                var batch = img.view(batch_view);
                var batch_dims = new[]{batch.size(0)}.Concat(Enumerable.Repeat(1L, (int)batch.ndim - 1)).ToArray();

                var m = torch._sample_dirichlet(
                    torch.tensor(new[]{this.alpha, this.alpha}, device: batch.device).expand(batch_dims[0], -1)
                    );

                var combined_weights = torch._sample_dirichlet(
                    torch.tensor(Enumerable.Repeat(this.alpha, this.mixture_width).ToArray(), device: batch.device).expand(batch_dims[0], -1)
                    ) * m.select(1, 1).view(new[]{batch_dims[0], -1});

                var mix = m.select(1, 0).view(batch_dims) * batch;

                for(int i = 0; i < this.mixture_width; ++i)
                {
                    var aug = batch;
                    var depth = this.chain_depth > 0 ? this.chain_depth : torch.randint(low: 1, high: 4, size: 1).ToInt32();
                    for(int d = 0; d < depth; ++d)
                    {
                        var op_index = torch.randint(op_meta.Count, size: 1).ToInt32();
                        var op_name = op_meta.Keys.ElementAt(op_index);
                        var (magnitudes, signed) = op_meta[op_name];

                        var magnitude = magnitudes.ndim > 0 ? magnitudes[torch.randint(this.severity, size: 1).ToInt32()].ToDouble() : 0.0;
                        if (signed && torch.randint(0, 2, 1).ToBoolean())
                            magnitude *= -1.0;

                        aug = apply_op(aug, op_name, magnitude, interpolation: this.interpolation, fill: this.fill);
                    }
                    mix.add_(combined_weights.select(1, i).view(batch_dims) * aug);
                }

                mix = mix.view(orig_dims).to(img.dtype);

                return mix.MoveToOuterDisposeScope();
            }

            private const int ParameterMax = 10;
            private readonly int severity;
            private readonly int mixture_width;
            private readonly int chain_depth;
            private readonly double alpha;
            private readonly bool all_ops;
            private readonly InterpolationMode interpolation;
            private readonly IList<float>? fill;
        }

        public static partial class transforms
        {
            /// <summary>
            /// RandAugment data augmentation method based on
            /// "RandAugment: Practical automated data augmentation with a reduced search space"
            /// https://arxiv.org/abs/1909.13719
            /// The image is expected to be a torch Tensor, it should be of type torch.uint8, and it is expected
            /// to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="num_ops">Number of augmentation transformations to apply sequentially. Default: 2</param>
            /// <param name="magnitude">Magnitude for all the transformations. Default: 9</param>
            /// <param name="num_magnitude_bins">The number of different magnitude values. Default: 31</param>
            /// <param name="interpolation">Desired interpolation enum defined by
            /// torchvision.transforms.InterpolationMode. Default: InterpolationMode.NEAREST.</param>
            /// <param name="fill">Pixel fill value for the area outside the transformed
            /// image. If given a number, the value is used for all bands respectively. Default: null</param>
            static public ITransform RandAugment(
                int num_ops = 2,
                int magnitude = 9,
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null)
            {
                return new RandAugment(num_ops, magnitude, num_magnitude_bins, interpolation, fill);
            }

            static public ITransform AugMix(
                int severity = 3,
                int mixture_width = 3,
                int chain_depth = -1,
                double alpha = 1.0,
                bool all_ops = true,
                InterpolationMode interpolation = InterpolationMode.Bilinear,
                IList<float>? fill = null)
            {
                return new AugMix(severity, mixture_width, chain_depth, alpha, all_ops, interpolation, fill);
            }
        }
    }
}
