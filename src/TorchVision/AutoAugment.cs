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
        // Namespace hack to make use of the baseclass as a namespace for torchvision.autoaugment.AutoAugmentPolicy 
        public abstract class autoaugment
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

            /// <summary>AutoAugment policies learned on different datasets.
            /// Available policies are IMAGENET, CIFAR10 and SVHN.
            /// </summary>
            public enum AutoAugmentPolicy {
                ImageNet,
                CIFAR10,
                SVHN
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
         * https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#AutoAugment */
        internal class AutoAugment : autoaugment, ITransform
        {
            public AutoAugment(
                AutoAugmentPolicy policy = AutoAugmentPolicy.ImageNet,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                this.policy = policy;
                this.interpolation = interpolation;
                this.fill = fill;
                this.policies = this.get_policies(policy);
                this.generator = generator;
            }

            public Tensor call(Tensor img)
            {
                var (channels, height, width) = F.get_dimensions(img);

                var (transform_id, probs, signs) = this.get_params(this.policies.Count);

                var op_meta = this.augmentation_space(10, (height, width));

                var (policy1, policy2) = policies[transform_id];
                foreach (var (i, (op_name, p, magnitude_id)) in new[]{(0, policy1), (1, policy2)})
                {
                    if (probs[i].ToDouble() <= p) {
                        var (magnitudes, signed) = op_meta[op_name];
                        var magnitude = magnitude_id != null ? magnitudes[magnitude_id].ToDouble() : 0.0;
                        if (signed && signs[i].ToBoolean())
                            magnitude *= -1.0;
                        img = apply_op(img, op_name, magnitude, interpolation: this.interpolation, fill: this.fill);
                    }
                }

                return img;
            }

            private (int, Tensor, Tensor) get_params(int transform_num)
            {
                var policy_id = torch.randint(0, transform_num, size: 1, generator: this.generator).ToInt32();
                var probs = torch.rand(2, generator: this.generator);
                var signs = torch.randint(0, 2, size: 2, generator: this.generator);
                return (policy_id, probs, signs);
            }

            private Dictionary<opType, (Tensor, bool)> augmentation_space(int num_bins, (long height, long width) image_size)
            {
                return new Dictionary<opType, (Tensor, bool)> {
                        { opType.ShearX, (torch.linspace(0.0, 0.3, num_bins), true)},
                        { opType.ShearY, (torch.linspace(0.0, 0.3, num_bins), true)},
                        { opType.TranslateX, (torch.linspace(0.0, 150.0 / 331.0 * image_size.width, num_bins), true)},
                        { opType.TranslateY, (torch.linspace(0.0, 150.0 / 331.0 * image_size.height, num_bins), true)},
                        { opType.Rotate, (torch.linspace(0.0, 30.0, num_bins), true)},
                        { opType.Brightness, (torch.linspace(0.0, 0.9, num_bins), true)},
                        { opType.Color, (torch.linspace(0.0, 0.9, num_bins), true)},
                        { opType.Contrast, (torch.linspace(0.0, 0.9, num_bins), true)},
                        { opType.Sharpness, (torch.linspace(0.0, 0.9, num_bins), true)},
                        { opType.Posterize, (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().@int(), false)},
                        { opType.Solarize, (torch.linspace(255.0, 0.0, num_bins), false)},
                        { opType.AutoContrast, (torch.tensor(0.0), false)},
                        { opType.Equalize, (torch.tensor(0.0), false)},
                        { opType.Invert, (torch.tensor(0.0), false)}
                    };
            }

            private IList<((opType, double, int?), (opType, double, int?))> get_policies(AutoAugmentPolicy policy)
            {
                switch(policy)
                {
                    case AutoAugmentPolicy.ImageNet:
                        return new ((opType, double, int?), (opType, double, int?))[] {
                            ((opType.Posterize, 0.4, 8), (opType.Rotate, 0.6, 9)),
                            ((opType.Solarize, 0.6, 5), (opType.AutoContrast, 0.6, null)),
                            ((opType.Equalize, 0.8, null), (opType.Equalize, 0.6, null)),
                            ((opType.Posterize, 0.6, 7), (opType.Posterize, 0.6, 6)),
                            ((opType.Equalize, 0.4, null), (opType.Solarize, 0.2, 4)),
                            ((opType.Equalize, 0.4, null), (opType.Rotate, 0.8, 8)),
                            ((opType.Solarize, 0.6, 3), (opType.Equalize, 0.6, null)),
                            ((opType.Posterize, 0.8, 5), (opType.Equalize, 1.0, null)),
                            ((opType.Rotate, 0.2, 3), (opType.Solarize, 0.6, 8)),
                            ((opType.Equalize, 0.6, null), (opType.Posterize, 0.4, 6)),
                            ((opType.Rotate, 0.8, 8), (opType.Color, 0.4, 0)),
                            ((opType.Rotate, 0.4, 9), (opType.Equalize, 0.6, null)),
                            ((opType.Equalize, 0.0, null), (opType.Equalize, 0.8, null)),
                            ((opType.Invert, 0.6, null), (opType.Equalize, 1.0, null)),
                            ((opType.Color, 0.6, 4), (opType.Contrast, 1.0, 8)),
                            ((opType.Rotate, 0.8, 8), (opType.Color, 1.0, 2)),
                            ((opType.Color, 0.8, 8), (opType.Solarize, 0.8, 7)),
                            ((opType.Sharpness, 0.4, 7), (opType.Invert, 0.6, null)),
                            ((opType.ShearX, 0.6, 5), (opType.Equalize, 1.0, null)),
                            ((opType.Color, 0.4, 0), (opType.Equalize, 0.6, null)),
                            ((opType.Equalize, 0.4, null), (opType.Solarize, 0.2, 4)),
                            ((opType.Solarize, 0.6, 5), (opType.AutoContrast, 0.6, null)),
                            ((opType.Invert, 0.6, null), (opType.Equalize, 1.0, null)),
                            ((opType.Color, 0.6, 4), (opType.Contrast, 1.0, 8)),
                            ((opType.Equalize, 0.8, null), (opType.Equalize, 0.6, null))
                        };
                    case AutoAugmentPolicy.CIFAR10:
                        return new ((opType, double, int?), (opType, double, int?))[] {
                            ((opType.Invert, 0.1, null), (opType.Contrast, 0.2, 6)),
                            ((opType.Rotate, 0.7, 2), (opType.TranslateX, 0.3, 9)),
                            ((opType.Sharpness, 0.8, 1), (opType.Sharpness, 0.9, 3)),
                            ((opType.ShearY, 0.5, 8), (opType.TranslateY, 0.7, 9)),
                            ((opType.AutoContrast, 0.5, null), (opType.Equalize, 0.9, null)),
                            ((opType.ShearY, 0.2, 7), (opType.Posterize, 0.3, 7)),
                            ((opType.Color, 0.4, 3), (opType.Brightness, 0.6, 7)),
                            ((opType.Sharpness, 0.3, 9), (opType.Brightness, 0.7, 9)),
                            ((opType.Equalize, 0.6, null), (opType.Equalize, 0.5, null)),
                            ((opType.Contrast, 0.6, 7), (opType.Sharpness, 0.6, 5)),
                            ((opType.Color, 0.7, 7), (opType.TranslateX, 0.5, 8)),
                            ((opType.Equalize, 0.3, null), (opType.AutoContrast, 0.4, null)),
                            ((opType.TranslateY, 0.4, 3), (opType.Sharpness, 0.2, 6)),
                            ((opType.Brightness, 0.9, 6), (opType.Color, 0.2, 8)),
                            ((opType.Solarize, 0.5, 2), (opType.Invert, 0.0, null)),
                            ((opType.Equalize, 0.2, null), (opType.AutoContrast, 0.6, null)),
                            ((opType.Equalize, 0.2, null), (opType.Equalize, 0.6, null)),
                            ((opType.Color, 0.9, 9), (opType.Equalize, 0.6, null)),
                            ((opType.AutoContrast, 0.8, null), (opType.Solarize, 0.2, 8)),
                            ((opType.Brightness, 0.1, 3), (opType.Color, 0.7, 0)),
                            ((opType.Solarize, 0.4, 5), (opType.AutoContrast, 0.9, null)),
                            ((opType.TranslateY, 0.9, 9), (opType.TranslateY, 0.7, 9)),
                            ((opType.AutoContrast, 0.9, null), (opType.Solarize, 0.8, 3)),
                            ((opType.Equalize, 0.8, null), (opType.Invert, 0.1, null)),
                            ((opType.TranslateY, 0.7, 9), (opType.AutoContrast, 0.9, null))
                        };
                    case AutoAugmentPolicy.SVHN:
                        return new ((opType, double, int?), (opType, double, int?))[] {
                            ((opType.ShearX, 0.9, 4), (opType.Invert, 0.2, null)),
                            ((opType.ShearY, 0.9, 8), (opType.Invert, 0.7, null)),
                            ((opType.Equalize, 0.6, null), (opType.Solarize, 0.6, 6)),
                            ((opType.Invert, 0.9, null), (opType.Equalize, 0.6, null)),
                            ((opType.Equalize, 0.6, null), (opType.Rotate, 0.9, 3)),
                            ((opType.ShearX, 0.9, 4), (opType.AutoContrast, 0.8, null)),
                            ((opType.ShearY, 0.9, 8), (opType.Invert, 0.4, null)),
                            ((opType.ShearY, 0.9, 5), (opType.Solarize, 0.2, 6)),
                            ((opType.Invert, 0.9, null), (opType.AutoContrast, 0.8, null)),
                            ((opType.Equalize, 0.6, null), (opType.Rotate, 0.9, 3)),
                            ((opType.ShearX, 0.9, 4), (opType.Solarize, 0.3, 3)),
                            ((opType.ShearY, 0.8, 8), (opType.Invert, 0.7, null)),
                            ((opType.Equalize, 0.9, null), (opType.TranslateY, 0.6, 6)),
                            ((opType.Invert, 0.9, null), (opType.Equalize, 0.6, null)),
                            ((opType.Contrast, 0.3, 3), (opType.Rotate, 0.8, 4)),
                            ((opType.Invert, 0.8, null), (opType.TranslateY, 0.0, 2)),
                            ((opType.ShearY, 0.7, 6), (opType.Solarize, 0.4, 8)),
                            ((opType.Invert, 0.6, null), (opType.Rotate, 0.8, 4)),
                            ((opType.ShearY, 0.3, 7), (opType.TranslateX, 0.9, 3)),
                            ((opType.ShearX, 0.1, 6), (opType.Invert, 0.6, null)),
                            ((opType.Solarize, 0.7, 2), (opType.TranslateY, 0.6, 7)),
                            ((opType.ShearY, 0.8, 4), (opType.Invert, 0.8, null)),
                            ((opType.ShearX, 0.7, 9), (opType.TranslateY, 0.8, 3)),
                            ((opType.ShearY, 0.8, 5), (opType.AutoContrast, 0.7, null)),
                            ((opType.ShearX, 0.7, 2), (opType.Invert, 0.1, null))
                        };
                    default:
                        throw new ArgumentException($"The provided policy {policy} is not recognized.");
                }
            }

            private readonly AutoAugmentPolicy policy;
            private readonly InterpolationMode interpolation;
            private readonly IList<float>? fill;
            private readonly IList<((opType, double, int?), (opType, double, int?))> policies;
            private readonly Generator? generator;
        }

        /* Original implementation from:
         * https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#RandAugment */
        internal class RandAugment : autoaugment, ITransform
        {
            public RandAugment(
                int num_ops = 2,
                int magnitude = 9,
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                this.num_ops = num_ops;
                this.magnitude = magnitude;
                this.num_magnitude_bins = num_magnitude_bins;
                this.interpolation = interpolation;
                this.fill = fill;
                this.generator = generator;
            }

            public Tensor call(Tensor img)
            {
                using var _ = torch.NewDisposeScope();
                var (_, height, width) = F.get_dimensions(img);
                var op_meta = augmentation_space(num_magnitude_bins, (height, width));
                for (int i = 0; i < num_ops; ++i) {
                    var op_index = torch.randint(0, op_meta.Count, size: 1, generator: this.generator).ToInt32();
                    var op_name = op_meta.Keys.ElementAt(op_index);
                    var (magnitudes, signed) = op_meta[op_name];
                    var magnitude = magnitudes.Dimensions > 0 ? magnitudes[this.magnitude].ToDouble() : 0.0;

                    if (signed && torch.randint(0, 2, size: 1, generator: this.generator).ToBoolean())
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
                    { opType.TranslateX, (torch.linspace(0.0, 150.0 / 331.0 * image_size.width, num_bins), true) },
                    { opType.TranslateY, (torch.linspace(0.0, 150.0 / 331.0 * image_size.height, num_bins), true) },
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
            private readonly Generator? generator;
        }

        /* Original implementation from:
         * https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide */
        internal class TrivialAugmentWide : autoaugment, ITransform
        {
            public TrivialAugmentWide(
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                this.num_magnitude_bins = num_magnitude_bins;
                this.interpolation = interpolation;
                this.fill = fill;
                this.generator = generator;
            }

            public Tensor call(Tensor img)
            {
                var op_meta = this.augmentation_space(this.num_magnitude_bins);
                var op_index = torch.randint(0, op_meta.Count(), size: 1, generator: this.generator).ToInt32();
                var op_name = op_meta.Keys.ElementAt(op_index);
                var (magnitudes, signed) = op_meta[op_name];
                var magnitude = magnitudes.ndim > 0 ? magnitudes[torch.randint(0, num_magnitude_bins, size: 1, generator: this.generator).ToInt32()] .ToDouble() : 0.0;

                return apply_op(img, op_name, magnitude, interpolation: this.interpolation, fill: this.fill);
            }

            private Dictionary<opType, (Tensor, bool)> augmentation_space(int num_bins)
            {
                return new Dictionary<opType, (Tensor, bool)> {
                    { opType.Identity, (torch.tensor(0.0), false) },
                    { opType.ShearX, (torch.linspace(0.0, 0.99, num_bins), true) },
                    { opType.ShearY, (torch.linspace(0.0, 0.99, num_bins), true) },
                    { opType.TranslateX, (torch.linspace(0.0, 32.0, num_bins), true) },
                    { opType.TranslateY, (torch.linspace(0.0, 32.0, num_bins), true) },
                    { opType.Rotate, (torch.linspace(0.0, 135.0, num_bins), true) },
                    { opType.Brightness, (torch.linspace(0.0, 0.99, num_bins), true) },
                    { opType.Color, (torch.linspace(0.0, 0.99, num_bins), true) },
                    { opType.Contrast, (torch.linspace(0.0, 0.99, num_bins), true) },
                    { opType.Sharpness, (torch.linspace(0.0, 0.99, num_bins), true) },
                    { opType.Posterize, (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().@int(), false) },
                    { opType.Solarize, (torch.linspace(255.0, 0.0, num_bins), false) },
                    { opType.AutoContrast, (torch.tensor(0.0), false) },
                    { opType.Equalize, (torch.tensor(0.0), false) },
                };
            }

            public readonly int num_magnitude_bins;
            public readonly InterpolationMode interpolation;
            public readonly IList<float>? fill;
            public readonly Generator? generator;

        }

        /* Original implementation from:
         * https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#AugMix */
        internal class AugMix :autoaugment, ITransform
        {
            public AugMix(
                int severity = 3,
                int mixture_width = 3,
                int chain_depth = -1,
                double alpha = 1.0,
                bool all_ops = true,
                InterpolationMode interpolation = InterpolationMode.Bilinear,
                IList<float>? fill = null,
                Generator? generator = null)
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
                this.generator = generator;
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
                    torch.tensor(new[]{this.alpha, this.alpha}, device: batch.device).expand(batch_dims[0], -1),
                    generator: this.generator
                    );

                var combined_weights = torch._sample_dirichlet(
                    torch.tensor(Enumerable.Repeat(this.alpha, this.mixture_width).ToArray(), device: batch.device).expand(batch_dims[0], -1),
                    generator: this.generator
                    ) * m.select(1, 1).view(new[]{batch_dims[0], -1});

                var mix = m.select(1, 0).view(batch_dims) * batch;

                for(int i = 0; i < this.mixture_width; ++i)
                {
                    var aug = batch;
                    var depth = this.chain_depth > 0 ? this.chain_depth : torch.randint(low: 1, high: 4, size: 1, generator: this.generator).ToInt32();
                    for(int d = 0; d < depth; ++d)
                    {
                        var op_index = torch.randint(0, op_meta.Count, size: 1, generator: this.generator).ToInt32();
                        var op_name = op_meta.Keys.ElementAt(op_index);
                        var (magnitudes, signed) = op_meta[op_name];

                        var magnitude = magnitudes.ndim > 0 ? magnitudes[torch.randint(0, this.severity, size: 1, generator: this.generator).ToInt32()].ToDouble() : 0.0;
                        if (signed && torch.randint(0, 2, size: 1, generator: this.generator).ToBoolean())
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
            private readonly Generator? generator;
        }

        public static partial class transforms
        {
            /// <summary>
            /// AutoAugment data augmentation method based on
            /// "AutoAugment: Learning Augmentation Strategies from Data"
            /// https://arxiv.org/pdf/1805.09501.pdf
            /// The image is expected to be a torch Tensor, it should be of type torch.uint8, and it is expected
            /// to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="policy"> Desired policy enum defined by
            /// torchvision.transforms.autoaugment.AutoAugmentPolicy. Default: AutoAugmentPolicy.ImageNet</param>
            /// <param name="interpolation">Desired interpolation enum defined by
            /// torchvision.transforms.InterpolationMode. Default: InterpolationMode.Nearest.</param>
            /// <param name="fill">Pixel fill value for the area outside the transformed
            /// image. If given a number, the value is used for all bands respectively. Default: null</param>
            /// <param name="generator">The generator used for random values. Default: null</param>
            static public ITransform AutoAugment(
                autoaugment.AutoAugmentPolicy policy = autoaugment.AutoAugmentPolicy.ImageNet,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                return new AutoAugment(policy, interpolation, fill, generator);
            }

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
            /// <param name="generator">The generator used for random values. Default: null</param>
            static public ITransform RandAugment(
                int num_ops = 2,
                int magnitude = 9,
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                return new RandAugment(num_ops, magnitude, num_magnitude_bins, interpolation, fill, generator);
            }

            /// <summary>
            /// Dataset-independent data-augmentation with TrivialAugment Wide, as described in
            /// "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation"
            /// https://arxiv.org/abs/2103.10158
            /// The image is expected to be a torch Tensor, it should be of type torch.uint8, and it is expected
            /// to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="num_magnitude_bins">The number of different magnitude values. Default: 31</param>
            /// <param name="interpolation">Desired interpolation enum defined by
            /// torchvision.transforms.InterpolationMode. Default: InterpolationMode.Nearest.</param>
            /// <param name="fill">Pixel fill value for the area outside the transformed
            /// image. If given a number, the value is used for all bands respectively. Default: null</param>
            /// <param name="generator">The generator used for random values. Default: null</param>
            static public ITransform TrivialAugmentWide(
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                return new TrivialAugmentWide(num_magnitude_bins, interpolation, fill, generator);
            }

            /// <summary>
            /// AugMix data augmentation method based on
            /// "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty"
            /// https://arxiv.org/abs/1912.02781
            /// The image is expected to be a torch Tensor, it should be of type torch.uint8, and it is expected
            /// to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="severity">The severity of base augmentation operators. Default: 3</param>
            /// <param name="mixture_width">The number of augmentation chains. Default: 3</param>
            /// <param name="chain_depth">The depth of augmentation chains. A negative value denotes stochastic
            /// depth sampled from the interval [1, 3]. Default: -1</param>
            /// <param name="alpha">The hyperparameter for the probability distributions. Default 1.0</param>
            /// <param name="all_ops">Use all operations (including brightness, contrast, color and sharpness). Default: true</param>
            /// <param name="interpolation">Desired interpolation enum defined by
            /// torchvision.transforms.InterpolationMode. Default: InterpolationMode.Bilinear.</param>
            /// <param name="fill">Pixel fill value for the area outside the transformed
            /// image. If given a number, the value is used for all bands respectively. Default: null</param>
            /// <param name="generator">The generator used for random values. Default: null</param>
            static public ITransform AugMix(
                int severity = 3,
                int mixture_width = 3,
                int chain_depth = -1,
                double alpha = 1.0,
                bool all_ops = true,
                InterpolationMode interpolation = InterpolationMode.Bilinear,
                IList<float>? fill = null,
                Generator? generator = null)
            {
                return new AugMix(severity, mixture_width, chain_depth, alpha, all_ops, interpolation, fill, generator);
            }
        }
    }
}
