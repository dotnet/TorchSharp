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
                var cDim = img.Dimensions - 3;
                var height = img.shape[cDim + 1];
                var width = img.shape[cDim + 2];
                var op_meta = augmentation_space(num_magnitude_bins, (height, width));
                for (int i = 0; i < num_ops; ++i) {
                    var op_index = torch.randint(op_meta.Count, new[] { 1 }).ToInt32();
                    var op_name = op_meta.Keys.ToList()[op_index];
                    var (magnitudes, signed) = op_meta[op_name];
                    var magnitude = magnitudes.Dimensions > 0 ? magnitudes[this.magnitude].ToDouble() : 0.0;

                    if (signed && torch.randint(2, new[] { 1 }).ToBoolean())
                        magnitude *= -1.0;

                    img = apply_op(img, op_name, magnitude, interpolation, this.fill);
                }
                return img;
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

        public static partial class transforms
        {
            static public ITransform RandAugment(
                int num_ops = 2,
                int magnitude = 9,
                int num_magnitude_bins = 31,
                InterpolationMode interpolation = InterpolationMode.Nearest,
                IList<float>? fill = null)
            {
                return new RandAugment(num_ops, magnitude, num_magnitude_bins, interpolation, fill);
            }
        }
    }
}
