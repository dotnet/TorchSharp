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
        private static Tensor apply_op(
                                Tensor img,
                                string op_name,
                                float magnitude,
                                InterpolationMode interpolation,
                                IList<float>? fill)
        {
            if (op_name == "ShearX") {
                img = F.affine(
                    img,
                    angle: 0.0f,
                    translate: new[] { 0, 0 },
                    scale: 1.0f,
                    shear: new[] { (float)((180.0f / Math.PI) * Math.Atan(magnitude)), 0.0f },
                    interpolation: interpolation,
                    fill: fill?.FirstOrDefault());
            } else if (op_name == "ShearY") {
                img = F.affine(
                    img,
                    angle: 0.0f,
                    translate: new[] { 0, 0 },
                    scale: 1.0f,
                    shear: new[] { 0.0f, (float)((180.0f / Math.PI) * Math.Atan(magnitude)) },
                    interpolation: interpolation,
                    fill: fill?.FirstOrDefault());
            } else if (op_name == "TranslateX") {
                img = F.affine(
                    img,
                    angle: 0.0f,
                    translate: new[] { (int)(magnitude), 0 },
                    scale: 1.0f,
                    interpolation: interpolation,
                    shear: new[] { 0.0f, 0.0f },
                    fill: fill?.FirstOrDefault());
            } else if (op_name == "TranslateY") {
                img = F.affine(
                    img,
                    angle: 0.0f,
                    translate: new[] { 0, (int)(magnitude) },
                    scale: 1.0f,
                    interpolation: interpolation,
                    shear: new[] { 0.0f, 0.0f },
                    fill: fill?.FirstOrDefault());
            } else if (op_name == "Rotate") {
                img = F.rotate(img, magnitude, interpolation, fill: fill);
            } else if (op_name == "Brightness") {
                img = F.adjust_brightness(img, 1.0 + magnitude);
            } else if (op_name == "Color") {
                img = F.adjust_saturation(img, 1.0 + magnitude);
            } else if (op_name == "Contrast") {
                img = F.adjust_contrast(img, 1.0 + magnitude);
            } else if (op_name == "Sharpness") {
                img = F.adjust_sharpness(img, 1.0 + magnitude);
            } else if (op_name == "Posterize") {
                img = F.posterize(img, (int)magnitude);
            } else if (op_name == "Solarize") {
                img = F.solarize(img, magnitude);
            } else if (op_name == "AutoContrast") {
                img = F.autocontrast(img);
            } else if (op_name == "Equalize") {
                img = F.equalize(img);
            } else if (op_name == "Invert") {
                img = F.invert(img);
            } else if (op_name == "Identity") {
                // Pass
            } else {
                throw new ArgumentException($"The provided operator {op_name} is not recognized.");
            }

            return img;
        }


        internal class RandAugment : ITransform
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
                var fill = this.fill;
                var cDim = img.Dimensions - 3;
                var height = img.shape[cDim + 1];
                var width = img.shape[cDim + 2];
                var op_meta = augmentation_space(num_magnitude_bins, (height, width));
                for (int i = 0; i < num_ops; ++i) {
                    var op_index = torch.randint(op_meta.Count, new[] { 1 }).ToInt32();
                    var op_name = op_meta.Keys.ToList()[op_index];
                    var (magnitudes, signed) = op_meta[op_name];
                    var magnitude = magnitudes.Dimensions > 0 ? magnitudes[this.magnitude] : 0.0f;

                    if (signed && torch.randint(2, new[] { 1 }).ToBoolean())
                        magnitude *= -1.0;

                    img = apply_op(img, op_name, magnitude.ToSingle(), interpolation, fill);
                }
                return img;
            }

            private Dictionary<string, (Tensor, bool)> augmentation_space(int num_bins, (long height, long width) image_size)
            {
                return new Dictionary<string, (Tensor, bool)> {
                    { "Identity", (torch.tensor(0.0), false) },
                    { "ShearX", (torch.linspace(0.0, 0.3, num_bins), true) },
                    { "ShearY", (torch.linspace(0.0, 0.3, num_bins), true) },
                    { "TranslateX", (torch.linspace(0.0, 150.0 / 331.0 * image_size.height, num_bins), true) },
                    { "TranslateY", (torch.linspace(0.0, 150.0 / 331.0 * image_size.width, num_bins), true) },
                    { "Rotate", (torch.linspace(0.0, 30.0, num_bins), true) },
                    { "Brightness", (torch.linspace(0.0, 0.9, num_bins), true) },
                    { "Color", (torch.linspace(0.0, 0.9, num_bins), true) },
                    { "Contrast", (torch.linspace(0.0, 0.9, num_bins), true) },
                    { "Sharpness", (torch.linspace(0.0, 0.9, num_bins), true) },
                    { "Posterize", (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().@int(), false)},
                    { "Solarize", (torch.linspace(255.0, 0.0, num_bins), false) },
                    { "AutoContrast", (torch.tensor(0.0), false) },
                    { "Equalize", (torch.tensor(0.0), false) }
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
