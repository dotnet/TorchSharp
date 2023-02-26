// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class ColorJitter : ITransform
        {
            internal ColorJitter((float, float) brightness, (float, float) contrast, (float, float) saturation, (float, float) hue)
            {
                this.brightness = brightness;
                this.contrast = contrast;
                this.saturation = saturation;
                this.hue = hue;
            }

            public Tensor call(Tensor image)
            {
                var randoms = torch.rand(4, ScalarType.Float32).data<float>().ToArray();
                var b = Adjust(randoms[0], brightness.Item1, brightness.Item2);
                var c = Adjust(randoms[1], contrast.Item1, contrast.Item2);
                var s = Adjust(randoms[2], saturation.Item1, saturation.Item2);
                var h = Adjust(randoms[3], hue.Item1, hue.Item2);

                var transform = torchvision.transforms.Compose(
                    transforms.AdjustBrightness(b),
                    transforms.AdjustContrast(c),
                    transforms.AdjustSaturation(s),
                    transforms.AdjustHue(h)
                    );
                return transform.call(image);
            }

            internal static float Adjust(float input, float min, float max)
            {
                return input * (max - min) + min;
            }

            private (float, float) brightness;
            private (float, float) contrast;
            private (float, float) saturation;
            private (float, float) hue;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Randomly change the brightness, contrast, saturation and hue of an image.
            /// The image is expected to have […, 3, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="brightness">How much to jitter brightness, [min,max]. Should be non-negative.</param>
            /// <param name="contrast">How much to jitter contrast, [min,max]. Should be non-negative.</param>
            /// <param name="saturation">How much to jitter saturation, [min,max]. Should be non-negative. </param>
            /// <param name="hue">How much to jitter hue. Should lie between -0.5 and 0.5, with min less than max.</param>
            /// <returns></returns>
            /// <remarks>The image will not be cropped outside its boundaries.</remarks>
            static public ITransform ColorJitter((float, float) brightness, (float, float) contrast, (float, float) saturation, (float, float) hue)
            {
                return new ColorJitter(brightness, contrast, saturation, hue);
            }

            /// <summary>
            /// Randomly change the brightness, contrast, saturation and hue of an image.
            /// The image is expected to have […, 3, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="brightness">How much to jitter brightness. Should be non-negative.
            /// The brightness_factor used is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]</param>
            /// <param name="contrast">How much to jitter contrast. Should be non-negative.
            /// The contrast_factor used is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]</param>
            /// <param name="saturation">How much to jitter saturation. Should be non-negative.
            /// The saturation_factor used is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] </param>
            /// <param name="hue">How much to jitter hue. Should be between 0 and 0.5.</param>
            /// <returns></returns>
            /// <remarks>The image will not be cropped outside its boundaries.</remarks>
            static public ITransform ColorJitter(float brightness = 0, float contrast = 0, float saturation = 0, float hue = 0)
            {
                return new ColorJitter(
                    (MathF.Max(0, 1 - brightness), 1 + brightness),
                    (MathF.Max(0, 1 - contrast), 1 + contrast),
                    (MathF.Max(0, 1 - saturation), 1 + saturation),
                    (-hue, hue));
            }
        }
    }
}