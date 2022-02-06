using System;
using System.Collections.Generic;
using System.IO;

using static TorchSharp.torch;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats;

namespace TorchSharp.torchvision
{
    public static partial class io
    {
        public class ImagerSharp : IImager
        {
            public ImageFormat DetectFormat(byte[] image)
            {
                var format = Image.DetectFormat(image);
                if (format is PngFormat) {
                    return ImageFormat.PNG;
                } else if (format is JpegFormat) {
                    return ImageFormat.JPEG;
                } else {
                    return ImageFormat.UNKNOWN;
                }
            }

            public Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
               return DecodeImage(image, DetectFormat(image), mode);
            }
            public Tensor DecodeImage(byte[] bytes, ImageFormat _, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
                switch (mode) {
                case ImageReadMode.UNCHANGED:
                    var image = Image.Load<Rgba32>(bytes);
                    using (var dataTensor = torch.zeros(4, image.Height, image.Width)) {
                        for (var i = 0; i < image.Height; i++) {
                            var span = image.GetPixelRowSpan(i);
                            for (var j = 0; j < span.Length; j++) {
                                var pixel = span[j];

                                var idx = image.Height * image.Width;

                                dataTensor.index_put_(pixel.R, TensorIndex.Single(idx));
                                dataTensor.index_put_(pixel.G, TensorIndex.Single(idx + 1));
                                dataTensor.index_put_(pixel.B, TensorIndex.Single(idx + 2));
                                dataTensor.index_put_(pixel.A, TensorIndex.Single(idx + 3));
                            }
                        }

                        return dataTensor.reshape(new long[] { 4, (long)image.Height, (long)image.Width });
                    }
                default: throw new NotImplementedException();
                }
            }

            public byte[] EncodeImage(Tensor image, ImageFormat format)
            {
                return null;
            }
        }
    }
}
