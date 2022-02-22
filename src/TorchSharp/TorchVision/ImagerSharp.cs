using System;
using System.IO;
using System.Runtime.CompilerServices;

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

            Tensor ToTensor<TPixel>(byte[] bytes) where TPixel : unmanaged, IPixel<TPixel>
            {
                var image = Image.Load<TPixel>(bytes);

                var channels = Unsafe.SizeOf<TPixel>();

                byte[] imageBytes = new byte[image.Width * image.Height * channels];

                image.CopyPixelDataTo(imageBytes);

                return tensor(imageBytes, new long[] { image.Width, image.Height, channels}).permute(2, 0, 1);
            }

            byte[] FromTensor<TPixel>(Tensor t, ImageFormat format) where TPixel : unmanaged, IPixel<TPixel>
            {
                var shape = t.shape;
                var tt = t.reshape(new long[] { shape[0] * shape[1] * shape[2] });
                var image = Image.LoadPixelData<TPixel>(tt.bytes.ToArray(), (int)shape[0], (int)shape[1]);
                IImageEncoder encoder = format switch {
                   ImageFormat.PNG => new PngEncoder(),
                   ImageFormat.JPEG => new JpegEncoder(),
                   _ => throw new ArgumentException("Cannot encode to Unknown format"),
                };
                var stream = new MemoryStream();
                encoder.Encode(image, stream);
                return stream.ToArray();
            }

            public Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
               return DecodeImage(image, DetectFormat(image), mode);
            }
            public Tensor DecodeImage(byte[] bytes, ImageFormat format, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
                switch (mode) {
                
                case ImageReadMode.UNCHANGED:
                    return format switch {
                        ImageFormat.PNG => ToTensor<Rgba32>(bytes),
                        ImageFormat.JPEG => ToTensor<Rgb24>(bytes),
                        _ => throw new ArgumentException("Cannot decode Unknown format")
                    };

                case ImageReadMode.RGB_ALPHA:
                    return ToTensor<Rgba32>(bytes);
                case ImageReadMode.RGB:
                    return ToTensor<Rgb24>(bytes);
                case ImageReadMode.GRAY_ALPHA:
                    return ToTensor<La16>(bytes);
                case ImageReadMode.GRAY:
                    return ToTensor<L8>(bytes);
                default: throw new NotImplementedException();
                }
            }

            public byte[] EncodeImage(Tensor image, ImageFormat format)
            {
                Tensor permuted = image.permute(1, 2, 0);
                return image.shape[0] switch {
                    1 => FromTensor<L8>(permuted, format),
                    2 => FromTensor<La16>(permuted, format),
                    3 => FromTensor<Rgb24>(permuted, format),
                    4 => FromTensor<Rgba32>(permuted, format),
                    _ => throw new ArgumentException("image tensor must have a colour channel of 1,2,3 or 4")
                };
            }
        }
    }
}
