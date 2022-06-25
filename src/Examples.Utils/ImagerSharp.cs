using System;
using System.IO;
using System.Runtime.CompilerServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp.torchvision;
using static TorchSharp.torch;

namespace TorchSharp.Examples.Utils
{
        /// <summary>
        /// <cref>Imager</cref> implemented using ImageSharp.
        /// </summary>
        public sealed class ImagerSharp : io.Imager
        {
            private Tensor ToTensor<TPixel>(Stream stream) where TPixel : unmanaged, IPixel<TPixel>
            {
                var image = Image.Load<TPixel>(stream);
                var channels = Unsafe.SizeOf<TPixel>();
                byte[] imageBytes = new byte[image.Height * image.Width * channels];
                image.CopyPixelDataTo(imageBytes);
                return tensor(imageBytes, new long[] { image.Height, image.Width, channels }).permute(2, 0, 1);
            }

            private void FromTensor<TPixel>(Tensor t, ImageFormat format, Stream stream) where TPixel : unmanaged, IPixel<TPixel>
            {
                var shape = t.shape;
                var tt = t.reshape(new long[] { shape[0] * shape[1] * shape[2] });
                var image = Image.LoadPixelData<TPixel>(tt.data<byte>().ToArray(), (int)shape[1], (int)shape[0]);
                IImageEncoder encoder = format switch {
                    ImageFormat.Png => new PngEncoder(),
                    ImageFormat.Jpeg => new JpegEncoder(),
                    _ => throw new ArgumentException("Cannot encode to Unknown format"),
                };
                encoder.Encode(image, stream);
            }

            public override Tensor DecodeImage(byte[] bytes, io.ImageReadMode mode = io.ImageReadMode.UNCHANGED)
            {
                using (var stream = new MemoryStream(bytes))
                    return DecodeImage(stream, mode);
            }

            public override Tensor DecodeImage(Stream stream, io.ImageReadMode mode = io.ImageReadMode.UNCHANGED)
            {
                switch (mode) {
                case io.ImageReadMode.UNCHANGED:
                    var format = Image.DetectFormat(stream);
                    if (format is PngFormat) {
                        return ToTensor<Rgba32>(stream);
                    } else {
                        return ToTensor<Rgb24>(stream);
                    }
                case io.ImageReadMode.RGB_ALPHA:
                    return ToTensor<Rgba32>(stream);
                case io.ImageReadMode.RGB:
                    return ToTensor<Rgb24>(stream);
                case io.ImageReadMode.GRAY_ALPHA:
                    return ToTensor<La16>(stream);
                case io.ImageReadMode.GRAY:
                    return ToTensor<L8>(stream);
                default: throw new NotImplementedException();
                }
            }

            public override byte[] EncodeImage(Tensor image, ImageFormat format)
            {
                using (var stream = new MemoryStream()) {
                    EncodeImage(image, format, stream);
                    return stream.ToArray();
                }
            }

            public override void EncodeImage(Tensor image, ImageFormat format, Stream stream)
            {
                Tensor permuted = image.permute(1, 2, 0);
                switch (image.shape[0]) {
                case 1:
                    FromTensor<L8>(permuted, format, stream);
                    break;
                case 2:
                    FromTensor<L16>(permuted, format, stream);
                    break;
                case 3:
                    FromTensor<Rgb24>(permuted, format, stream);
                    break;
                case 4:
                    FromTensor<Rgba32>(permuted, format, stream);
                    break;
                default:
                    throw new ArgumentException("image tensor must have a colour channel of 1,2,3 or 4");
                }
            }
    }
}
