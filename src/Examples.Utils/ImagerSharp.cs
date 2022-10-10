using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

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

        private async Task<Tensor> ToTensorAsync<TPixel>(Stream stream, CancellationToken token) where TPixel : unmanaged, IPixel<TPixel>
        {
            var image = await Image.LoadAsync<TPixel>(stream, token);
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

        private Task FromTensorAsync<TPixel>(Tensor t, ImageFormat format, Stream stream, CancellationToken token) where TPixel : unmanaged, IPixel<TPixel>
        {
            var shape = t.shape;
            var tt = t.reshape(new long[] { shape[0] * shape[1] * shape[2] });
            var image = Image.LoadPixelData<TPixel>(tt.data<byte>().ToArray(), (int)shape[1], (int)shape[0]);
            IImageEncoder encoder = format switch {
                ImageFormat.Png => new PngEncoder(),
                ImageFormat.Jpeg => new JpegEncoder(),
                _ => throw new ArgumentException("Cannot encode to Unknown format"),
            };
            return encoder.EncodeAsync(image, stream, token);
        }

        public override async Task<Tensor> DecodeImageAsync(Stream stream, io.ImageReadMode mode = io.ImageReadMode.UNCHANGED, CancellationToken cancellationToken = default)
        {
            switch (mode) {
            case io.ImageReadMode.UNCHANGED:
                var format = Image.DetectFormat(stream);
                if (format is PngFormat) {
                    return await ToTensorAsync<Rgba32>(stream, cancellationToken);
                } else {
                    return await ToTensorAsync<Rgb24>(stream, cancellationToken);
                }
            case io.ImageReadMode.RGB_ALPHA:
                return await ToTensorAsync<Rgba32>(stream, cancellationToken);
            case io.ImageReadMode.RGB:
                return await ToTensorAsync<Rgb24>(stream, cancellationToken);
            case io.ImageReadMode.GRAY_ALPHA:
                return await ToTensorAsync<La16>(stream, cancellationToken);
            case io.ImageReadMode.GRAY:
                return await ToTensorAsync<L8>(stream, cancellationToken);
            default: throw new NotImplementedException();
            }
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

        public override async Task EncodeImageAsync(Tensor image, ImageFormat format, Stream stream, CancellationToken cancellationToken = default)
        {
            Tensor permuted = image.permute(1, 2, 0);
            switch (image.shape[0]) {
            case 1:
                await FromTensorAsync<L8>(permuted, format, stream, cancellationToken);
                break;
            case 2:
                await FromTensorAsync<L16>(permuted, format, stream, cancellationToken);
                break;
            case 3:
                await FromTensorAsync<Rgb24>(permuted, format, stream, cancellationToken);
                break;
            case 4:
                await FromTensorAsync<Rgba32>(permuted, format, stream, cancellationToken);
                break;
            default:
                throw new ArgumentException("image tensor must have a colour channel of 1,2,3 or 4");
            }
        }

        public override Tensor DecodeImage(byte[] data, io.ImageReadMode mode = io.ImageReadMode.UNCHANGED)
        {
            using var memStream = new MemoryStream(data);
            return DecodeImage(memStream, mode);
        }

        public override byte[] EncodeImage(Tensor image, ImageFormat format)
        {
            using var memStream = new MemoryStream();
            EncodeImage(image, format, memStream);
            return memStream.ToArray();
        }
    }
}
