using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.InteropServices;

using TorchSharp;

using static TorchSharp.torch;

using SkiaSharp;

namespace TorchSharp.Examples
{
    class IOReadWrite
    {
        /// Imager implemented with SkiaSharp.
        /// Ignores async implementations.
        internal class SkiaImager : torchvision.io.NotImplementedImager
        {
            Tensor DecodeBitmap(SKBitmap bitmap, torchvision.io.ImageReadMode mode)
            {
                if (bitmap.ColorType != SKColorType.Bgra8888)
                    throw new Exception("Unsupported format");

                var inputBytes = bitmap.Bytes;

                var outBytes = new byte[bitmap.Width * bitmap.Height * 4];
                var cl = bitmap.Width * bitmap.Height;
                for (int o = 0, i = 0; o < cl; o += 1, i += 4) {
                    outBytes[o] = inputBytes[i + 2];
                    outBytes[o + cl * 1] = inputBytes[i + 1];
                    outBytes[o + cl * 2] = inputBytes[i];
                    outBytes[o + cl * 3] = inputBytes[i + 3];
                }
                var rgba = tensor(outBytes, new long[] { 4, bitmap.Height, bitmap.Width });

                switch (mode) {
                case torchvision.io.ImageReadMode.UNCHANGED:
                case torchvision.io.ImageReadMode.RGB_ALPHA:
                    return rgba;
                case torchvision.io.ImageReadMode.GRAY:
                    return torchvision.transforms.Compose(
                        torchvision.transforms.ConvertImageDType(ScalarType.Float32),
                        torchvision.transforms.Grayscale(),
                        torchvision.transforms.ConvertImageDType(ScalarType.Byte)
                        ).forward(rgba);
                default:
                    throw new NotImplementedException();
                }
            }

            public override Tensor DecodeImage(Stream stream, torchvision.io.ImageReadMode mode)
            {
                return DecodeBitmap(SKBitmap.Decode(stream), mode);
            }

            public SKData EncodeToData(Tensor image, torchvision.ImageFormat format)
            {
                // Basic implementation for only 1 and 4 color channels.
                var shape = image.shape;

                var bytes = image.permute(1, 2, 0).reshape(new long[] { shape[0] * shape[1] * shape[2] }).bytes.ToArray();

                // pin the managed array so that the GC doesn't move it
                var gcHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);

                SKColorType ctype = shape[0] switch {
                    1 => SKColorType.Gray8,
                    4 => SKColorType.Rgba8888,
                    _ => throw new ArgumentException("Unsupported color channels"),
                };

                // install the pixels with the color type of the pixel data
                var info = new SKImageInfo((int) shape[2], (int) shape[1], ctype , SKAlphaType.Unpremul);

                var result = new SKBitmap();

                result.InstallPixels(info, gcHandle.AddrOfPinnedObject(), info.RowBytes, delegate { gcHandle.Free(); }, null);

                var fmt = format switch {
                    torchvision.ImageFormat.Png => SKEncodedImageFormat.Png,
                    torchvision.ImageFormat.Jpeg => SKEncodedImageFormat.Jpeg,
                    torchvision.ImageFormat.Gif => SKEncodedImageFormat.Gif,
                    _ => throw new ArgumentException("Unsupported format"),
                };
                return result.Encode(fmt, 100);
                
            }

            public override void EncodeImage(Tensor image, torchvision.ImageFormat format, Stream stream)
            {
                EncodeToData(image, format).SaveTo(stream);
            }
        }

        internal static void Main(string[] args)
        {
            torchvision.io.DefaultImager = new SkiaImager();

            var filename = args[0];

            Console.WriteLine($"Reading file {filename}");

            var img = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.GRAY);

            Console.WriteLine($"Image has {img.shape[0]} colour channels with dimensions {img.shape[1]}x{img.shape[2]}");

            var transformed = torchvision.transforms.Compose(
                //torchvision.transforms.Invert(),
                torchvision.transforms.HorizontalFlip(),
                //torchvision.transforms.CenterCrop(256),
                torchvision.transforms.Rotate(50)
                ).forward(img);

            var out_file = "image_transformed.jpg";

            torchvision.io.write_image(transformed, out_file, torchvision.ImageFormat.Jpeg);

            Console.WriteLine($"Wrote file {out_file}");
        }
    }
}
