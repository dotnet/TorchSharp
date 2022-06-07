using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Runtime.InteropServices;

using TorchSharp;

using static TorchSharp.torch;

using SkiaSharp;

namespace TorchSharp.Examples
{
    class IOReadWrite
    {
        internal class SkiaImager : torchvision.io.Imager
        {
            public override torchvision.ImageFormat DetectFormat(byte[] bytes)
            {
                var skdata = SKData.CreateCopy(bytes);

                var skcodec = SKCodec.Create(skdata);

                return skcodec.EncodedFormat switch {
                    SKEncodedImageFormat.Jpeg => torchvision.ImageFormat.Jpeg,
                    SKEncodedImageFormat.Png => torchvision.ImageFormat.Png,
                    _ => torchvision.ImageFormat.Unknown
                };
            }

            public override Tensor DecodeImage(byte[] image, torchvision.ImageFormat format, torchvision.io.ImageReadMode mode)
            {
                // Basic implemetation for only ImageReadMode.Unchanged.
                var bitmap = SKBitmap.Decode(image);

                // TODO: Roll the channels dim so this returns rgb instead of bgr.
                return tensor(bitmap.Bytes, new long[] { bitmap.Height, bitmap.Width, bitmap.BytesPerPixel }).permute(2, 0, 1);
            }
            public override byte[] EncodeImage(Tensor image, torchvision.ImageFormat format)
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
                
                return result.Encode(fmt, 100).ToArray();
            }
        }

        internal static void Main(string[] args)
        {
            var filename = args[0];

            Console.WriteLine($"Reading file {filename}");

            torchvision.io.DefaultImager = new SkiaImager();

            var img = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB_ALPHA);

            // Add a batch dimension
            var expanded = img.unsqueeze(0);

            Console.WriteLine($"Image has {expanded.shape[1]} colour channels with dimensions {expanded.shape[2]}x{expanded.shape[3]}");

            var transformed = torchvision.transforms.Compose(
                torchvision.transforms.Invert(),
                torchvision.transforms.HorizontalFlip(),
                //torchvision.transforms.CenterCrop(256),
                torchvision.transforms.Rotate(50)
                ).forward(expanded);


            torchvision.io.write_image(transformed.squeeze(), "image_transformed.jpg", torchvision.ImageFormat.Jpeg);
        }
    }
}
