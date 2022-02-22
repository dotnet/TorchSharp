using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using TorchSharp;
using static TorchSharp.torch;

using SkiaSharp;

namespace TorchSharp.Examples
{
    class IOReadWrite
    {
        class SkiaImager : torchvision.io.IImager
        {
            public torchvision.io.ImageFormat DetectFormat(byte[] bytes)
            {
                throw new NotImplementedException();
            }
            public Tensor DecodeImage(byte[] image, torchvision.io.ImageFormat format,  torchvision.io.ImageReadMode mode)
            {
                throw new NotImplementedException();
            }
            public Tensor DecodeImage(byte[] image, torchvision.io.ImageReadMode mode)
            {
                throw new NotImplementedException();
            }
            public byte[] EncodeImage(Tensor image, torchvision.io.ImageFormat format)
            {
                throw new NotImplementedException();
            }
        }

        internal static void Main(string[] args)
        {
            var filename = args[0];

            //torchvision.io.DefaultImager = new SkiaImager();

            var img = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB);

            // Add a batch dimension
            var expanded  = img.unsqueeze(0);

            Console.WriteLine($"Image has {expanded.shape[1]} colour channels with dimensions {expanded.shape[2]}x{expanded.shape[3]}");

            //var transformed = torchvision.transforms.Compose(
            //    torchvision.transforms.Invert()
            //    ).forward(img);

            var transformed = torchvision.transforms.functional.vflip(expanded);

            torchvision.io.write_image(transformed.squeeze(), "image_transformed_vflip.jpg", torchvision.io.ImageFormat.JPEG);
        }
    }
}
