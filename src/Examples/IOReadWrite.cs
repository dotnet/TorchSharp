using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using TorchSharp;
using static TorchSharp.torch;

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

            Console.WriteLine($"Image has {img.shape[0]} colour channels with dimensions {img.shape[1]}x{img.shape[2]}");

            //var transformed = torchvision.transforms.Compose(
            //    torchvision.transforms.AutoContrast(),
            //    torchvision.transforms.Invert()
            //    ).forward(img);

            var transformed = torchvision.transforms.RandomVerticalFlip(1).forward(img);

            torchvision.io.write_image(transformed, "image_transformed_2.png", torchvision.io.ImageFormat.PNG);
        }
    }
}
