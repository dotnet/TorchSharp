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

            torchvision.io.DefaultImager = new SkiaImager();

            var img = torchvision.io.read_image(filename);

            Console.WriteLine($"Image has shape {img.shape}");

            var imgGray = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.GRAY);

            Console.WriteLine($"Image Tensor: {img}\n Gray Image Tensor: {imgGray}");

            torchvision.io.write_image(img, "image_orig.png");
            var encoded = torchvision.io.encode_image(imgGray);
            torchvision.io.write_file("image_gray", encoded);
        }
    }
}
