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
            public Tensor DecodeImage(byte[] image, torchvision.io.ImageReadMode mode)
            {
                throw new NotImplementedException();
            }
            public byte[] EncodeImage(Tensor image, torchvision.io.ImageReadMode mode)
            {
                throw new NotImplementedException();
            }
        }

        internal static async void Main(string[] args)
        {
            var filename = "image.png";

            torchvision.io.PrimaryImager = new SkiaImager();

            var img = torchvision.io.read_image(filename);

            var imgGray = await torchvision.io.read_image_async(filename, torchvision.io.ImageReadMode.GRAY);

            Console.WriteLine($"Image Tensor: {img}\n Gray Image Tensor: {imgGray}");

            torchvision.io.write_image(img, "image_orig.png");
            var encoded = torchvision.io.encode_image(imgGray);
            torchvision.io.write_file("image_gray", encoded);
        }
    }
}
