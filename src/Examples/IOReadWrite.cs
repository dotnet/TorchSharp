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
        internal static void Main(string[] args)
        {
            var filename = "image.png";

            var img = torchvision.io.read_image(filename);

            var imgGray = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.GRAY);


            Console.WriteLine($"Image Tensor: {img}\n Gray Image Tensor: {imgGray}");

            torchvision.io.save_image(img, "image_orig.png");
            torchvision.io.save_image(imgGray, "image_gray.png");
        }
    }
}
