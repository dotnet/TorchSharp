using System;

namespace TorchSharp.Examples
{
    class IOReadWrite
    {
        internal static void Main(string[] args)
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

            var filename = args[0];

            Console.WriteLine($"Reading file {filename}");

            var img = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.GRAY);

            Console.WriteLine($"Image has {img.shape[0]} colour channels with dimensions {img.shape[1]}x{img.shape[2]}");

            var transformed = torchvision.transforms.Compose(
                //torchvision.transforms.Invert(),
                torchvision.transforms.HorizontalFlip(),
                //torchvision.transforms.CenterCrop(256),
                torchvision.transforms.Rotate(50)
                ).call(img);

            var out_file = "image_transformed.jpg";

            torchvision.io.write_image(transformed, out_file, torchvision.ImageFormat.Jpeg);

            Console.WriteLine($"Wrote file {out_file}");
        }
    }
}
