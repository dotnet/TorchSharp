// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SkiaSharp;
using static TorchSharp.torch;

namespace TorchSharp.Examples
{
    public static class Tensorboard
    {
        private static string imagePath = "TensorboardExample/Images";
        internal static async Task Main(string[] args)
        {
            await DownloadExampleData("https://agirls.aottercdn.com/media/0d532b3f-0196-466a-96e1-7c6db56d0142.gif");
            var device = cuda.is_available() ? CUDA : CPU;
            var writer = utils.tensorboard.SummaryWriter("tensorboard");
            Console.WriteLine($"Running Tensorboard on {device.type}");
            AddText(writer);
            AddImageUsePath(writer);
            AddImageUseTensor(writer, device);
            AddVideo(writer, device);
            AddHistogram(writer, device);
        }

        private static void AddHistogram(Modules.SummaryWriter writer, Device device)
        {
            for (int i = 0; i < 10; i++) {
                Tensor x = randn(1000, device: device);
                writer.add_histogram("AddHistogram", x + i, i);
            }
        }

        private static void AddText(Modules.SummaryWriter writer)
        {
            writer.add_text("AddText", "step_1", 1);
            writer.add_text("AddText", "step_2", 2);
            writer.add_text("AddText", "step_3", 3);
            writer.add_text("AddText", "step_4", 4);
            writer.add_text("AddText", "step_5", 5);
        }

        private static void AddImageUsePath(Modules.SummaryWriter writer)
        {
            var imagesPath = Directory.GetFiles(imagePath);
            for (var i = 0; i < imagesPath.Length; i++) {
                writer.add_img("AddImageUsePath", imagesPath[i], i);
            }
        }

        private static void AddImageUseTensor(Modules.SummaryWriter writer, Device device)
        {
            var images = Directory.GetFiles(imagePath).Select(item => SKBitmap.Decode(item)).ToArray();
            using var d = NewDisposeScope();
            for (var i = 0; i < images.Length; i++) {
                var tensor = SKBitmapToTensor(images[i], device);
                writer.add_img("AddImageUseTensor", tensor, i, dataformats: "CHW");
                images[i].Dispose();
            }
        }

        private static void AddVideo(Modules.SummaryWriter writer, Device device)
        {
            var images = Directory.GetFiles(imagePath).Select(item => SKBitmap.Decode(item)).ToArray();
            using var d = NewDisposeScope();
            var tensor = stack(images.Select(item => SKBitmapToTensor(item, device)).ToArray());
            tensor = stack(new Tensor[] { tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor });
            foreach (var image in images)
                image.Dispose();
            writer.add_video("AddVideo", tensor, 1, 10);
            writer.add_video("AddVideo", tensor, 2, 10);
            writer.add_video("AddVideo", tensor, 3, 10);
        }

        private static Tensor SKBitmapToTensor(SKBitmap skBitmap, Device device)
        {
            var width = skBitmap.Width;
            var height = skBitmap.Height;
            var hwcData = new byte[4, height, width];
            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    var color = skBitmap.GetPixel(x, y);
                    hwcData[0, y, x] = color.Red;
                    hwcData[1, y, x] = color.Green;
                    hwcData[2, y, x] = color.Blue;
                    hwcData[3, y, x] = color.Alpha;
                }
            }

            return tensor(hwcData, ScalarType.Byte, device);
        }

        private static async Task DownloadExampleData(string url)
        {
            if (Directory.Exists(imagePath))
                Directory.Delete(imagePath, true);
            Directory.CreateDirectory(imagePath);
            using var client = new HttpClient();

            using var message = await client.GetAsync(url);
            using var stream = await message.Content.ReadAsStreamAsync();
            using var animatedImage = Image.Load<Rgba32>(stream);
            for (var i = 0; i < animatedImage.Frames.Count; i++) {
                var pngFilename = Path.Combine(imagePath, $"frame_{i}.png");
                using var pngImage = animatedImage.Frames.CloneFrame(i);
                pngImage.SaveAsPng(pngFilename);
            }
        }
    }
}
