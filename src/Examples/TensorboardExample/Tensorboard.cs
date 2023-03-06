// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.IO;
using System.Linq;
using SkiaSharp;
using static TorchSharp.torch;

namespace TorchSharp.Examples.TensorboardExample
{
    public static class Tensorboard
    {
        internal static void Main(string[] args)
        {
            var writer = torch.utils.tensorboard.SummaryWriter("tensorboard");
            AddText(writer);
            AddImageUsePath(writer);
            AddImageUseTensor(writer);
            AddVideo(writer);
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
            string[] imagesPath = Directory.GetFiles("TensorboardExample/Images");
            for (int i = 0; i < imagesPath.Length; i++) {
                writer.add_img("AddImageUsePath", imagesPath[i], i);
            }
        }

        private static void AddImageUseTensor(Modules.SummaryWriter writer)
        {
            SKBitmap[] images = Directory.GetFiles("TensorboardExample/Images").Select(item => SKBitmap.Decode(item)).ToArray();
            for (int i = 0; i < images.Length; i++) {
                Tensor tensor = SKBitmapToTensor(images[i]);
                writer.add_img("AddImageUseTensor", tensor, i, dataformats: "CHW");
                images[i].Dispose();
            }
        }

        private static void AddVideo(Modules.SummaryWriter writer)
        {
            SKBitmap[] images = Directory.GetFiles("TensorboardExample/Images").Select(item => SKBitmap.Decode(item)).ToArray();
            Tensor tensor = stack(images.Select(item => SKBitmapToTensor(item)).ToArray()).unsqueeze(0);
            foreach (var image in images)
                image.Dispose();
            writer.add_video("AddVideo", tensor, 1, 10);
            writer.add_video("AddVideo", tensor, 2, 10);
            writer.add_video("AddVideo", tensor, 3, 10);

        }

        private static Tensor SKBitmapToTensor(SKBitmap skBitmap)
        {
            int width = skBitmap.Width;
            int height = skBitmap.Height;
            byte[,,] hwcData = new byte[3, height, width];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    SKColor color = skBitmap.GetPixel(x, y);
                    hwcData[0, y, x] = color.Red;
                    hwcData[1, y, x] = color.Green;
                    hwcData[2, y, x] = color.Blue;
                }
            }

            return tensor(hwcData, ScalarType.Byte);
        }
    }
}
