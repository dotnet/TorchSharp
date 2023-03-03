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
            //AddText(writer);
            //AddImageUsePath(writer);
            //AddImageUseTensor(writer);
            AddVideo(writer);
        }

        public static void AddText(Modules.SummaryWriter writer)
        {
            writer.add_text("AddText", "step_1", 1);
            writer.add_text("AddText", "step_2", 2);
            writer.add_text("AddText", "step_3", 3);
            writer.add_text("AddText", "step_4", 4);
            writer.add_text("AddText", "step_5", 5);
        }

        public static void AddImageUsePath(Modules.SummaryWriter writer)
        {
            string[] imagesPath = Directory.GetFiles("TensorboardExample/Images");
            for (int i = 0; i < imagesPath.Length; i++) {
                writer.add_img("AddImageUsePath", imagesPath[i], i);
            }
        }

        public static void AddImageUseTensor(Modules.SummaryWriter writer)
        {
            SKBitmap[] images = Directory.GetFiles("TensorboardExample/Images").Select(item => SKBitmap.Decode(item)).ToArray();
            for (int i = 0; i < images.Length; i++) {
                Tensor tensor = SKBitmapToTensor(images[i]);
                writer.add_img("AddImageUseTensor", tensor, i, dataformats: "HWC");
                images[i].Dispose();
            }
        }

        public static void AddVideo(Modules.SummaryWriter writer)
        {
            SKBitmap[] images = Directory.GetFiles("TensorboardExample/Images").Select(item => SKBitmap.Decode(item)).ToArray();
            Tensor tensor = stack(images.Select(item => SKBitmapToTensor(item)).ToArray()).unsqueeze(0);
            foreach (var image in images)
                image.Dispose();
            writer.add_video("AddVideo", tensor, 1, 10);
            //writer.add_video("AddVideo", tensor, 2, 10);
            //writer.add_video("AddVideo", tensor, 3, 10);

        }

        private static Tensor SKBitmapToTensor(SKBitmap skBitmap)
        {
            int width = skBitmap.Width;
            int height = skBitmap.Height;
            byte[,,] hwcData = new byte[height, width, 3];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    SKColor color = skBitmap.GetPixel(x, y);
                    hwcData[y, x, 0] = color.Red;
                    hwcData[y, x, 1] = color.Green;
                    hwcData[y, x, 2] = color.Blue;
                }
            }

            return tensor(hwcData, ScalarType.Byte);
        }
    }
}
