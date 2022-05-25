using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {

        public static Imager DefaultImager { get; set; } = new ImagerSharp();

        public abstract class Imager
        {
            public abstract ImageFormat DetectFormat(byte[] image);
            public Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
                return DecodeImage(image, DetectFormat(image), mode);
            }
            public abstract Tensor DecodeImage(byte[] image, ImageFormat format, ImageReadMode mode = ImageReadMode.UNCHANGED);
            
            public abstract byte[] EncodeImage(Tensor image, ImageFormat format);
        }

        public enum ImageReadMode
        {
            UNCHANGED,
            GRAY,
            GRAY_ALPHA,
            RGB,
            RGB_ALPHA
        }

        public static Tensor read_image(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            return (imager ?? DefaultImager).DecodeImage(File.ReadAllBytes(filename), mode);
        }

        public static async Task<Tensor> read_image_async(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            byte[] data;

            using (FileStream stream = File.Open(filename, FileMode.Open)) {
                data = new byte[stream.Length];
                await stream.ReadAsync(data, 0, data.Length);
            }

            return (imager ?? DefaultImager).DecodeImage(data, mode);
        }

        public static void write_image(Tensor image, string filename, ImageFormat format, Imager imager = null)
        {
            File.WriteAllBytes(filename, (imager ?? DefaultImager).EncodeImage(image, format));
        }

        public static async void write_image_async(Tensor image, string filename, ImageFormat format, Imager imager = null)
        {
            var data = (imager ?? DefaultImager).EncodeImage(image, format);
            using (FileStream stream = File.Open(filename, FileMode.OpenOrCreate)) {
                await stream.WriteAsync(data, 0, data.Length);
            }
        }

        public static Tensor encode_image(Tensor image, ImageFormat format, Imager imager = null)
        {
            return (imager ?? DefaultImager).EncodeImage(image, format);
        }

        public static Tensor decode_image(Tensor image, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            return (imager ?? DefaultImager).DecodeImage(image.bytes.ToArray(), mode);
        }
    }
}
