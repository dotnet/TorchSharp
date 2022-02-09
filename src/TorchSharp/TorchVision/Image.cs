using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {

        public static IImager DefaultImager { get; set; } = new ImagerSharp();

        public enum ImageFormat
        {
            PNG,
            JPEG,
            UNKNOWN
        }
        public interface IImager
        {
            ImageFormat DetectFormat(byte[] image);
            Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED);
            Tensor DecodeImage(byte[] image, ImageFormat format, ImageReadMode mode = ImageReadMode.UNCHANGED);
            byte[] EncodeImage(Tensor image, ImageFormat format);
        }

        public enum ImageReadMode
        {
            UNCHANGED,
            GRAY,
            GRAY_ALPHA,
            RGB,
            RGB_ALPHA
        }

        public static Tensor read_image(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, IImager imager = null)
        {
            return (imager ?? DefaultImager).DecodeImage(File.ReadAllBytes(filename), mode);
        }

        public static async Task<Tensor> read_image_async(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, IImager imager = null)
        {
            var data = await File.ReadAllBytesAsync(filename);
            return (imager ?? DefaultImager).DecodeImage(data, mode);
        }

        public static void write_image(Tensor image, string filename, IImager imager = null)
        {
            // Use DefaultImager if imager == null
            throw new NotImplementedException();
        }

        public static Tensor encode_image(Tensor image, IImager imager = null)
        {
            // Use DefaultImager if imager == null
            throw new NotImplementedException();
        }

        public static Tensor decode_image(Tensor image, IImager imager = null)
        {
            // Use DefaultImager if imager == null
            throw new NotImplementedException();
        }
    }
}
