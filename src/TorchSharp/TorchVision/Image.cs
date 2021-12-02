using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {

        public static IImager PrimaryImager { get; set; }
        public interface IImager
        {
            Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED);
            byte[] EncodeImage(Tensor image, ImageReadMode mode = ImageReadMode.UNCHANGED);
        }

        public enum ImageReadMode
        {
            UNCHANGED,
            GRAY,
            //GRAY_ALPHA,
            RGB,
            //RGB_ALPHA
        }

        public static Tensor read_image(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, IImager imager = null)
        {
            // Use PrimaryImager if imager == null
            throw new NotImplementedException();
        }

        public static async Task<Tensor> read_image_async(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, IImager imager = null)
        {
            return await Task.Run(() =>
            {
                return ones(1);
            });
            
        }

        public static void write_image(Tensor image, string filename, IImager imager = null)
        {
            // Use PrimaryImager if imager == null
            throw new NotImplementedException();
        }

        public static Tensor encode_image(Tensor image, IImager imager = null)
        {
            // Use PrimaryImager if imager == null
            throw new NotImplementedException();
        }

        public static Tensor decode_image(Tensor image, IImager imager = null)
        {
            // Use PrimaryImager if imager == null
            throw new NotImplementedException();
        }
    }
}
