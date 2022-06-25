using System.IO;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {
        public sealed class NotImplementedImager : Imager
        {
            System.Exception exp = new System.NotImplementedException("You need to provide your own DefaultImager or specify the Imager for all image I/O method calls");
            public override Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
                throw exp;
            }
            public override Tensor DecodeImage(Stream image, ImageReadMode mode = ImageReadMode.UNCHANGED)
            {
                throw exp;
            }
            public override byte[] EncodeImage(Tensor image, ImageFormat format)
            {
                throw exp;
            }
            public override void EncodeImage(Tensor image, ImageFormat format, Stream stream)
            {
                throw exp;
            }
        }
    }
}