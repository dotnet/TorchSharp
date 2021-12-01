using System;
using System.Collections.Generic;
using System.Text;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {
        public enum ImageReadMode
        {
            UNCHANGED,
            GRAY,
            GRAY_ALPHA,
            RGB,
            RGB_ALPHA
        }

        public static Tensor read_image(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED)
        {
            throw new NotImplementedException();
        }

        public static void save_image(Tensor image, string filename)
        {
            throw new NotImplementedException();
        }
    }
}
