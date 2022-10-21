// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class io
        {
            public class NotImplementedImager : Imager
            {
                System.Exception exp = new System.NotImplementedException("You need to provide your own DefaultImager or specify the Imager for all image I/O method calls");
                public override Tensor DecodeImage(Stream image, ImageReadMode mode = ImageReadMode.UNCHANGED)
                {
                    throw exp;
                }

                public override Tensor DecodeImage(byte[] data, ImageReadMode mode = ImageReadMode.UNCHANGED)
                {
                    throw new System.NotImplementedException();
                }

                public override Task<Tensor> DecodeImageAsync(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED, CancellationToken cancellationToken = default)
                {
                    throw exp;
                }
                public override void EncodeImage(Tensor image, ImageFormat format, Stream stream)
                {
                    throw exp;
                }

                public override byte[] EncodeImage(Tensor image, ImageFormat format)
                {
                    throw new System.NotImplementedException();
                }

                public override Task EncodeImageAsync(Tensor image, ImageFormat format, Stream stream, CancellationToken cancellationToken = default)
                {
                    throw exp;
                }
            }
        }
    }
}