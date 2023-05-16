// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Runtime.InteropServices;
using SkiaSharp;
using static TorchSharp.torch;
using static TorchSharp.NativeMethods;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class io
        {
            public sealed class SkiaImager : Imager
            {

                public SkiaImager(int quality = 75)
                {
                    this.quality = quality;
                }
                private int quality;

                public override Tensor DecodeImage(byte[] bytes, ImageReadMode mode = ImageReadMode.UNCHANGED)
                {
                    using var bitmap = SKBitmap.Decode(bytes);
                    return ToTensor(mode, bitmap);
                }

                public override Tensor DecodeImage(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED)
                {
                    using var bitmap = SKBitmap.Decode(stream);
                    return ToTensor(mode, bitmap);
                }

                public override void EncodeImage(Tensor image, ImageFormat format, Stream stream)
                {
                    var skiaFormat = TranslateImageFormat(format); // Better to take the exception early.

                    var result = new SKBitmap();

                    var lstIdx = image.shape.Length;

                    var channels = image.shape[lstIdx - 3];
                    var height = image.shape[lstIdx - 2];
                    var width = image.shape[lstIdx - 1];

                    var imageSize = height * width;

                    var isGrey = channels == 1;

                    byte[] outBytes = null;

                    if (isGrey) {
                        outBytes = image.bytes.ToArray();
                    } else {
                        outBytes = new byte[4 * imageSize];

                        unsafe {
                            fixed (byte* input = image.bytes, output = outBytes) {
                                THSVision_RGB_BRGA((IntPtr)input, (IntPtr)output, channels, imageSize);
                            }
                        }
                    }

                    SKColorType colorType = isGrey ? SKColorType.Gray8 : SKColorType.Bgra8888;

                    // pin the managed array so that the GC doesn't move it
                    var gcHandle = GCHandle.Alloc(outBytes, GCHandleType.Pinned);

                    // install the pixels with the color type of the pixel data
                    var info = new SKImageInfo((int)width, (int)height, colorType, SKAlphaType.Unpremul);
                    result.InstallPixels(info, gcHandle.AddrOfPinnedObject(), info.RowBytes, delegate { gcHandle.Free(); }, null);

                    result.Encode(stream, skiaFormat, this.quality);
                    stream.Flush();
                }

                public override byte[] EncodeImage(Tensor image, ImageFormat format)
                {
                    using var memStream = new MemoryStream();
                    EncodeImage(image, format, memStream);
                    return memStream.ToArray();
                }

                private static Tensor ToTensor(ImageReadMode mode, SKBitmap bitmap)
                {
                    if (mode == ImageReadMode.UNCHANGED) {
                        mode = bitmap.ColorType == SKColorType.Gray8 ? ImageReadMode.GRAY : ImageReadMode.RGB;
                    }

                    if (bitmap.ColorType == SKColorType.Gray8 && mode == ImageReadMode.GRAY)
                        return torch.tensor(bitmap.Bytes, 1, bitmap.Height, bitmap.Width);

                    using var scope = NewDisposeScope();

                    if (bitmap.ColorType == SKColorType.Gray8 && mode == ImageReadMode.RGB) {
                        Tensor t = torch.tensor(bitmap.Bytes, 1, bitmap.Height, bitmap.Width);
                        return t.expand(3, bitmap.Height, bitmap.Width).MoveToOuterDisposeScope();
                    }

                    if (bitmap.BytesPerPixel != 4 && bitmap.BytesPerPixel != 1)
                        throw new ArgumentException("Conversion only supports grayscale and ARGB");

                    var imageSize = bitmap.Height * bitmap.Width;

                    byte[] redBytes = new byte[imageSize], greenBytes = new byte[imageSize], blueBytes = new byte[imageSize];
                    byte[] alphaBytes = null;

                    bool outputGrayScale = mode == ImageReadMode.GRAY_ALPHA || mode == ImageReadMode.GRAY;

                    Tensor result;

                    unsafe {
                        if (mode == ImageReadMode.GRAY || mode == ImageReadMode.RGB) {
                            fixed (byte* inputs = bitmap.Bytes, red = redBytes, blue = blueBytes, green = greenBytes) {
                                THSVision_BRGA_RGB((IntPtr)inputs, (IntPtr)red, (IntPtr)green, (IntPtr)blue, 4, imageSize);
                            }
                            result = torch.vstack( new[] {
                                    torch.tensor(redBytes, 1, bitmap.Height, bitmap.Width),
                                    torch.tensor(greenBytes, 1, bitmap.Height, bitmap.Width),
                                    torch.tensor(blueBytes, 1, bitmap.Height, bitmap.Width) });
                        } else if (mode == ImageReadMode.RGB_ALPHA || mode == ImageReadMode.GRAY_ALPHA) {
                            alphaBytes = new byte[imageSize];
                            fixed (byte* inputs = bitmap.Bytes, red = redBytes, blue = blueBytes, green = greenBytes, alpha = alphaBytes) {
                                THSVision_BRGA_RGBA((IntPtr)inputs, (IntPtr)red, (IntPtr)green, (IntPtr)blue, (IntPtr)alpha, 4, imageSize);
                                result = torch.vstack(new[] {
                                    torch.tensor(alphaBytes, 1, bitmap.Height, bitmap.Width),
                                    torch.tensor(redBytes, 1, bitmap.Height, bitmap.Width),
                                    torch.tensor(greenBytes, 1, bitmap.Height, bitmap.Width),
                                    torch.tensor(blueBytes, 1, bitmap.Height, bitmap.Width) });
                            }
                        } else {
                            throw new NotImplementedException();
                        }
                    }

                    if (outputGrayScale) {
                        result = torchvision.transforms.functional.rgb_to_grayscale(result);
                    }

                    return result.MoveToOuterDisposeScope();
                }

                private SKEncodedImageFormat TranslateImageFormat(ImageFormat format)
                {
                    switch (format) {
                    case ImageFormat.Png:
                        return SKEncodedImageFormat.Png;
                    case ImageFormat.Jpeg:
                        return SKEncodedImageFormat.Jpeg;
                    case ImageFormat.Bmp:
                        return SKEncodedImageFormat.Bmp;
                    case ImageFormat.Gif:
                        return SKEncodedImageFormat.Gif;
                    case ImageFormat.Webp:
                        return SKEncodedImageFormat.Webp;
                    }
                    throw new NotSupportedException($"SkiaSharp does not support encoding images in the '{format}' format.");
                }
            }
        }
    }
}
