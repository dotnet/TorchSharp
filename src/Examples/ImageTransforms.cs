// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SkiaSharp;
using static TorchSharp.torch;
using System.Runtime.InteropServices;

namespace TorchSharp.Examples
{
    public class ImageTransforms
    {
        internal static void Main(string[] args)
        {
            var images = new string[] {
                //
                // Find some PNG (or JPEG, etc.) files, download them, and then put their file paths here.
                //
            };

            const string outputPathPrefix = /* Add the very first part of your repo path here. */ @"\TorchSharp\output-";
            var tensors = LoadImages(images, 4, 3, 256, 256);

            var first = tensors[0];

            int n = 0;

            // First, use the transform version.

            var transform = torchvision.transforms.Compose(
                torchvision.transforms.ConvertImageDtype(ScalarType.Float32),
                //torchvision.transforms.ColorJitter(.5f, .5f, .5f, .25f),
                torchvision.transforms.ConvertImageDtype(ScalarType.Byte),
                torchvision.transforms.Resize(256, 256)
                );

            var second = transform.call(first);

            for (; n < second.shape[0]; n++) {

                var image = second[n]; // CxHxW
                var channels = image.shape[0];

                using (var stream = File.OpenWrite(outputPathPrefix + n + ".png")) {
                    var bitmap = GetBitmapFromBytes(image.data<byte>().ToArray(), 256, 256, channels == 1 ? SKColorType.Gray8 : SKColorType.Bgra8888);
                    bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            }

            // Then the functional API version.

            second = torchvision.transforms.functional.convert_image_dtype(first);
            // Have to do this to make sure that everything's in the right format before saving.
            second = torchvision.transforms.functional.convert_image_dtype(second, dtype: ScalarType.Byte);
            second = torchvision.transforms.functional.equalize(second);
            second = torchvision.transforms.functional.resize(second, 256, 256);

            for (n = 0; n < second.shape[0]; n++) {

                var image = second[n]; // CxHxW
                var channels = image.shape[0];

                using (var stream = File.OpenWrite(outputPathPrefix + (n + first.shape[0]) + ".png")) {
                    var bitmap = GetBitmapFromBytes(image.data<byte>().ToArray(), 256, 256, channels == 1 ? SKColorType.Gray8 : SKColorType.Bgra8888);
                    bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            }
        }

        private static List<Tensor> LoadImages(IList<string> images, int batchSize, int channels, int height, int width)
        {
            List<Tensor> tensors = new List<Tensor>();

            var imgSize = channels * height * width;
            bool shuffle = false;

            Random rnd = new Random();
            var indices = !shuffle ?
                Enumerable.Range(0, images.Count).ToArray() :
                Enumerable.Range(0, images.Count).OrderBy(c => rnd.Next()).ToArray();


            // Go through the data and create tensors
            for (var i = 0; i < images.Count;) {

                var take = Math.Min(batchSize, Math.Max(0, images.Count - i));

                if (take < 1) break;

                var dataTensor = torch.zeros(new long[] { take, imgSize }, ScalarType.Byte);

                // Take
                for (var j = 0; j < take; j++) {
                    var idx = indices[i++];
                    var lblStart = idx * (1 + imgSize);
                    var imgStart = lblStart + 1;

                    using (var stream = new SKManagedStream(File.OpenRead(images[idx])))
                    using (var bitmap = SKBitmap.Decode(stream)) {
                        using (var inputTensor = torch.tensor(GetBytesWithoutAlpha(bitmap))) {

                            Tensor finalized = inputTensor;

                            var nz = inputTensor.count_nonzero().item<long>();

                            if (bitmap.Width != width || bitmap.Height != height) {
                                var t = inputTensor.reshape(1, channels, bitmap.Height, bitmap.Width);
                                finalized = torchvision.transforms.functional.resize(t, height, width).reshape(imgSize);
                            }

                            dataTensor.index_put_(finalized, TensorIndex.Single(j));
                        }
                    }
                }

                tensors.Add(dataTensor.reshape(take, channels, height, width));
                dataTensor.Dispose();
            }

            return tensors;
        }

        private static byte[] GetBytesWithoutAlpha(SKBitmap bitmap)
        {
            var height = bitmap.Height;
            var width = bitmap.Width;

            var inputBytes = bitmap.Bytes;

            if (bitmap.ColorType == SKColorType.Gray8)
                return inputBytes;

            if (bitmap.BytesPerPixel != 4 && bitmap.BytesPerPixel != 1)
                throw new ArgumentException("Conversion only supports grayscale and ARGB");

            var channelLength = height * width;

            var channelCount = 3;

            int inputBlue = 0, inputGreen = 0, inputRed = 0;
            int outputRed = 0, outputGreen = channelLength, outputBlue = channelLength * 2;

            switch (bitmap.ColorType) {
            case SKColorType.Bgra8888:
                inputBlue = 0;
                inputGreen = 1;
                inputRed = 2;
                break;

            default:
                throw new NotImplementedException($"Conversion from {bitmap.ColorType} to bytes");
            }
            var outBytes = new byte[channelCount * channelLength];

            for (int i = 0, j = 0; i < channelLength; i += 1, j += 4) {
                outBytes[outputRed + i] = inputBytes[inputRed + j];
                outBytes[outputGreen + i] = inputBytes[inputGreen + j];
                outBytes[outputBlue + i] = inputBytes[inputBlue + j];
            }

            return outBytes;
        }

        private static SKBitmap GetBitmapFromBytes(byte[] inputBytes, int height, int width, SKColorType colorType)
        {
            var result = new SKBitmap();

            var channelLength = height * width;

            var channelCount = 0;

            int inputRed = 0, inputGreen = channelLength, inputBlue = channelLength * 2;
            int outputBlue = 0, outputGreen = 0, outputRed = 0, outputAlpha = 0;

            switch (colorType) {
            case SKColorType.Bgra8888:
                outputBlue = 0;
                outputGreen = 1;
                outputRed = 2;
                outputAlpha = 3;
                channelCount = 3;
                break;

            case SKColorType.Gray8:
                channelCount = 1;
                break;

            default:
                throw new NotImplementedException($"Conversion from {colorType} to bytes");
            }

            byte[] outBytes = null;

            if (channelCount == 1) {

                // Greyscale

                outBytes = inputBytes;
            } else {

                outBytes = new byte[(channelCount + 1) * channelLength];

                for (int i = 0, j = 0; i < channelLength; i += 1, j += 4) {
                    outBytes[outputRed + j] = inputBytes[inputRed + i];
                    outBytes[outputGreen + j] = inputBytes[inputGreen + i];
                    outBytes[outputBlue + j] = inputBytes[inputBlue + i];
                    outBytes[outputAlpha + j] = 255;
                }
            }

            // pin the managed array so that the GC doesn't move it
            var gcHandle = GCHandle.Alloc(outBytes, GCHandleType.Pinned);

            // install the pixels with the color type of the pixel data
            var info = new SKImageInfo(width, height, colorType, SKAlphaType.Unpremul);
            result.InstallPixels(info, gcHandle.AddrOfPinnedObject(), info.RowBytes, delegate { gcHandle.Free(); }, null);

            return result;
        }
    }
}
