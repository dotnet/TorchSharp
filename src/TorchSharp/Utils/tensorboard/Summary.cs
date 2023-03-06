// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using SkiaSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class tensorboard
            {
                public static partial class Summary
                {
                    private static int calc_scale_factor(Tensor tensor)
                        => tensor.dtype == ScalarType.Byte || tensor.dtype == ScalarType.Int8 ? 1 : 255;

                    /// <summary>
                    /// Outputs a `Summary` protocol buffer with images.
                    /// The summary has up to `max_images` summary values containing images. The
                    /// images are built from `tensor` which must be 3-D with shape `[height, width,
                    /// channels]` and where `channels` can be:
                    /// *  1: `tensor` is interpreted as Grayscale.
                    /// *  3: `tensor` is interpreted as RGB.
                    /// *  4: `tensor` is interpreted as RGBA.
                    /// The `name` in the outputted Summary.Value protobufs is generated based on the
                    /// name, with a suffix depending on the max_outputs setting:
                    /// *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
                    /// *  If `max_outputs` is greater than 1, the summary value tags are
                    ///    generated sequentially as '*name*/image/0', '*name*/image/1', etc.
                    /// </summary>
                    /// <param name="tag"> A name for the generated node. Will also serve as a series name in TensorBoard. </param>
                    /// <param name="tensor">
                    /// A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
                    /// channels]` where `channels` is 1, 3, or 4.
                    /// 'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
                    /// The image() function will scale the image values to [0, 255] by applying
                    /// a scale factor of either 1 (uint8) or 255 (float32). Out-of-range values
                    /// will be clipped.
                    /// </param>
                    /// <param name="rescale"> Rescale image size </param>
                    /// <param name="dataformats"> Image data format specification of the form CHW, HWC, HW, WH, etc. </param>
                    /// <returns> A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer. </returns>
                    public static Tensorboard.Summary image(string tag, Tensor tensor, double rescale = 1, string dataformats = "NCHW")
                    {
                        tensor = utils.convert_to_HWC(tensor, dataformats);
                        int scale_factor = calc_scale_factor(tensor);
                        tensor = tensor.to_type(ScalarType.Float32);
                        tensor = (tensor * scale_factor).clip(0, 255).to_type(ScalarType.Byte);
                        Tensorboard.Summary.Types.Image image = make_image(tensor, rescale);
                        var summary = new Tensorboard.Summary();
                        summary.Value.Add(new Tensorboard.Summary.Types.Value() { Tag = tag, Image = image });
                        return summary;
                    }

                    /// <summary>
                    /// Outputs a `Summary` protocol buffer with images.
                    /// </summary>
                    /// <param name="tag"> A name for the generated node. Will also serve as a series name in TensorBoard. </param>
                    /// <param name="file_name"> local image filename </param>
                    /// <param name="rescale"> Rescale image size </param>
                    /// <returns> A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer. </returns>
                    public static Tensorboard.Summary image(string tag, string file_name, double rescale = 1)
                    {
                        using var img = SKBitmap.Decode(file_name);
                        Tensorboard.Summary.Types.Image image = make_image(img, rescale);
                        var summary = new Tensorboard.Summary();
                        summary.Value.Add(new Tensorboard.Summary.Types.Value() { Tag = tag, Image = image });
                        return summary;
                    }

                    /// <summary>
                    /// Convert a tensor representation of an image to Image protobuf
                    /// 
                    /// https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/summary.py#L481
                    /// </summary>
                    /// <param name="tensor"> HWC(0~255) image tensor </param>
                    /// <param name="rescale"> Rescale image size </param>
                    /// <returns></returns>
                    public static Tensorboard.Summary.Types.Image make_image(Tensor tensor, double rescale = 1)
                    {
                        using SKBitmap skBmp = TensorToSKBitmap(tensor);
                        return make_image(skBmp, rescale);
                    }

                    /// <summary>
                    /// Convert an image to Image protobuf
                    /// 
                    /// https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/summary.py#L495
                    /// </summary>
                    /// <param name="img"> Image </param>
                    /// <param name="rescale"> Rescale image size </param>
                    /// <returns></returns>
                    public static Tensorboard.Summary.Types.Image make_image(SKBitmap img, double rescale = 1)
                    {
                        using var image = img.Copy();
                        byte[] bmpData = image.Resize(new SKSizeI((int)(image.Width * rescale), (int)(image.Height * rescale)), SKFilterQuality.High).Encode(SKEncodedImageFormat.Png, 100).ToArray();
                        return new Tensorboard.Summary.Types.Image() { Height = image.Height, Width = image.Width, Colorspace = 3, EncodedImageString = ByteString.CopyFrom(bmpData) };
                    }

                    /// <summary>
                    /// https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/summary.py#L509
                    /// </summary>
                    /// <param name="tag"> A name for the generated node. Will also serve as a series name in TensorBoard. </param>
                    /// <param name="tensor"> Video data </param>
                    /// <param name="fps"> Frames per second </param>
                    /// <returns></returns>
                    public static Tensorboard.Summary video(string tag, Tensor tensor, int fps)
                    {
                        tensor = utils.prepare_video(tensor);
                        int scale_factor = calc_scale_factor(tensor);
                        tensor = tensor.to_type(ScalarType.Float32);
                        tensor = (tensor * scale_factor).clip(0, 255).to_type(ScalarType.Byte);
                        Tensorboard.Summary.Types.Image video = make_video(tensor, fps);
                        var summary = new Tensorboard.Summary();
                        summary.Value.Add(new Tensorboard.Summary.Types.Value() { Tag = tag, Image = video });
                        return summary;
                    }

                    /// <summary>
                    /// https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/summary.py#L520
                    /// </summary>
                    /// <param name="tensor"> Video data </param>
                    /// <param name="fps"> Frames per second </param>
                    /// <returns></returns>
                    public static Tensorboard.Summary.Types.Image make_video(Tensor tensor, int fps)
                    {
                        int h = (int)tensor.shape[1];
                        int w = (int)tensor.shape[2];
                        int c = (int)tensor.shape[3];
                        using GifEncoder.Encoder encoder = new GifEncoder.Encoder();
                        encoder.Start();
                        encoder.SetRepeat(0);
                        encoder.SetFrameRate(fps);
                        foreach (var t in tensor.split(1)) {
                            using SKBitmap bitmap = TensorToSKBitmap(t.squeeze());
                            encoder.AddFrame(bitmap);
                        }
                        encoder.Finish();
                        Stream stream = encoder.Output();
                        stream.Position = 0;
                        return new Tensorboard.Summary.Types.Image() { Height = h, Width = w, Colorspace = c, EncodedImageString = ByteString.FromStream(stream) };
                    }

                    private static SKBitmap TensorToSKBitmap(Tensor tensor)
                    {
                        int h = (int)tensor.shape[0];
                        int w = (int)tensor.shape[1];

                        byte[,,] data = tensor.cpu().data<byte>().ToNDArray() as byte[,,];
                        var skBmp = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Opaque);
                        int pixelSize = (int)tensor.shape[2] + 1;
                        unsafe {
                            byte* pSkBmp = (byte*)skBmp.GetPixels().ToPointer();
                            for (int i = 0; i < h; i++) {
                                for (int j = 0; j < w; j++) {
                                    pSkBmp[j * pixelSize] = data[i, j, 0];
                                    pSkBmp[j * pixelSize + 1] = data[i, j, 1];
                                    pSkBmp[j * pixelSize + 2] = data[i, j, 2];
                                    pSkBmp[j * pixelSize + 3] = 255;
                                }
                                pSkBmp += skBmp.Info.RowBytes;
                            }
                        }

                        return skBmp;
                    }

                    /// <summary>
                    /// https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/summary.py#L630
                    /// </summary>
                    /// <param name="tag"> Data identifier </param>
                    /// <param name="text"> String to save </param>
                    /// <returns></returns>
                    public static Tensorboard.Summary text(string tag, string text)
                    {
                        // TextPluginData(version=0).SerializeToString()
                        // output: b''
                        var pluginData = new Tensorboard.SummaryMetadata.Types.PluginData() { PluginName = "text", Content = ByteString.CopyFromUtf8("") };
                        var smd = new Tensorboard.SummaryMetadata() { PluginData = pluginData };
                        var shapeProto = new Tensorboard.TensorShapeProto();
                        shapeProto.Dim.Add(new Tensorboard.TensorShapeProto.Types.Dim() { Size = 1 });
                        var tensor = new Tensorboard.TensorProto() { Dtype = Tensorboard.DataType.DtString, TensorShape = shapeProto };
                        tensor.StringVal.Add(ByteString.CopyFromUtf8(text));

                        var summary = new Tensorboard.Summary();
                        summary.Value.Add(new Tensorboard.Summary.Types.Value() { Tag = tag + "/text_summary", Metadata = smd, Tensor = tensor });
                        return summary;
                    }
                }
            }
        }
    }
}
