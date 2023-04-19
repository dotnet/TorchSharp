// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
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
                    /// Outputs a `Summary` protocol buffer with a histogram.
                    /// The generated
                    /// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
                    /// has one summary value containing a histogram for `values`.
                    /// This op reports an `InvalidArgument` error if any value is not finite.
                    ///
                    /// https://github.com/pytorch/pytorch/blob/1.7/torch/utils/tensorboard/summary.py#L283
                    /// </summary>
                    /// <param name="name"> A name for the generated node. Will also serve as a series name in TensorBoard. </param>
                    /// <param name="values"> A real numeric `Tensor`. Any shape. Values to use to build the histogram. </param>
                    /// <param name="bins"></param>
                    /// <param name="max_bins"></param>
                    /// <returns> A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer. </returns>
                    public static Tensorboard.Summary histogram(string name, Tensor values, Tensor bins, long? max_bins = null)
                    {
                        if (values.device.type != DeviceType.CPU)
                            values = values.cpu();
                        Tensorboard.HistogramProto hist = make_histogram(values, bins, max_bins);
                        var summary = new Tensorboard.Summary();
                        summary.Value.Add(new Tensorboard.Summary.Types.Value() { Tag = name, Histo = hist });
                        return summary;
                    }

                    /// <summary>
                    /// Outputs a `Summary` protocol buffer with a histogram.
                    /// The generated
                    /// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
                    /// has one summary value containing a histogram for `values`.
                    /// This op reports an `InvalidArgument` error if any value is not finite.
                    ///
                    /// https://github.com/pytorch/pytorch/blob/1.7/torch/utils/tensorboard/summary.py#L283
                    /// </summary>
                    /// <param name="name"> A name for the generated node. Will also serve as a series name in TensorBoard. </param>
                    /// <param name="values"> A real numeric `Tensor`. Any shape. Values to use to build the histogram. </param>
                    /// <param name="bins"></param>
                    /// <param name="max_bins"></param>
                    /// <returns> A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer. </returns>
                    public static Tensorboard.Summary histogram(string name, Tensor values, HistogramBinSelector bins, long? max_bins = null)
                    {
                        Tensorboard.HistogramProto hist = make_histogram(values, bins, max_bins);
                        var summary = new Tensorboard.Summary();
                        summary.Value.Add(new Tensorboard.Summary.Types.Value() { Tag = name, Histo = hist });
                        return summary;
                    }

                    /// <summary>
                    /// Convert values into a histogram proto using logic from histogram.cc.
                    ///
                    /// https://github.com/pytorch/pytorch/blob/1.7/torch/utils/tensorboard/summary.py#L304
                    /// </summary>
                    /// <param name="values"></param>
                    /// <param name="bins"></param>
                    /// <param name="max_bins"></param>
                    /// <returns></returns>
                    /// <exception cref="ArgumentException"></exception>
                    public static Tensorboard.HistogramProto make_histogram(Tensor values, Tensor bins, long? max_bins = null)
                    {
                        if (values.numel() == 0)
                            throw new ArgumentException("The input has no element.");
                        if (values.dtype != ScalarType.Float64)
                            values = values.to_type(ScalarType.Float64);
                        values = values.reshape(-1);
                        (Tensor counts, Tensor limits) = torch.histogram(values, bins);
                        return make_histogram(values, counts, limits, max_bins);
                    }

                    /// <summary>
                    /// Convert values into a histogram proto using logic from histogram.cc.
                    ///
                    /// https://github.com/pytorch/pytorch/blob/1.7/torch/utils/tensorboard/summary.py#L304
                    /// </summary>
                    /// <param name="values"></param>
                    /// <param name="bins"></param>
                    /// <param name="max_bins"></param>
                    /// <returns></returns>
                    /// <exception cref="ArgumentException"></exception>
                    public static Tensorboard.HistogramProto make_histogram(Tensor values, HistogramBinSelector bins, long? max_bins = null)
                    {
                        if (values.numel() == 0)
                            throw new ArgumentException("The input has no element.");
                        if (values.dtype != ScalarType.Float64)
                            values = values.to_type(ScalarType.Float64);
                        values = values.reshape(-1);
                        (Tensor counts, Tensor limits) = torch.histogram(values, bins);
                        return make_histogram(values, counts, limits, max_bins);
                    }

                    private static Tensorboard.HistogramProto make_histogram(Tensor values, Tensor counts, Tensor limits, long? max_bins = null)
                    {
                        long num_bins = counts.shape[0];
                        if (max_bins != null && num_bins > max_bins) {
                            long subsampling = num_bins / max_bins.Value;
                            long subsampling_remainder = num_bins % subsampling;
                            if (subsampling_remainder != 0)
                                counts = nn.functional.pad(counts, (0, subsampling - subsampling_remainder), PaddingModes.Constant, 0);
                            counts = counts.reshape(-1, subsampling).sum(1);
                            Tensor new_limits = empty(new long[] { counts.numel() + 1 }, limits.dtype);
                            new_limits[TensorIndex.Slice(null, -1)] = limits[TensorIndex.Slice(null, -1, subsampling)];
                            new_limits[-1] = limits[-1];
                            limits = new_limits;
                        }

                        Tensor cum_counts = cumsum(greater(counts, 0), 0);
                        Tensor search_value = empty(2, cum_counts.dtype);
                        search_value[0] = 0;
                        search_value[1] = cum_counts[-1] - 1;
                        Tensor search_result = searchsorted(cum_counts, search_value, right: true);
                        long start = search_result[0].item<long>();
                        long end = search_result[1].item<long>() + 1;
                        counts = start > 0 ? counts[TensorIndex.Slice(start - 1, end)] : concatenate(new Tensor[] { tensor(new[] { 0 }), counts[TensorIndex.Slice(stop: end)] });
                        limits = limits[TensorIndex.Slice(start, end + 1)];

                        if (counts.numel() == 0 || limits.numel() == 0)
                            throw new ArgumentException("The histogram is empty, please file a bug report.");

                        Tensor sum_sq = values.dot(values);
                        Tensorboard.HistogramProto histogramProto = new Tensorboard.HistogramProto() {
                            Min = values.min().item<double>(),
                            Max = values.max().item<double>(),
                            Num = values.shape[0],
                            Sum = values.sum().item<double>(),
                            SumSquares = sum_sq.item<double>(),
                        };
                        histogramProto.BucketLimit.AddRange(limits.data<double>().ToArray());
                        histogramProto.Bucket.AddRange(counts.data<double>().ToArray());
                        return histogramProto;
                    }

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
                    internal static Tensorboard.Summary.Types.Image make_image(SKBitmap img, double rescale = 1)
                    {
                        using var image = img.Copy();
                        byte[] bmpData = image.Resize(new SKSizeI((int)(image.Width * rescale), (int)(image.Height * rescale)), SKFilterQuality.High).Encode(SKEncodedImageFormat.Png, 100).ToArray();
                        return new Tensorboard.Summary.Types.Image() { Height = image.Height, Width = image.Width, Colorspace = 4, EncodedImageString = ByteString.CopyFrom(bmpData) };
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
                        return new Tensorboard.Summary.Types.Image() { Height = h, Width = w, Colorspace = 4, EncodedImageString = ByteString.FromStream(stream) };
                    }

                    private static SKBitmap TensorToSKBitmap(Tensor tensor)
                    {
                        int h = (int)tensor.shape[0];
                        int w = (int)tensor.shape[1];
                        int c = (int)tensor.shape[2];

                        byte[,,] data = tensor.cpu().data<byte>().ToNDArray() as byte[,,];
                        var skBmp = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Opaque);
                        int pixelSize = 4;
                        unsafe {
                            byte* pSkBmp = (byte*)skBmp.GetPixels().ToPointer();
                            for (int i = 0; i < h; i++) {
                                for (int j = 0; j < w; j++) {
                                    pSkBmp[j * pixelSize] = data[i, j, 0];
                                    pSkBmp[j * pixelSize + 1] = data[i, j, 1];
                                    pSkBmp[j * pixelSize + 2] = data[i, j, 2];
                                    pSkBmp[j * pixelSize + 3] = c == 4 ? data[i, j, 3] : (byte)255;
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
