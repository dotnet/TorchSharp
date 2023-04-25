// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Diagnostics;
using System.Linq;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class tensorboard
            {
                internal static partial class utils
                {
                    /// <summary>
                    /// Converts a 5D tensor [batchsize, time(frame), channel(color), height, width]
                    /// into 4D tensor with dimension[time(frame), new_width, new_height, channel].
                    /// A batch of images are spreaded to a grid, which forms a frame.
                    /// e.g. Video with batchsize 16 will have a 4x4 grid.
                    ///
                    /// https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py#L110
                    /// </summary>
                    /// <param name="V"></param>
                    /// <returns></returns>
                    public static Tensor prepare_video(Tensor V)
                    {
                        long b = V.shape[0];
                        long t = V.shape[1];
                        long c = V.shape[2];
                        long h = V.shape[3];
                        long w = V.shape[4];

                        if (V.dtype == ScalarType.Int8 || V.dtype == ScalarType.Byte)
                            V = V.to_type(ScalarType.Float32) / 255.0;

                        bool is_power2(long num)
                            => num != 0 && ((num & (num - 1)) == 0);
                        int bit_length(long value)
                            => Convert.ToString(value, 2).Length;

                        if (!is_power2(V.shape[0])) {
                            int len_addition = Convert.ToInt32(Math.Pow(2, bit_length(V.shape[0])) - V.shape[0]);
                            V = cat(new Tensor[] { V, zeros(new long[] { len_addition, t, c, h, w }, device: V.device) });
                        }

                        long n_rows = Convert.ToInt32(Math.Pow(2, (bit_length(V.shape[0]) - 1) / 2));
                        long n_cols = V.shape[0] / n_rows;

                        V = V.reshape(n_rows, n_cols, t, c, h, w);
                        V = V.permute(2, 0, 4, 1, 5, 3);
                        V = V.reshape(t, n_rows * h, n_cols * w, c);
                        return V;
                    }

                    /// <summary>
                    /// https://github.com/pytorch/pytorch/blob/6c30dc6ceed5542351b3be4f8043b28020f93f3a/torch/utils/tensorboard/_utils.py#L69
                    /// </summary>
                    /// <param name="I"></param>
                    /// <param name="ncols"></param>
                    /// <returns></returns>
                    public static Tensor make_grid(Tensor I, long ncols = 8)
                    {
                        if (I.shape[1] == 1)
                            I = I.expand(-1, 3);
                        Trace.Assert(I.ndim == 4 && I.shape[1] == 3);
                        long nimg = I.shape[0];
                        long H = I.shape[2];
                        long W = I.shape[3];
                        ncols = Math.Min(ncols, nimg);
                        long nrows = (long)Math.Ceiling((double)nimg / (double)ncols);
                        Tensor canvas = zeros(new long[] { 3, H * nrows, W * ncols }, I.dtype);
                        long i = 0;
                        for (long r = 0; r < nrows; r++) {
                            for (long c = 0; c < ncols; c++) {
                                if (i >= nimg)
                                    break;
                                canvas.narrow(1, r * H, H).narrow(2, c * W, W).copy_(I[i]);
                                i++;
                            }
                        }
                        return canvas;
                    }

                    /// <summary>
                    /// https://github.com/pytorch/pytorch/blob/6c30dc6ceed5542351b3be4f8043b28020f93f3a/torch/utils/tensorboard/_utils.py#L95
                    /// </summary>
                    /// <param name="tensor"> Image data </param>
                    /// <param name="input_format"> Image data format specification of the form NCHW, NHWC, CHW, HWC, HW, WH, etc. </param>
                    /// <returns></returns>
                    public static Tensor convert_to_HWC(Tensor tensor, string input_format)
                    {
                        Trace.Assert(tensor.shape.Length == input_format.Length, $"size of input tensor and input format are different. tensor shape: ({string.Join(", ", tensor.shape)}), input_format: {input_format}");
                        input_format = input_format.ToUpper();

                        if (input_format.Length == 4) {
                            long[] index = "NCHW".Select(c => Convert.ToInt64(input_format.IndexOf(c))).ToArray();
                            Tensor tensor_NCHW = tensor.permute(index);
                            Tensor tensor_CHW = make_grid(tensor_NCHW);
                            return tensor_CHW.permute(1, 2, 0);
                        }

                        if (input_format.Length == 3) {
                            long[] index = "HWC".Select(c => Convert.ToInt64(input_format.IndexOf(c))).ToArray();
                            Tensor tensor_HWC = tensor.permute(index);
                            if (tensor_HWC.shape[2] == 1)
                                tensor_HWC = tensor_HWC.expand(-1, -1, 3);
                            return tensor_HWC;
                        }

                        if (input_format.Length == 2) {
                            long[] index = "HW".Select(c => Convert.ToInt64(input_format.IndexOf(c))).ToArray();
                            tensor = tensor.permute(index);
                            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3);
                            return tensor;
                        }

                        throw new NotImplementedException();
                    }
                }
            }
        }
    }
}
