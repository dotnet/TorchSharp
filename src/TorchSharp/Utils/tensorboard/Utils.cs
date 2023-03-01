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
                    /// https://github.com/pytorch/pytorch/blob/6c30dc6ceed5542351b3be4f8043b28020f93f3a/torch/utils/tensorboard/_utils.py#L69
                    /// </summary>
                    /// <param name="I"></param>
                    /// <param name="ncols"></param>
                    /// <returns></returns>
                    public static Tensor make_grid(Tensor I, long ncols = 8)
                    {
                        if (I.shape[1] == 1)
                            I = cat(new Tensor[] { I, I, I }, 1);
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
                        Trace.Assert(tensor.shape[0] == input_format.Length, $"size of input tensor and input format are different. tensor shape: ({string.Join(", ", tensor.shape)}), input_format: {input_format}");
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
                                tensor_HWC = cat(new Tensor[] { tensor_HWC, tensor_HWC, tensor_HWC }, 2);
                            return tensor_HWC;
                        }

                        if (input_format.Length == 2) {
                            long[] index = "HW".Select(c => Convert.ToInt64(input_format.IndexOf(c))).ToArray();
                            tensor = tensor.permute(index);
                            tensor = stack(new Tensor[] { tensor, tensor, tensor }, 2);
                            return tensor;
                        }

                        throw new NotImplementedException();
                    }
                }
            }
        }
    }
}
