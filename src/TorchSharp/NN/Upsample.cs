// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
            /// The input data is assumed to be of the form minibatch x channels x[optional depth] x[optional height] x width.
            /// Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
            /// </summary>
            /// <param name="size">Output spatial sizes</param>
            /// <param name="scale_factor">Multiplier for spatial size. Has to match input size</param>
            /// <param name="mode">The upsampling algorithm: one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. Default: 'nearest'</param>
            /// <param name="alignCorners">If true, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
            /// This only has effect when mode is 'linear', 'bilinear', or 'trilinear'. Default: false</param>
            /// <returns></returns>
            public static Upsample Upsample(long[]? size = null, double[]? scale_factor = null, UpsampleMode mode = UpsampleMode.Nearest, bool? alignCorners = null)
            {
                unsafe {
                    fixed (long* psize = size) {
                        fixed (double* pSF = scale_factor) {
                            byte ac = (byte)((alignCorners.HasValue) ? (alignCorners.Value ? 1 : 2) : 0);
                            var res = THSNN_Upsample_ctor((IntPtr)psize, size is null ? 0 : size.Length, (IntPtr)pSF, scale_factor is null ? 0 : scale_factor.Length, (byte)mode, ac, out var boxedHandle);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Upsample(res, boxedHandle);
                        }
                    }
                }
            }

            public static partial class functional
            {
                /// <summary>
                /// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
                /// The input data is assumed to be of the form minibatch x channels x[optional depth] x[optional height] x width.
                /// Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
                /// </summary>
                /// <param name="x">Input tensor</param>
                /// <param name="size">Output spatial sizes</param>
                /// <param name="scale_factor">Multiplier for spatial size. Has to match input size</param>
                /// <param name="mode">The upsampling algorithm: one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. Default: 'nearest'</param>
                /// <param name="alignCorners">If true, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
                /// This only has effect when mode is 'linear', 'bilinear', or 'trilinear'. Default: false</param>
                /// <returns></returns>
                public static Tensor upsample(Tensor x, long[]? size = null, double[]? scale_factor = null, UpsampleMode mode = UpsampleMode.Nearest, bool alignCorners = false)
                {
                    using (var d = nn.Upsample(size, scale_factor, mode, alignCorners)) {
                        return d.call(x);
                    }
                }


                /// <summary>
                /// Upsamples the input, using nearest neighbours’ pixel values.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSize"></param>
                /// <param name="scaleFactor"></param>
                /// <returns></returns>
                public static Tensor upsample_nearest1d(Tensor input, long? outputSize, double? scaleFactor)
                {
                    var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
                    var outputSizesLength = outputSize.HasValue ? 1 : 0;
                    var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
                    var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest1d(input.Handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                public static Tensor upsample_nearest1d_backward(Tensor grad_output, long? outputSize, long inputSize, double? scaleFactor)
                {
                    var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
                    var outputSizesLength = outputSize.HasValue ? 1 : 0;
                    var inputSizes = new long[] { inputSize };
                    var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
                    var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest1d_backward(grad_output.Handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pinputSizes, inputSizes.Length,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                /// <summary>
                /// Upsamples the input, using nearest neighbours’ pixel values.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSizes"></param>
                /// <param name="scaleFactors"></param>
                /// <returns></returns>
                public static Tensor upsample_nearest2d(Tensor input, long[]? outputSizes = null, double[]? scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest2d(input.Handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                public static Tensor upsample_nearest2d_backward(Tensor grad_output, long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest2d_backward(grad_output.Handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pinputSizes, inputSizes.Length,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                public static Tensor upsample_nearest3d_backward(Tensor grad_output, long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest3d_backward(grad_output.Handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pinputSizes, inputSizes.Length,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                /// <summary>
                /// Upsamples the input, using nearest neighbours’ pixel values.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSizes"></param>
                /// <param name="scaleFactors"></param>
                /// <returns></returns>
                public static Tensor upsample_nearest3d(Tensor input, long[]? outputSizes = null, double[]? scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest3d(input.Handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }
            }
        }
    }

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent an Upsample module.
        /// </summary>
        public sealed class Upsample : torch.nn.Module<Tensor, Tensor>
        {
            internal Upsample(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Upsample_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }
}
