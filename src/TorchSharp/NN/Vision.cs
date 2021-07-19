// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {

        public enum UpsampleMode
        {
            Nearest = 0,
            Linear = 1,
            Bilinear = 2,
            Bicubic = 3,
            Trilinear = 4
        }

        public enum InterpolationMode
        {
            Nearest = 0,
            Linear = 1,
            Bilinear = 2,
            Bicubic = 3,
            Trilinear = 4,
            Area = 5
        }

        public enum GridSampleMode
        {
            Nearest = 0,
            Bilinear = 2,
            // PyTorch docs say this should be available,
            // but the libtorch native code does not support it.
            //Bicubic = 3
        }

        public enum GridSamplePaddingMode
        {
            Zeros = 0,
            Reflection = 1,
            Border = 2,
        }

        public static partial class nn
        {

            public static partial class functional
            {
                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_pad(IntPtr input, IntPtr pad, int pad_length, byte mode, double value);

                [DllImport("LibTorchSharp")]
                // align_corners -- 0=None, 1=true, 2=false
                private static extern IntPtr THSNN_interpolate(IntPtr input, IntPtr size, int size_len, IntPtr scale_factor, int scale_factor_len, byte mode, byte align_corners, bool recompute_scale_factor);


                [DllImport("LibTorchSharp")]
                // align_corners -- 0=None, 1=true, 2=false
                private static extern IntPtr THSNN_grid_sample(IntPtr input, IntPtr grid, byte mode, byte padding_mode, byte align_corners);

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_affine_grid(IntPtr theta, IntPtr size, int size_len, bool align_corners);


                /// <summary>
                /// Pads tensor.
                /// </summary>
                /// <param name="input">N-dimensional tensor</param>
                /// <param name="pad">m-elements tuple, where m/2 ≤ input dimensions and m is even</param>
                /// <param name="mode">'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'</param>
                /// <param name="value">Fill value for 'constant' padding. Default: 0</param>
                /// <returns></returns>
                static public Tensor pad(Tensor input, long[] pad, PaddingModes mode = PaddingModes.Constant, double value = 0)
                {
                    //
                    // The Pytorch documentation does not cover what is actually happening in the native code, as far as
                    // the ordering of padding elements goes. This code converts from the documented order, to the actual.
                    // See: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.pad
                    //
                    long[] correctedPad = pad;

                    if (input.ndim > 1) {
                        switch (pad.Length) {
                        case 1:
                            correctedPad = new long[] { pad[0], pad[0], pad[0], pad[0] };
                            break;
                        case 2:
                            correctedPad = new long[] { pad[0], pad[0], pad[1], pad[1] };
                            break;
                        case 4:
                            correctedPad = new long[] { pad[0], pad[2], pad[1], pad[3] };
                            break;
                        }
                    }

                    unsafe {
                        fixed (long* psize = correctedPad) {
                            var res = THSNN_pad(input.Handle, (IntPtr)psize, correctedPad.Length, (byte)mode, value);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Pads tensor.
                /// </summary>
                /// <param name="input">N-dimensional tensor</param>
                /// <param name="pad">A single padding size, used for all edges.</param>
                /// <param name="mode">'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'</param>
                /// <param name="value">Fill value for 'constant' padding. Default: 0</param>
                /// <returns></returns>
                static public Tensor pad(Tensor input, long pad, PaddingModes mode = PaddingModes.Constant, double value = 0)
                {
                    long[] correctedPad = new long[] { pad, pad, pad, pad };

                    unsafe {
                        fixed (long* psize = correctedPad) {
                            var res = THSNN_pad(input.Handle, (IntPtr)psize, 4, (byte)mode, value);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }


                /// <summary>
                /// Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.
                /// </summary>
                /// <param name="input">Tensor of 4D or 5D</param>
                /// <param name="grid">Flow-field tensor.</param>
                /// <param name="mode">Interpolation mode to calculate output values 'bilinear' | 'nearest' | 'bicubic'.</param>
                /// <param name="padding_mode">Padding mode for outside grid values 'zeros' | 'border' | 'reflection'. Default: 'zeros'</param>
                /// <param name="align_corners">Geometrically, we consider the pixels of the input as squares rather than points.
                /// If set to true, the extrema (-1 and 1) are considered as referring to the center points of the input’s corner pixels.
                /// If set to false, they are instead considered as referring to the corner points of the input’s corner pixels, making the sampling more resolution agnostic.</param>
                /// <returns></returns>
                /// <remarks>
                /// Currently, only spatial (4-D) and volumetric (5-D) input are supported.
                /// Note: mode='bicubic' supports only 4-D input.
                /// </remarks>
                static public Tensor grid_sample(Tensor input, Tensor grid, GridSampleMode mode = GridSampleMode.Bilinear, GridSamplePaddingMode padding_mode = GridSamplePaddingMode.Zeros, bool? align_corners = null)
                {
                    byte ac = (byte)((align_corners.HasValue) ? (align_corners.Value ? 1 : 2) : 0);
                    var res = THSNN_grid_sample(input.Handle, grid.Handle, (byte)mode, (byte)padding_mode, ac);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta.
                /// </summary>
                /// <param name="theta">Input batch of affine matrices with shape (N, 2, 3 ) for 2D or (N, 3, 4 ) for 3D</param>
                /// <param name="size">The target output image size</param>
                /// <param name="align_corners">if true, consider -1 and 1 to refer to the centers of the corner pixels rather than the image corners.
                /// Refer to grid_sample() for a more complete description.</param>
                /// <returns></returns>
                static public Tensor affine_grid(Tensor theta, long[]? size = null, bool align_corners = false)
                {
                    unsafe {
                        fixed (long* psize = size) {
                            var res = THSNN_affine_grid(theta.Handle, (IntPtr)psize, size is null ? 0 : size.Length, align_corners);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// 
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="size">Output spatial size</param>
                /// <param name="scale_factor">Multiplier for spatial size. Has to match input size if it is a tuple.</param>
                /// <param name="mode">The algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'</param>
                /// <param name="align_corners">Geometrically, we consider the pixels of the input and output as squares rather than points.
                /// If set to true, the input and output tensors are aligned by the center points of their corner pixels, preserving the values at the corner pixels.
                /// If set to false, the input and output tensors are aligned by the corner points of their corner pixels, and the interpolation uses edge value padding for out-of-boundary values, making this operation independent of input size when scale_factor is kept the same.</param>
                /// <param name="recompute_scale_factor">
                /// Recompute the scale_factor for use in the interpolation calculation.
                /// When scale_factor is passed as a parameter, it is used to compute the output_size.
                /// If recompute_scale_factor is False or not specified, the passed-in scale_factor will be used in the interpolation computation.
                /// Otherwise, a new scale_factor will be computed based on the output and input sizes for use in the interpolation computation
                /// (i.e. the computation will be identical to if the computed output_size were passed-in explicitly).
                /// </param>
                /// <returns></returns>
                static public Tensor interpolate(Tensor x, long[]? size = null, double[]? scale_factor = null, InterpolationMode mode = InterpolationMode.Nearest, bool? align_corners = null, bool recompute_scale_factor = false)
                {
                    unsafe {
                        fixed (long* psize = size) {
                            fixed (double* pSF = scale_factor) {
                                byte ac = (byte)((align_corners.HasValue) ? (align_corners.Value ? 1 : 2) : 0);
                                var res = THSNN_interpolate(x.Handle, (IntPtr)psize, size is null ? 0 : size.Length, (IntPtr)pSF, scale_factor is null ? 0 : scale_factor.Length, (byte)mode, ac, recompute_scale_factor);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                /// <summary>
                /// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
                /// The input data is assumed to be of the form minibatch x channels x[optional depth] x[optional height] x width.
                /// Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
                /// </summary>
                /// <param name="x">Input tensor</param>
                /// <param name="size">Output spatial sizes</param>
                /// <param name="scale_factor">Multiplier for spatial size. Has to match input size</param>
                /// <param name="align_corners">If true, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
                /// This only has effect when mode is 'linear', 'bilinear', or 'trilinear'. Default: false</param>
                /// <returns></returns>
                static public Tensor upsample_bilinear(Tensor x, long[]? size = null, double[]? scale_factor = null, bool align_corners = false)
                {
                    using (var d = torch.nn.Upsample(size, scale_factor, UpsampleMode.Bilinear, align_corners)) {
                        return d.forward(x);
                    }
                }

                /// <summary>
                /// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
                /// The input data is assumed to be of the form minibatch x channels x[optional depth] x[optional height] x width.
                /// Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
                /// </summary>
                /// <param name="x">Input tensor</param>
                /// <param name="size">Output spatial sizes</param>
                /// <param name="scale_factor">Multiplier for spatial size. Has to match input size</param>
                /// <param name="align_corners">If true, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
                /// This only has effect when mode is 'linear', 'bilinear', or 'trilinear'. Default: false</param>
                /// <returns></returns>
                static public Tensor upsample_nearest(Tensor x, long[]? size = null, double[]? scale_factor = null, bool align_corners = false)
                {
                    using (var d = torch.nn.Upsample(size, scale_factor, UpsampleMode.Nearest, align_corners)) {
                        return d.forward(x);
                    }
                }
            }
        }
    }
}
