// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        public partial class Tensor
        {
            // https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine
            /// <summary>
            /// Returns a new tensor with the data in this fake quantized per channel using
            /// <paramref name="scale"/>, <paramref name="zero_point"/>, <paramref name="quant_min"/> and <paramref name="quant_max"/>,
            /// across the channel specified by <paramref name="axis"/>.
            /// </summary>
            /// <param name="scale">quantization scale, per channel (float32)</param>
            /// <param name="zero_point">quantization zero_point, per channel (torch.int32, torch.half, or torch.float32)</param>
            /// <param name="axis">channel axis</param>
            /// <param name="quant_min">lower bound of the quantized domain</param>
            /// <param name="quant_max">upper bound of the quantized domain</param>
            /// <returns>A newly fake_quantized per channel torch.float32 tensor</returns>
            public Tensor fake_quantize_per_channel_affine(Tensor scale, Tensor zero_point, long axis, long quant_min, long quant_max)
            {
                var res = THSTensor_fake_quantize_per_channel_affine(
                    Handle, scale.Handle, zero_point.handle,
                    axis, quant_min, quant_max);

                if (res == IntPtr.Zero)
                    CheckForErrors();

                return new Tensor(res);
            }

            // see: aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp
            internal (Tensor res, Tensor mask) fake_quantize_per_channel_affine_cachemask(Tensor scale, Tensor zero_point, long axis, long quant_min, long quant_max)
            {
                var res = THSTensor_fake_quantize_per_channel_affine_cachemask(
                    Handle, scale.Handle, zero_point.handle,
                    axis, quant_min, quant_max, out IntPtr mask);

                if (res == IntPtr.Zero || mask == IntPtr.Zero)
                    CheckForErrors();

                return (new Tensor(res), new Tensor(mask));
            }
        }
    }
}