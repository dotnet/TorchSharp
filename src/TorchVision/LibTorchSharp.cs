// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    internal static class NativeMethods
    {
        /* Tensor THSVision_AdjustHue(const Tensor i, const double hue_factor) */
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSVision_AdjustHue(IntPtr img, double hue_factor);

        // EXPORT_API(Tensor) THSVision_ApplyGridTransform(Tensor img, Tensor grid, const int8_t m, const float* fill, const int64_t fill_length);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSVision_ApplyGridTransform(IntPtr img, IntPtr grid, sbyte mode, IntPtr fill,
            long fill_length);

        /* Tensor THSVision_GenerateAffineGrid(Tensor theta, const int64_t w, const int64_t h, const int64_t ow, const int64_t oh); */
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSVision_GenerateAffineGrid(IntPtr theta, long w, long h, long ow, long oh);

        /* Tensor THSVision_ComputeOutputSize(const float* matrix, const int64_t matrix_length, const int64_t w, const int64_t h); */
        [DllImport("LibTorchSharp")]
        internal static extern void THSVision_ComputeOutputSize(IntPtr matrix, long matrix_length, long w, long h,
            out int first, out int second);

        /* Tensor THSVision_PerspectiveGrid(const float* coeffs, const int64_t coeffs_length, const int64_t ow, const int64_t oh, const int8_t scalar_type, const int device_type, const int device_index); */
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSVision_PerspectiveGrid(IntPtr coeffs, long coeffs_length, long ow, long oh,
            sbyte dtype, int device_type, int device_index);

        /* Tensor THSVision_ScaleChannel(Tensor ic); */
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSVision_ScaleChannel(IntPtr img);

        [DllImport("LibTorchSharp")]
        internal static extern void THSVision_BRGA_RGB(IntPtr inputBytes, IntPtr redBytes, IntPtr greenBytes, IntPtr blueBytes, long inputChannelCount, long imageSize);

        [DllImport("LibTorchSharp")]
        internal static extern void THSVision_BRGA_RGBA(IntPtr inputBytes, IntPtr redBytes, IntPtr greenBytes, IntPtr blueBytes, IntPtr alphaBytes, long inputChannelCount, long imageSize);

        [DllImport("LibTorchSharp")]
        internal static extern void THSVision_RGB_BRGA(IntPtr inputBytes, IntPtr outBytes, long inputChannelCount, long imageSize);
    }
}