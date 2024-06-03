// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // This file contains the mathematical operators on Tensor
    public enum FFTNormType
    {
        Backward = 0,
        Forward = 1,
        Ortho = 2
    }

    public static partial class torch
    {
        public static partial class fft
        {
            /// <summary>
            /// Computes the one dimensional discrete Fourier transform of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="n">Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.</param>
            /// <param name="dim">The dimension along which to take the one dimensional FFT.</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            /// <remarks>The name was changed because it would conflict with its surrounding scope. That's not legal in .NET.</remarks>
            public static Tensor fft_(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_fft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the one dimensional inverse discrete Fourier transform of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="n">Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the IFFT.</param>
            /// <param name="dim">The dimension along which to take the one dimensional IFFT.</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            public static Tensor ifft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_ifft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the two-dimensional discrete Fourier transform of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the FFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            public static Tensor fft2(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("fft2() input should be at least 2D");
                if (dim == null) dim = new long[] { -2, -1 };
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_fft2(input.Handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the two-dimensional inverse discrete Fourier transform of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor ifft2(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("ifft2() input should be at least 2D");
                if (dim == null) dim = new long[] { -2, -1 };
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_ifft2(input.Handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the N-dimensional discrete Fourier transform of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            public static Tensor fftn(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                var slen = (s == null) ? 0 : s.Length;
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_fftn(input.Handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the N-dimensional inverse discrete Fourier transform of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            public static Tensor ifftn(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                var slen = (s == null) ? 0 : s.Length;
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_ifftn(input.Handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the one dimensional inverse Fourier transform of real-valued input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="n">Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.</param>
            /// <param name="dim">The dimension along which to take the one dimensional FFT.</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor irfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_irfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the one dimensional Fourier transform of real-valued input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="n">Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.</param>
            /// <param name="dim">The dimension along which to take the one dimensional FFT.</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            public static Tensor rfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_rfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the two-dimensional discrete Fourier transform of real-vaued input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the FFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor rfft2(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("rfft2() input should be at least 2D");
                if (dim == null) dim = new long[] { -2, -1 };
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_rfft2(input.Handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the two-dimensional inverse discrete Fourier transform of real-valued input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor irfft2(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("irfft2() input should be at least 2D");
                if (dim == null) dim = new long[] { -2, -1 };
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_irfft2(input.Handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the N-dimensional discrete Fourier transform of real-valued input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor rfftn(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                var slen = (s == null) ? 0 : s.Length;
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_rfftn(input.Handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the N-dimensional inverse discrete Fourier transform of real-valued input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor irfftn(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                var slen = (s == null) ? 0 : s.Length;
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_irfftn(input.Handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the one dimensional discrete Fourier transform of a Hermitian symmetric input signal.
            /// </summary>
            /// <param name="input">The input tensor representing a half-Hermitian signal</param>
            /// <param name="n">
            /// Output signal length. This determines the length of the real output.
            /// If given, the input will either be zero-padded or trimmed to this length before computing the Hermitian FFT.</param>
            /// <param name="dim">The dimension along which to take the one dimensional Hermitian FFT.</param>
            /// <param name="norm">Normalization mode.</param>
            /// <returns></returns>
            public static Tensor hfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_hfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse of hfft().
            /// </summary>
            /// <param name="input">The input tensor representing a half-Hermitian signal</param>
            /// <param name="n">
            /// Output signal length. This determines the length of the real output.
            /// If given, the input will either be zero-padded or trimmed to this length before computing the Hermitian FFT.</param>
            /// <param name="dim">The dimension along which to take the one dimensional Hermitian FFT.</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor ihfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_ihfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Reorders n-dimensional FFT data, as provided by fftn(), to have negative frequency terms first.
            /// This performs a periodic shift of n-dimensional data such that the origin(0, ..., 0) is moved to the center of the tensor.Specifically, to input.shape[dim] // 2 in each selected dimension.
            /// </summary>
            /// <param name="input">The tensor in FFT order</param>
            /// <param name="dim">The dimensions to rearrange. Only dimensions specified here will be rearranged, any other dimensions will be left in their original order. Default: All dimensions of input.</param>
            /// <returns></returns>
            public static Tensor fftshift(Tensor input, long[] dim = null)
            {
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* pDim = dim) {
                        var res = THSTensor_fftshift(input.Handle, (IntPtr)pDim, dlen);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Inverse of fftshift().
            /// </summary>
            /// <param name="input">The tensor in FFT order</param>
            /// <param name="dim">The dimensions to rearrange. Only dimensions specified here will be rearranged, any other dimensions will be left in their original order. Default: All dimensions of input.</param>
            public static Tensor ifftshift(Tensor input, long[] dim = null)
            {
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* pDim = dim) {
                        var res = THSTensor_ifftshift(input.Handle, (IntPtr)pDim, dlen);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
            /// </summary>
            /// <param name="n">The FFT length</param>
            /// <param name="d">The sampling length scale. </param>
            /// <param name="dtype">The desired data type of the returned tensor</param>
            /// <param name="device">the desired device of the returned tensor</param>
            /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
            public static Tensor fftfreq(long n, double d = 1.0, torch.ScalarType? dtype = null, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);
                if (!dtype.HasValue) {
                    // Determine the element type dynamically.
                    dtype = get_default_dtype();
                }

                var handle = THSTensor_fftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_fftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            /// Computes the sample frequencies for rfft() with a signal of size n.
            /// </summary>
            /// <param name="n">The FFT length</param>
            /// <param name="d">The sampling length scale. </param>
            /// <param name="dtype">The desired data type of the returned tensor</param>
            /// <param name="device">the desired device of the returned tensor</param>
            /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
            public static Tensor rfftfreq(long n, double d = 1.0, torch.ScalarType? dtype = null, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);
                if (!dtype.HasValue) {
                    // Determine the element type dynamically.
                    dtype = get_default_dtype();
                }

                var handle = THSTensor_rfftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_rfftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            /// Computes the 2-dimensional discrete Fourier transform of a Hermitian symmetric input signal.
            ///
            /// Equivalent to hfftn() but only transforms the last two dimensions by default.
            /// input is interpreted as a one-sided Hermitian signal in the time domain.
            /// By the Hermitian property, the Fourier transform will be real-valued.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor hfft2(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("hfft2() input should be at least 2D");
                if (dim == null) dim = new long[] { -2, -1 };
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_hfft2(input.Handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the 2-dimensional inverse discrete Fourier transform of a Hermitian symmetric input signal.
            ///
            /// Equivalent to hfftn() but only transforms the last two dimensions by default.
            /// input is interpreted as a one-sided Hermitian signal in the time domain.
            /// By the Hermitian property, the Fourier transform will be real-valued.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor ihfft2(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("ihfft2() input should be at least 2D");
                if (dim == null) dim = new long[] { -2, -1 };
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_ihfft2(input.Handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the n-dimensional discrete Fourier transform of a Herimitian symmetric input signal.
            ///
            /// input is interpreted as a one-sided Hermitian signal in the time domain.
            /// By the Hermitian property, the Fourier transform will be real-valued.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor hfftn(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("hfftn() input should be at least 2D");
                var slen = (s == null) ? 0 : s.Length;
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_hfftn(input.Handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the n-dimensional inverse discrete Fourier transform of a Herimitian symmetric input signal.
            ///
            /// input is interpreted as a one-sided Hermitian signal in the time domain.
            /// By the Hermitian property, the Fourier transform will be real-valued.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="s">
            /// Signal size in the transformed dimensions.
            /// If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT.
            /// If a length -1 is specified, no padding is done in that dimension.
            /// </param>
            /// <param name="dim">Dimensions to be transformed</param>
            /// <param name="norm">Normalization mode.</param>
            public static Tensor ihfftn(Tensor input, long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
            {
                if (input.Dimensions < 2) throw new ArgumentException("ihfftn() input should be at least 2D");
                var slen = (s == null) ? 0 : s.Length;
                var dlen = (dim == null) ? 0 : dim.Length;
                unsafe {
                    fixed (long* ps = s, pDim = dim) {
                        var res = THSTensor_ihfftn(input.Handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }
        }
    }
}
