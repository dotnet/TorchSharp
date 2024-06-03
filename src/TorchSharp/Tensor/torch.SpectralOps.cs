// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable

using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#spectral-ops
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.stft
        /// <summary>
        /// Returns a tensor containing the result of Short-time Fourier transform (STFT).
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="n_fft">The size of Fourier transform</param>
        /// <param name="hop_length">The hop length</param>
        /// <param name="win_length">The window length</param>
        /// <param name="window">The window function</param>
        /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
        /// <param name="pad_mode">The padding mode used when center is true.</param>
        /// <param name="normalized">Whether the output is normalized, or not.</param>
        /// <param name="onesided">Whether the output is onesided or not.</param>
        /// <param name="return_complex">Whether a complex tensor is returned, or not.</param>
        /// <returns>A tensor containing the result of Short-time Fourier transform (STFT).</returns>
        public static Tensor stft(Tensor input, long n_fft, long hop_length = -1, long win_length = -1, Tensor? window = null, bool center = true, PaddingModes pad_mode = PaddingModes.Reflect, bool normalized = false, bool? onesided = null, bool? return_complex = null)
            => input.stft(n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex);

        // https://pytorch.org/docs/stable/generated/torch.istft
        /// <summary>
        /// Returns a tensor containing the result of Inverse Short-time Fourier transform.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="n_fft">The size of Fourier transform</param>
        /// <param name="hop_length">The hop length</param>
        /// <param name="win_length">The window length</param>
        /// <param name="window">The window function</param>
        /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
        /// <param name="normalized">Whether the output is normalized, or not.</param>
        /// <param name="onesided">Whether the output is onesided or not.</param>
        /// <param name="length">The length of the output tensor.</param>
        /// <param name="return_complex">Whether a complex tensor is returned, or not.</param>
        /// <returns>A tensor containing the result of Inverse Short-time Fourier transform</returns>
        public static Tensor istft(Tensor input, long n_fft, long hop_length = -1, long win_length = -1, Tensor? window = null, bool center = true, bool normalized = false, bool? onesided = null, long length = -1, bool return_complex = false)
            => input.istft(n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);

        // https://pytorch.org/docs/stable/generated/torch.bartlett_window
        /// <summary>
        /// Bartlett window function.
        /// </summary>
        public static Tensor bartlett_window(long len, bool periodic = true, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_bartlett_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_bartlett_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        // https://pytorch.org/docs/stable/generated/torch.blackman_window
        /// <summary>
        /// Blackman window function.
        /// </summary>
        public static Tensor blackman_window(long len, bool periodic = true, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_blackman_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_blackman_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        // https://pytorch.org/docs/stable/generated/torch.hamming_window

        /// <summary>
        /// Hamming window function.
        /// </summary>
        public static Tensor hamming_window(long len, bool periodic = true, float alpha = 0.54f, float beta = 0.46f, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_hamming_window(len, periodic, alpha, beta, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hamming_window(len, periodic, alpha, beta, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        // https://pytorch.org/docs/stable/generated/torch.hann_window
        /// <summary>
        /// Hann window function.
        /// </summary>
        public static Tensor hann_window(long len, bool periodic = true, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_hann_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hann_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        // https://pytorch.org/docs/stable/generated/torch.kaiser_window
        /// <summary>
        /// Computes the Kaiser window with window length window_length and shape parameter beta.
        /// </summary>
        public static Tensor kaiser_window(long len, bool periodic = true, float beta = 12.0f, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_kaiser_window(len, periodic, beta, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_kaiser_window(len, periodic, beta, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }
    }
}