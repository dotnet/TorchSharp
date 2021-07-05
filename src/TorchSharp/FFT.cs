using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;


namespace TorchSharp
{
    using static torch;

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
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fft(IntPtr tensor, long n, long dim, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ifft(IntPtr tensor, long n, long dim, sbyte norm);

            public static Tensor ifft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_ifft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ifft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ifftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_irfft(IntPtr tensor, long n, long dim, sbyte norm);

            public static Tensor irfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_irfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rfft(IntPtr tensor, long n, long dim, sbyte norm);

            public static Tensor rfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_rfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_irfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_irfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hfft(IntPtr tensor, long n, long dim, sbyte norm);

            public static Tensor hfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_hfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ihfft(IntPtr tensor, long n, long dim, sbyte norm);

            public static Tensor ihfft(Tensor input, long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
            {
                var res = THSTensor_ihfft(input.Handle, n, dim, (sbyte)norm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fftshift(IntPtr tensor, IntPtr dim, int dim_length);

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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ifftshift(IntPtr tensor, IntPtr dim, int dim_length);

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

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_fftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
            /// </summary>
            static public Tensor fftfreq(long n, double d = 1.0, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);
                if (!dtype.HasValue) {
                    // Determine the element type dynamically.
                    dtype = get_default_dtype();
                }

                var handle = THSTensor_fftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_fftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_rfftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Computes the sample frequencies for rfft() with a signal of size n.
            /// </summary>
            static public Tensor rfftfreq(long n, double d = 1.0, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);
                if (!dtype.HasValue) {
                    // Determine the element type dynamically.
                    dtype = get_default_dtype();
                }

                var handle = THSTensor_rfftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_rfftfreq(n, d, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }
    }
}
