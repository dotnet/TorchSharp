using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.Tensor
{
    // This file contains the mathematical operators on TorchTensor

    public enum FFTNormType
    {
        Backward = 0,
        Forward = 1,
        Ortho = 2
    }

    public sealed partial class TorchTensor
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fft(IntPtr tensor, long n, long dim, sbyte norm);

        public TorchTensor fft(long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
        {
            var res = THSTensor_fft(handle, n, dim, (sbyte)norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ifft(IntPtr tensor, long n, long dim, sbyte norm);

        public TorchTensor ifft(long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
        {
            var res = THSTensor_ifft(handle, n, dim, (sbyte)norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        public TorchTensor fft2(long [] s = null, long [] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            if (this.Dimensions < 2) throw new ArgumentException("fft2() input should be at least 2D");
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_fft2(handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        public TorchTensor fftn(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            var slen = (s == null) ? 0 : s.Length;
            var dlen = (dim == null) ? 0 : dim.Length;
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_fftn(handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ifftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        public TorchTensor ifftn(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            var slen = (s == null) ? 0 : s.Length;
            var dlen = (dim == null) ? 0 : dim.Length;
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_ifftn(handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ifft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        public TorchTensor ifft2(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            if (this.Dimensions < 2) throw new ArgumentException("ifft2() input should be at least 2D");
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_ifft2(handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_irfft(IntPtr tensor, long n, long dim, sbyte norm);

        public TorchTensor irfft(long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
        {
            var res = THSTensor_irfft(handle, n, dim, (sbyte)norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rfft(IntPtr tensor, long n, long dim, sbyte norm);

        public TorchTensor rfft(long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
        {
            var res = THSTensor_rfft(handle, n, dim, (sbyte)norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        public TorchTensor rfft2(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            if (this.Dimensions < 2) throw new ArgumentException("rfft2() input should be at least 2D");
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_rfft2(handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_irfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        public TorchTensor irfft2(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            if (this.Dimensions < 2) throw new ArgumentException("irfft2() input should be at least 2D");
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_irfft2(handle, (IntPtr)ps, (IntPtr)pDim, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        public TorchTensor rfftn(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            var slen = (s == null) ? 0 : s.Length;
            var dlen = (dim == null) ? 0 : dim.Length;
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_rfftn(handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_irfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        public TorchTensor irfftn(long[] s = null, long[] dim = null, FFTNormType norm = FFTNormType.Backward)
        {
            var slen = (s == null) ? 0 : s.Length;
            var dlen = (dim == null) ? 0 : dim.Length;
            unsafe {
                fixed (long* ps = s, pDim = dim) {
                    var res = THSTensor_irfftn(handle, (IntPtr)ps, slen, (IntPtr)pDim, dlen, (sbyte)norm);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hfft(IntPtr tensor, long n, long dim, sbyte norm);

        public TorchTensor hfft(long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
        {
            var res = THSTensor_hfft(handle, n, dim, (sbyte)norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ihfft(IntPtr tensor, long n, long dim, sbyte norm);

        public TorchTensor ihfft(long n = -1, long dim = -1, FFTNormType norm = FFTNormType.Backward)
        {
            var res = THSTensor_ihfft(handle, n, dim, (sbyte)norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fftshift(IntPtr tensor, IntPtr dim, int dim_length);

        public TorchTensor fftshift(long[] dim = null)
        {
            var dlen = (dim == null) ? 0 : dim.Length;
            unsafe {
                fixed (long* pDim = dim) {
                    var res = THSTensor_fftshift(handle, (IntPtr)pDim, dlen);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ifftshift(IntPtr tensor, IntPtr dim, int dim_length);

        public TorchTensor ifftshift(long[] dim = null)
        {
            var dlen = (dim == null) ? 0 : dim.Length;
            unsafe {
                fixed (long* pDim = dim) {
                    var res = THSTensor_ifftshift(handle, (IntPtr)pDim, dlen);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }
    }
}
