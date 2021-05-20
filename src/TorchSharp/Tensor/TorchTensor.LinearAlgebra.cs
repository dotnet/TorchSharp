using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
namespace TorchSharp.Tensor
{
    public sealed partial class TorchTensor
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky(IntPtr input, bool upper);

        public TorchTensor cholesky(bool upper = false)
        {
            var res = THSTensor_cholesky(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, bool upper);

        public TorchTensor cholesky_inverse(bool upper = false)
        {
            var res = THSTensor_cholesky_inverse(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, bool upper);

        public TorchTensor cholesky_solve(TorchTensor input2, bool upper = false)
        {
            var res = THSTensor_cholesky_solve(handle, input2.Handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public TorchTensor cross(TorchScalar other, long dim)
        {
            var res = THSTensor_cross(handle, other.Handle, dim);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_matmul(IntPtr tensor, IntPtr target);

        public TorchTensor matmul(TorchTensor target)
        {
            var res = THSTensor_matmul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mm(IntPtr tensor, IntPtr target);

        public TorchTensor mm(TorchTensor target)
        {
            var res = THSTensor_mm(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mv(IntPtr tensor, IntPtr target);

        public TorchTensor mv(TorchTensor target)
        {
            var res = THSTensor_mv(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_vdot(IntPtr tensor, IntPtr target);

        public TorchTensor vdot(TorchTensor target)
        {
            if (shape.Length != 1 || target.shape.Length != 1 || shape[0] != target.shape[0]) throw new InvalidOperationException("vdot arguments must have the same shape.");
            var res = THSTensor_vdot(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


    }
}
