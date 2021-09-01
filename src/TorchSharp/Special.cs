// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torch
    {
        public static class special
        {
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_entr(IntPtr tensor);

            /// <summary>
            /// Computes the entropy on input, elementwise.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor entr(Tensor input)
            {
                var res = THSSpecial_entr(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_erf(IntPtr tensor);

            /// <summary>
            /// Computes the error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erf(Tensor input)
            {
                var res = THSSpecial_erf(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_erfc(IntPtr tensor);

            /// <summary>
            /// Computes the complementary error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erfc(Tensor input)
            {
                var res = THSSpecial_erfc(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_erfinv(IntPtr tensor);

            /// <summary>
            /// Computes the inverse error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erfinv(Tensor input)
            {
                var res = THSSpecial_erfinv(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_expit(IntPtr tensor);

            /// <summary>
            /// Computes the expit (also known as the logistic sigmoid function) of the elements of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor expit(Tensor input)
            {
                var res = THSSpecial_expit(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_expm1(IntPtr tensor);

            /// <summary>
            /// Computes the exponential of the elements minus 1 of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor expm1(Tensor input)
            {
                var res = THSSpecial_expm1(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_exp2(IntPtr tensor);

            /// <summary>
            /// Computes the base two exponential function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor exp2(Tensor input)
            {
                var res = THSSpecial_exp2(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_gammaln(IntPtr tensor);

            /// <summary>
            /// Computes the natural logarithm of the absolute value of the gamma function on input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor gammaln(Tensor input)
            {
                var res = THSSpecial_gammaln(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_i0e(IntPtr tensor);

            /// <summary>
            /// Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below) for each element of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor i0e(Tensor input)
            {
                var res = THSSpecial_i0e(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_logit(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the logit of the elements of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor logit(Tensor input)
            {
                var res = THSSpecial_logit(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSSpecial_xlog1py(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Computes input * log1p(other). Similar to SciPy’s scipy.special.xlog1py.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="other"></param>
            /// <returns></returns>
            public static Tensor xlog1py(Tensor input, Tensor other)
            {
                var res = THSSpecial_xlog1py(input.Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }
        }
    }
}
