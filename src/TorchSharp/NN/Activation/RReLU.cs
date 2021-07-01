// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a RReLU module.
    /// </summary>
    public class RReLU : torch.nn.Module
    {
        internal RReLU (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_RReLU_forward (torch.nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_RReLU_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (RReLU).Name;
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_RReLU_ctor (double lower, double upper, bool inplace, out IntPtr pBoxedModule);

        /// <summary>
        /// Randomized Rectified Linear Unit
        /// </summary>
        /// <param name="lower">Lower bound of the uniform distribution. Default: 1/8</param>
        /// <param name="upper">Upper bound of the uniform distribution. Default: 1/3</param>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public RReLU RReLU(double lower = one_eighth, double upper = one_third, bool inPlace = false)
        {
            var handle = THSNN_RReLU_ctor (lower, upper, inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new RReLU (handle, boxedHandle);
        }

        private const double one_eighth = 1.0 / 8.0;
        private const double one_third  = 1.0 / 3.0;
    }
    public static partial class functional
    {
        /// <summary>
        /// Randomized Rectified Linear Unit
        /// </summary>
        /// <param name="x">The input tensor</param>
        /// <param name="lower">Lower bound of the uniform distribution. Default: 1/8</param>
        /// <param name="upper">Upper bound of the uniform distribution. Default: 1/3</param>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public TorchTensor RReLU(TorchTensor x, double lower, double upper, bool inPlace = false)
        {
            using (var m =nn.RReLU(lower, upper, inPlace)) {
                return m.forward (x);
            }
        }
    }

}
