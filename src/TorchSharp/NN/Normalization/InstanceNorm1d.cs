// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a InstanceNorm1D module.
    /// </summary>
    public class InstanceNorm1d : nn.Module
    {
        internal InstanceNorm1d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_InstanceNorm1d_forward (IntPtr module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            if (tensor.Dimensions < 2 || tensor.Dimensions > 3) throw new ArgumentException($"Invalid number of dimensions for InstanceNorm argument: {tensor.Dimensions}");
            var res = THSNN_InstanceNorm1d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_InstanceNorm1d_ctor (long features, double eps, double momentum, bool affine, bool track_running_stats, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies Instance Normalization over a 3D input (a mini-batch of 1D inputs with optional additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization.
        /// </summary>
        /// <param name="features">C from an expected input of size (N,C,L) or LL from input of size (N, L)</param>
        /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
        /// <param name="momentum">The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1</param>
        /// <param name="affine">A boolean value that when set to True, this module has learnable affine parameters. Default: true</param>
        /// <param name="track_running_stats">A boolean value that when set to True, this module tracks the running mean and variance, and when set to False,
        /// this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
        /// When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: true</param>
        /// <returns></returns>
        static public InstanceNorm1d InstanceNorm1d(long features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
        {
            unsafe {
                var handle = THSNN_InstanceNorm1d_ctor (features, eps, momentum, affine, track_running_stats, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new InstanceNorm1d (handle, boxedHandle);
            }
        }
    }

    public static partial class functional
    {
        /// <summary>
        /// Applies Instance Normalization over a 3D input (a mini-batch of 1D inputs with optional additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="features">C from an expected input of size (N,C,L) or LL from input of size (N, L)</param>
        /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
        /// <param name="momentum">The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1</param>
        /// <param name="affine">A boolean value that when set to True, this module has learnable affine parameters. Default: true</param>
        /// <param name="track_running_stats">A boolean value that when set to True, this module tracks the running mean and variance, and when set to False,
        /// this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
        /// When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: true</param>
        /// <returns></returns>
        static public TorchTensor InstanceNorm1d(TorchTensor x, long features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
        {
            using (var d =nn.InstanceNorm1d(features, eps, momentum, affine, track_running_stats)) {
                return d.forward (x);
            }
        }
    }
}
