// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a BatchNorm2D module.
    /// </summary>
    public class BatchNorm2D : Module
    {
        internal BatchNorm2D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_BatchNorm2d_forward (IntPtr module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            if (tensor.Dimensions != 4) throw new ArgumentException($"Invalid number of dimensions for BatchNorm argument: {tensor.Dimensions}");
            var res = THSNN_BatchNorm2d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_BatchNorm2d_ctor (long features, double eps, double momentum, bool affine, bool track_running_stats, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
        /// </summary>
        /// <param name="features">C from an expected input of size (N,C,H,W)</param>
        /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
        /// <param name="momentum">The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1</param>
        /// <param name="affine">A boolean value that when set to True, this module has learnable affine parameters. Default: true</param>
        /// <param name="track_running_stats">A boolean value that when set to True, this module tracks the running mean and variance, and when set to False,
        /// this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
        /// When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: true</param>
        /// <returns></returns>
        static public BatchNorm2D BatchNorm2D (long features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
        {
            unsafe {
                var handle = THSNN_BatchNorm2d_ctor (features, eps, momentum, affine, track_running_stats, out var boxedHandle);
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new BatchNorm2D (handle, boxedHandle);
            }
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="features">C from an expected input of size (N,C,H,W)</param>
        /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
        /// <param name="momentum">The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1</param>
        /// <param name="affine">A boolean value that when set to True, this module has learnable affine parameters. Default: true</param>
        /// <param name="track_running_stats">A boolean value that when set to True, this module tracks the running mean and variance, and when set to False,
        /// this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
        /// When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: true</param>
        /// <returns></returns>

        static public TorchTensor BatchNorm2D (TorchTensor x, long features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
        {
            using (var d = Modules.BatchNorm2D (features, eps, momentum, affine, track_running_stats)) {
                return d.forward (x);
            }
        }
    }
}
