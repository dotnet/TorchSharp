// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a BatchNorm3D module.
        /// </summary>
        public sealed class BatchNorm3d : BatchNorm
        {
            internal BatchNorm3d(long num_features, 
                            double eps, 
                            double momentum, 
                            bool affine, 
                            bool track_running_stats, 
                            Device? device, 
                            ScalarType? dtype) : base(num_features, eps, momentum, affine, track_running_stats, device, dtype, nameof(BatchNorm1d))
            {
            }

            protected override void ValidateInputDimensions(Tensor input)
            {
                if (input.ndim != 5)
                    throw new ArgumentException($"expected 4D input, but got {input.ndim}D input.");
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
            /// </summary>
            /// <param name="num_features">C from an expected input of size (N,C,D,H,W)</param>
            /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
            /// <param name="momentum">The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1</param>
            /// <param name="affine">A boolean value that when set to True, this module has learnable affine parameters. Default: true</param>
            /// <param name="track_running_stats">A boolean value that when set to True, this module tracks the running mean and variance, and when set to False,
            /// this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
            /// When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static BatchNorm3d BatchNorm3d(long num_features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true, Device? device = null, ScalarType? dtype = null)
            {
                return new BatchNorm3d(num_features, eps, momentum, affine, track_running_stats, device, dtype);
            }
        }
    }
}
