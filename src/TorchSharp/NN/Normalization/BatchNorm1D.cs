// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a BatchNorm1D module.
        /// </summary>
        public sealed class BatchNorm1d : torch.nn.Module<Tensor, Tensor>
        {
            internal BatchNorm1d(IntPtr handle, IntPtr boxedHandle,
                double eps, double momentum, bool affine, bool track_running_stats)
                : base(handle, boxedHandle)
            {
                this.eps = eps;
                this.momentum = momentum;
                this.affine = affine;
                this.track_running_stats = track_running_stats;
            }

            public long num_features {
                get {
                    var weight = this.weight;
                    Debug.Assert(weight is not null);
                    return weight.size(0);
                }
            }

            public double eps { get; }
            public double momentum { get; }
            public bool affine { get; }
            public bool track_running_stats { get; }

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.Dimensions < 2 || tensor.Dimensions > 3) throw new ArgumentException($"Invalid number of dimensions for BatchNorm argument: {tensor.Dimensions}");
                var res = THSNN_BatchNorm1d_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Parameter? bias {
                get {
                    var res = THSNN_BatchNorm1d_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    // Please ignore, for now, that the litorch call thinks you *can* set it to null.
                    if (value is null) throw new ArgumentNullException("bias cannot be set to 'null'");
                    THSNN_BatchNorm1d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Parameter? weight {
                get {
                    var res = THSNN_BatchNorm1d_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    // Please ignore, for now, that the litorch call thinks you *can* set it to null.
                    if (value is null) throw new ArgumentNullException("weight cannot be set to 'null'");
                    THSNN_BatchNorm1d_set_weight(handle, value is null ? IntPtr.Zero : value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight", value);
                }
            }

            public Tensor? running_mean {
                get {
                    var res = THSNN_BatchNorm1d_get_mean(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); return null; }
                    return new Tensor(res);
                }
                set {
                    // Please ignore, for now, that the litorch call thinks you *can* set it to null.
                    if (value is null) throw new ArgumentNullException("running_mean cannot be set to 'null'");
                    THSNN_BatchNorm1d_set_mean(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterBuffer("running_mean", value);
                }
            }

            public Tensor? running_var {
                get {
                    var res = THSNN_BatchNorm1d_get_var(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); return null; }
                    return new Tensor(res);
                }
                set {
                    // Please ignore, for now, that the litorch call thinks you *can* set it to null.
                    if (value is null) throw new ArgumentNullException("running_var cannot be set to 'null'");
                    THSNN_BatchNorm1d_set_var(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterBuffer("running_var", value);
                }
            }

            public Tensor? num_batches_tracked {
                get {
                    var res = THSNN_BatchNorm1d_get_batches(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); return null; }
                    return new Tensor(res);
                }
            }

            public void reset_running_stats()
            {
                THSNN_BatchNorm1d_reset_stats(handle);
                torch.CheckForErrors();
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputs with optional additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
            /// </summary>
            /// <param name="features">C from an expected input of size (N,C,L) or LL from input of size (N, L)</param>
            /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
            /// <param name="momentum">The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1</param>
            /// <param name="affine">A boolean value that when set to True, this module has learnable affine parameters. Default: true</param>
            /// <param name="track_running_stats">A boolean value that when set to True, this module tracks the running mean and variance, and when set to False,
            /// this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
            /// When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static BatchNorm1d BatchNorm1d(long features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true, Device? device = null, ScalarType? dtype = null)
            {
                unsafe {
                    var handle = THSNN_BatchNorm1d_ctor(features, eps, momentum, affine, track_running_stats, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new BatchNorm1d(handle, boxedHandle).MoveModule<BatchNorm1d>(device, dtype);
                }
            }
        }
    }
}
