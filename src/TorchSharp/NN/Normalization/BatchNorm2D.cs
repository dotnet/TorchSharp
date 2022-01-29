// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a BatchNorm2D module.
        /// </summary>
        public class BatchNorm2d : torch.nn.Module
        {
            internal BatchNorm2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_BatchNorm2d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.Dimensions != 4) throw new ArgumentException($"Invalid number of dimensions for BatchNorm argument: {tensor.Dimensions}");
                var res = THSNN_BatchNorm2d_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_BatchNorm2d_bias(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_BatchNorm2d_set_bias(torch.nn.Module.HType module, IntPtr bias);
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_BatchNorm2d_weight(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_BatchNorm2d_set_weight(torch.nn.Module.HType module, IntPtr weight);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_BatchNorm2d_reset_stats(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_BatchNorm2d_get_mean(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_BatchNorm2d_get_var(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_BatchNorm2d_set_mean(torch.nn.Module.HType module, IntPtr weight);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_BatchNorm2d_set_var(torch.nn.Module.HType module, IntPtr weight);

            public Tensor bias {
                get {
                    var res = THSNN_BatchNorm2d_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSNN_BatchNorm2d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Tensor weight {
                get {
                    var res = THSNN_BatchNorm2d_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSNN_BatchNorm2d_set_weight(handle, value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight", value);
                }
            }

            public Tensor running_mean {
                get {
                    var res = THSNN_BatchNorm2d_get_mean(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); return null; }
                    return new Tensor(res);
                }
                set {
                    THSNN_BatchNorm2d_set_mean(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Tensor running_var {
                get {
                    var res = THSNN_BatchNorm2d_get_var(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); return null; }
                    return new Tensor(res);
                }
                set {
                    THSNN_BatchNorm2d_set_var(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public void reset_running_stats()
            {
                THSNN_BatchNorm2d_reset_stats(handle);
                torch.CheckForErrors();
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_BatchNorm2d_ctor(long features, double eps, double momentum, bool affine, bool track_running_stats, out IntPtr pBoxedModule);

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
            static public BatchNorm2d BatchNorm2d(long features, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
            {
                unsafe {
                    var handle = THSNN_BatchNorm2d_ctor(features, eps, momentum, affine, track_running_stats, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new BatchNorm2d(handle, boxedHandle);
                }
            }
        }
    }
}
