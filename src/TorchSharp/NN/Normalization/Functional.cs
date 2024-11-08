// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class functional
            {
                /// <summary>
                /// Perform normalization of inputs over specified dimension.
                /// </summary>
                /// <param name="input">Input tensor of any shape.</param>
                /// <param name="p">the exponent value in the norm formulation</param>
                /// <param name="dim">the dimension to reduce</param>
                /// <param name="eps">small value to avoid division by zero</param>
                public static Tensor normalize(Tensor input, double p = 2.0, long dim = 1L, double eps = 1e-12)
                {
                    var res = THSNN_normalize(
                        input.Handle,
                        p, dim, eps);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies Batch Normalization for each channel across a batch of data.
                /// </summary>
                public static Tensor batch_norm(Tensor input, Tensor? running_mean, Tensor? running_var, Tensor? weight = null, Tensor? bias = null, bool training = false, double momentum = 0.1, double eps = 1e-5)
                {
                    var res = THSNN_batch_norm(
                        input.Handle,
                        running_mean is not null ? running_mean.Handle : IntPtr.Zero,
                        running_var is not null ? running_var.Handle : IntPtr.Zero,
                        weight is not null ? weight.Handle : IntPtr.Zero,
                        bias is not null ? bias.Handle : IntPtr.Zero,
                        training,
                        momentum, eps);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies Group Normalization for last certain number of dimensions.
                /// </summary>
                public static Tensor group_norm(Tensor input, long num_groups, Tensor? weight = null, Tensor? bias = null, double eps = 1e-5)
                {
                    var res = THSNN_group_norm(
                        input.Handle,
                        num_groups,
                        weight is not null ? weight.Handle : IntPtr.Zero,
                        bias is not null ? bias.Handle : IntPtr.Zero,
                        eps);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies Instance Normalization for each channel in each data sample in a batch.
                /// </summary>
                public static Tensor instance_norm(Tensor input, Tensor? running_mean = null, Tensor? running_var = null, Tensor? weight = null, Tensor? bias = null, bool use_input_stats = true, double momentum = 0.1, double eps = 1e-5)
                {
                    var res = THSNN_instance_norm(
                        input.Handle,
                        running_mean is not null ? running_mean.Handle : IntPtr.Zero,
                        running_var is not null ? running_var.Handle : IntPtr.Zero,
                        weight is not null ? weight.Handle : IntPtr.Zero,
                        bias is not null ? bias.Handle : IntPtr.Zero,
                        use_input_stats,
                        momentum, eps);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies Layer Normalization for last certain number of dimensions.
                /// </summary>
                public static Tensor layer_norm(Tensor input, long[] normalized_shape, Tensor? weight = null, Tensor? bias = null, double eps = 1e-5)
                {
                    IntPtr res;
                    unsafe {
                        fixed (long* normalized_shape_ptr = normalized_shape) {
                            res = THSNN_layer_norm(
                                input.Handle,
                                normalized_shape_ptr,
                                normalized_shape.LongLength,
                                weight is not null ? weight.Handle : IntPtr.Zero,
                                bias is not null ? bias.Handle : IntPtr.Zero,
                                eps);
                        }
                    }
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }

            }
        }
    }
}
