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
        /// This class is used to represent a GroupNorm module.
        /// </summary>
        public sealed class GroupNorm : torch.nn.Module<Tensor, Tensor>
        {
            internal GroupNorm(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.Dimensions < 3) throw new ArgumentException($"Invalid number of dimensions for GroupNorm argument: {tensor.Dimensions}");
                var res = THSNN_GroupNorm_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Parameter? bias {
                get {
                    var res = THSNN_GroupNorm_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_GroupNorm_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Parameter? weight {
                get {
                    var res = THSNN_GroupNorm_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_GroupNorm_set_weight(handle, value is null ? IntPtr.Zero : value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight", value);
                }
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization
            /// </summary>
            /// <param name="num_groups">Number of groups to separate the channels into</param>
            /// <param name="num_channels">Number of channels expected in input</param>
            /// <param name="eps">A value added to the denominator for numerical stability.</param>
            /// <param name="affine">A boolean value that when set to true, this module has learnable per-channel affine parameters initialized to ones (for weights) and zeros (for biases).</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static GroupNorm GroupNorm(long num_groups, long num_channels, double eps = 1e-05, bool affine = true, Device? device = null, ScalarType? dtype = null)
            {
                unsafe {
                    var handle = THSNN_GroupNorm_ctor(num_groups, num_channels, eps, affine, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new GroupNorm(handle, boxedHandle).MoveModule<GroupNorm>(device, dtype);
                }
            }
        }
    }
}
