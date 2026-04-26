// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using TorchSharp.Amp;
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

                return ReturnCheckForErrorsAutocast(THSNN_GroupNorm_forward(handle.DangerousGetHandle(), tensor.Handle), ScalarType.Float32);
                
            }

            public Parameter? bias
            {
                get => _bias;
                set
                {
                    _bias?.Dispose();
                    _bias = value?.DetachFromDisposeScope() as Parameter;
                    ConditionallyRegisterParameter(nameof(bias), _bias);
                }
            }

            public Parameter weight
            {
                get
                {
                    //May have problem with netstandard2.0?
                    return _weight!;
                }
                set
                {
                    if (value.Handle != _weight?.Handle)
                    {
                        _weight?.Dispose();
                        _weight = (value.DetachFromDisposeScope() as Parameter)!;
                        ConditionallyRegisterParameter(nameof(weight), _weight);
                    }
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
                /*unsafe {
                    var handle = THSNN_GroupNorm_ctor(num_groups, num_channels, eps, affine, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    handle= AutocastMode.AutoCast(handle, ScalarType.Float32);
                    return new GroupNorm(handle, boxedHandle).MoveModule<GroupNorm>(device, dtype);
                }*/
                return new GroupNorm(num_groups, num_channels, eps, affine, device, dtype);
            }
        }
    }
}
