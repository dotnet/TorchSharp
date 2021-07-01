// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using impl;

    namespace impl
    {

        /// <summary>
        /// This class is used to represent a GroupNorm module.
        /// </summary>
        public class GroupNorm : torch.nn.Module
        {
            internal GroupNorm(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_GroupNorm_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.Dimensions < 3) throw new ArgumentException($"Invalid number of dimensions for GroupNorm argument: {tensor.Dimensions}");
                var res = THSNN_GroupNorm_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_GroupNorm_ctor(long num_groups, long num_channels, double eps, bool affine, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization
            /// </summary>
            /// <returns></returns>
            static public GroupNorm GroupNorm(long numGroups, long numChannels, double eps = 1e-05, bool affine = true)
            {
                unsafe {
                    var handle = THSNN_GroupNorm_ctor(numGroups, numChannels, eps, affine, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new GroupNorm(handle, boxedHandle);
                }
            }
        }

        public static partial class functional
        {
            /// <summary>
            /// Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization
            /// </summary>
            static public Tensor GroupNorm(Tensor x, long numGroups, long numChannels, double eps = 1e-05, bool affine = true)
            {
                using (var d = nn.GroupNorm(numGroups, numChannels, eps, affine)) {
                    return d.forward(x);
                }
            }
        }
    }
}
