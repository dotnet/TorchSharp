// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Softmax module.
        /// </summary>
        public sealed class Softmax : torch.nn.Module<Tensor, Tensor>
        {
            internal Softmax(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softmax_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softmax).Name;
            }

           // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softmax
            /// </summary>
            /// <param name="dim">A dimension along which Softmax will be computed (so every slice along dim will sum to 1)</param>
            /// <returns></returns>
            public static Softmax Softmax(long dim)
            {
                var handle = THSNN_Softmax_ctor(dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softmax(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Computes the softmax function for the input tensor.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="dim">A dimension along which softmax will be computed.</param>
                /// <param name="dtype">The desired data type of returned tensor.</param>
                public static Tensor softmax(Tensor input, long dim, ScalarType? dtype = null) => torch.special.softmax(input, dim, dtype);
            }
        }
    }
}
