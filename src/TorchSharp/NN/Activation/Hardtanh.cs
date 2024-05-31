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
        /// This class is used to represent a Hardtanh module.
        /// </summary>
        public sealed class Hardtanh : torch.nn.Module<Tensor, Tensor>
        {
            internal Hardtanh(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Hardtanh_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Hardtanh).Name;
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
            /// Hardtanh
            /// </summary>
            /// <param name="min_val">Minimum value of the linear region range.</param>
            /// <param name="max_val">Maximum value of the linear region range.</param>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            public static Hardtanh Hardtanh(double min_val = -1.0, double max_val = 1.0, bool inplace = false)
            {
                var handle = THSNN_Hardtanh_ctor(min_val, max_val, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Hardtanh(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Hardtanh
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="min_val">Minimum value of the linear region range.</param>
                /// <param name="max_val">Maximum value of the linear region range.</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <returns></returns>
                public static Tensor Hardtanh(Tensor x, double min_val = -1.0, double max_val = 1.0, bool inplace = false)
                {
                    return inplace ? x.hardtanh_(min_val, max_val).alias() : x.hardtanh(min_val, max_val);
                }
            }
        }
    }
}
