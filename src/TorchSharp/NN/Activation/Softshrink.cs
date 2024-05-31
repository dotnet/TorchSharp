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
        /// This class is used to represent a Softshrink module.
        /// </summary>
        public sealed class Softshrink : torch.nn.Module<Tensor, Tensor>
        {
            internal Softshrink(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softshrink_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softshrink).Name;
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
            /// Softshrink
            /// </summary>
            /// <param name="lambda"> the λ value for the Softshrink formulation. Default: 0.5</param>
            /// <returns></returns>
            public static Softshrink Softshrink(double lambda = 0.5)
            {
                var handle = THSNN_Softshrink_ctor(lambda, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softshrink(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lambda">The λ value for the Softshrink formulation. Default: 0.5</param>
                /// <returns></returns>
                public static Tensor Softshrink(Tensor x, double lambda = 0.5)
                {
                    using (var m = nn.Softshrink(lambda)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
