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
        /// This class is used to represent a SiLU module.
        /// </summary>
        public sealed class SiLU : torch.nn.Module<Tensor, Tensor>
        {
            internal SiLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_SiLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(SiLU).Name;
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
            /// Sigmoid-Weighted Linear Unit
            /// </summary>
            /// <returns></returns>
            /// <remarks>The native libreary does not take an 'inplace' option, even though the PyTorch documentation mentions the parameter.</remarks>
            public static SiLU SiLU()
            {
                var handle = THSNN_SiLU_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new SiLU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Sigmoid-Weighted Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: false</param>
                public static Tensor silu(Tensor x, bool inplace = false)
                {
                    return inplace ? x.silu_().alias() : x.silu();
                }

                [Obsolete("Incorrect name capitalization", false)]
                public static Tensor SiLU(Tensor x, bool inplace = false) => silu(x, inplace);
            }
        }
    }
}
