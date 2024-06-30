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
        /// This class is used to represent a Softsign module.
        /// </summary>
        public sealed class Softsign : torch.nn.Module<Tensor, Tensor>
        {
            internal Softsign(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softsign_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softsign).Name;
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
            /// Softsign
            /// </summary>
            /// <returns></returns>
            public static Softsign Softsign()
            {
                var handle = THSNN_Softsign_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softsign(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softsign
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor Softsign(Tensor x)
                {
                    using (var m = nn.Softsign()) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
