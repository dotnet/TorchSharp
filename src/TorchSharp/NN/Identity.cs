// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Identity : torch.nn.Module<Tensor, Tensor>
        {
            internal Identity(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            protected override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Identity_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// A placeholder identity operator.
            /// </summary>
            /// <returns>The same tensor as is input.</returns>
            public static Identity Identity()
            {
                var res = THSNN_Identity_ctor(out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Identity(res, boxedHandle);
            }
        }
    }
}
