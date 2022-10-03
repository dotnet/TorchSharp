// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Identity : torch.nn.Module<Tensor, Tensor>
        {
            internal Identity(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Identity_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
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
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Identity_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// A placeholder identity operator.
            /// </summary>
            /// <returns>The same tensor as is input.</returns>
            static public Identity Identity()
            {
                var res = THSNN_Identity_ctor(out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Identity(res, boxedHandle);
            }
        }
    }
}
