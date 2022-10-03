// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent an unflattening operation.
        /// </summary>
        public class Unflatten : torch.nn.Module<Tensor, Tensor>
        {
            internal Unflatten(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Unflatten_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Unflatten_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_Unflatten_ctor(long dim, IntPtr shape, long shape_len, out IntPtr pBoxedModule);

            /// <summary>
            /// Unflattens a tensor dim expanding it to a desired shape. For use with Sequential.
            /// </summary>
            /// <param name="dim">Dimension to be unflattened</param>
            /// <param name="unflattenedSize">New shape of the unflattened dimension</param>
            /// <returns></returns>
            static public Unflatten Unflatten(long dim, long[] unflattenedSize)
            {
                unsafe {
                    fixed (long* pUnflattenedSize = unflattenedSize) {
                        var handle = THSNN_Unflatten_ctor(dim, (IntPtr)pUnflattenedSize, unflattenedSize.Length, out var boxedHandle);
                        if (handle == IntPtr.Zero) { CheckForErrors(); }
                        return new Unflatten(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
