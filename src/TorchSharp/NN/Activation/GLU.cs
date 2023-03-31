// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a GLU (gated linear unit) module.
        /// </summary>
        public sealed class GLU : torch.nn.Module<Tensor, Tensor>
        {
            internal GLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_GLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(GLU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Gated Linear Unit
            /// </summary>
            /// <param name="dim">the dimension on which to split the input. Default: -1</param>
            /// <returns></returns>
            public static GLU GLU(long dim = -1)
            {
                var handle = THSNN_GLU_ctor(dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new GLU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// The gated linear unit function.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="dim">the dimension on which to split the input. Default: -1</param>
                /// <returns></returns>
                public static Tensor glu(Tensor input, long dim = -1)
                {
                    using (var m = nn.GLU(dim)) {
                        return m.call(input);
                    }
                }
            }
        }
    }
}
