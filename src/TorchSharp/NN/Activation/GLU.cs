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
        /// This class is used to represent a GLU (gated linear unit) module.
        /// </summary>
        public class GLU : torch.nn.Module
        {
            internal GLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_GLU_forward(torch.nn.Module.HType module, IntPtr tensor);

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
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_GLU_ctor(long dim, out IntPtr pBoxedModule);

            /// <summary>
            /// Gated Linear Unit
            /// </summary>
            /// <param name="dim">the dimension on which to split the input. Default: -1</param>
            /// <returns></returns>
            static public GLU GLU(int dim = -1)
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
                /// <param name="dim">the dimension on which to split the input. Default: -1</param>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor GLU(Tensor x, int dim = -1)
                {
                    using (var m = nn.GLU(dim)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
