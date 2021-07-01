// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using impl;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent a ConstantPad3d module.
        /// </summary>
        public class ConstantPad3d : torch.nn.Module
        {
            internal ConstantPad3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_ConstantPad3d_forward(torch.nn.Module.HType module, IntPtr tensor);

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ConstantPad3d_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_ConstantPad3d_ctor(double value, long padding, out IntPtr pBoxedModule);

            /// <summary>
            /// Pads the input tensor using replication of the input boundary.
            /// </summary>
            /// <param name="value"></param>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            static public ConstantPad3d ConstantPad3d(long padding, double value)
            {
                var handle = THSNN_ConstantPad3d_ctor(value, padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad3d(handle, boxedHandle);
            }
        }

        public static partial class functional
        {
            /// <summary>
            /// Pads the input tensor using replication of the input boundary.
            /// </summary>
            /// <param name="x">Input tensor</param>
            /// <param name="padding">The size of the padding: (padding_left , padding_right, padding_top, padding_bottom, padding_front, padding_back)</param>
            /// <param name="value"></param>
            /// <returns></returns>
            static public Tensor ConstantPad3d(Tensor x, long padding, double value)
            {
                using (var d = nn.ConstantPad3d(padding, value)) {
                    return d.forward(x);
                }
            }
        }
    }
}
