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
        /// This class is used to represent a ConstantPad2d module.
        /// </summary>
        public class ConstantPad2d : torch.nn.Module
        {
            internal ConstantPad2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_ConstantPad2d_forward(torch.nn.Module.HType module, IntPtr tensor);

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ConstantPad2d_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_ConstantPad2d_ctor(double value, long padding, out IntPtr pBoxedModule);

            /// <summary>
            /// Pads the input tensor using replication of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <param name="value"></param>
            /// <returns></returns>
            static public ConstantPad2d ConstantPad2d(long padding, double value)
            {
                var handle = THSNN_ConstantPad2d_ctor(value, padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad2d(handle, boxedHandle);
            }
        }
    }
}
