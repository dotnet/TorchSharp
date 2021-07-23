// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a SELU module.
        /// </summary>
        public class SELU : torch.nn.Module
        {
            internal SELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_SELU_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_SELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(SELU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_SELU_ctor(bool inplace, out IntPtr pBoxedModule);

            /// <summary>
            /// Scaled Exponential Linear Unit
            /// </summary>
            /// <param name="inPlace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            static public SELU SELU(bool inPlace = false)
            {
                var handle = THSNN_SELU_ctor(inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new SELU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Scaled Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inPlace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                static public Tensor selu(Tensor x, bool inPlace = false)
                {
                    using (var m = nn.SELU(inPlace)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
