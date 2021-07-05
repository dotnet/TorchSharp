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
        /// This class is used to represent a LocalResponseNorm module.
        /// </summary>
        public class LocalResponseNorm : torch.nn.Module
        {
            internal LocalResponseNorm(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LocalResponseNorm_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.Dimensions < 3) throw new ArgumentException($"Invalid number of dimensions for LocalResponseNorm argument: {tensor.Dimensions}");
                var res = THSNN_LocalResponseNorm_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            extern static IntPtr THSNN_LocalResponseNorm_ctor(long size, double alpha, double beta, double k, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies local response normalization over an input signal composed of several input planes, where channels occupy the second dimension. Applies normalization across channels.
            /// </summary>
            static public LocalResponseNorm LocalResponseNorm(long size, double alpha = 0.0001, double beta = 0.75, double k = 1.0)
            {
                unsafe {
                    var handle = THSNN_LocalResponseNorm_ctor(size, alpha, beta, k, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new LocalResponseNorm(handle, boxedHandle);
                }
            }
        }
    }
}
