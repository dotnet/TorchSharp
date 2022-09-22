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
        /// This class is used to represent a AdaptiveMaxPool1D module.
        /// </summary>
        public class AdaptiveMaxPool1d : torch.nn.Module
        {
            internal AdaptiveMaxPool1d(IntPtr handle, IntPtr boxedHandle, bool return_indices) : base(handle, boxedHandle)
            {
                _return_indices = return_indices;
            }

            private bool _return_indices;

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdaptiveMaxPool1d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveMaxPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdaptiveMaxPool1d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor tensor)
            {
                var res = THSNN_AdaptiveMaxPool1d_forward_with_indices(handle, tensor.Handle, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }

            public override object forward(object input)
            {
                var tensor = ExtractOneTensor(input);

                if (_return_indices) {
                    var res = THSNN_AdaptiveMaxPool1d_forward_with_indices(handle, tensor.Handle, out var indices);
                    if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (new Tensor(res), new Tensor(indices));
                } else {
                    var res = THSNN_AdaptiveMaxPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_AdaptiveMaxPool1d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
            /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size H.</param>
            /// <param name="return_indices">If true, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d()</param>
            /// <returns></returns>
            static public AdaptiveMaxPool1d AdaptiveMaxPool1d(long outputSize, bool return_indices = false)
            {
                unsafe {
                    fixed (long* pkernelSize = new long[] { outputSize }) {
                        var handle = THSNN_AdaptiveMaxPool1d_ctor((IntPtr)pkernelSize, 1, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AdaptiveMaxPool1d(handle, boxedHandle, return_indices);
                    }
                }
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
                /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="x"></param>
                /// <param name="outputSize">The target output size H.</param>
                /// <param name="return_indices">If true, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d()</param>
                /// <returns></returns>
                static public Tensor adaptive_max_pool1d(Tensor x, long outputSize, bool return_indices = false)
                {
                    using (var d = nn.AdaptiveMaxPool1d(outputSize, return_indices)) {
                        return d.forward(x);
                    }
                }
            }
        }
    }
}
