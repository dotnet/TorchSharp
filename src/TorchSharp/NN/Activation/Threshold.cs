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
        /// This class is used to represent a Threshold module.
        /// </summary>
        public class Threshold : torch.nn.Module
        {
            internal Threshold(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Threshold_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Threshold_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Threshold).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Threshold_ctor(double threshold, double value, bool inplace, out IntPtr pBoxedModule);

            /// <summary>
            /// Threshold
            /// </summary>
            /// <param name="threshold">The value to threshold at</param>
            /// <param name="value">The value to replace with</param>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            static public Threshold Threshold(double threshold, double value, bool inplace = false)
            {
                var handle = THSNN_Threshold_ctor(threshold, value, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Threshold(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Thresholds each element of the input Tensor.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="threshold">The value to threshold at</param>
                /// <param name="value">The value to replace with</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <returns></returns>
                static public Tensor Threshold(Tensor x, double threshold, double value, bool inplace = false)
                {
                    using (var m = nn.Threshold(threshold, value, inplace)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
