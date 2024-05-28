// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Threshold module.
        /// </summary>
        public sealed class Threshold : torch.nn.Module<Tensor, Tensor>
        {
            internal Threshold(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

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

           // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Threshold
            /// </summary>
            /// <param name="threshold">The value to threshold at</param>
            /// <param name="value">The value to replace with</param>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            public static Threshold Threshold(double threshold, double value, bool inplace = false)
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
                public static Tensor Threshold(Tensor x, double threshold, double value, bool inplace = false)
                {
                    using (var m = nn.Threshold(threshold, value, inplace)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
