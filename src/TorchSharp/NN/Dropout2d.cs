// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    using impl;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent a Dropout2d module.
        /// </summary>
        public class Dropout2d : torch.nn.Module
        {
            internal Dropout2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Dropout2d_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override TorchTensor forward(TorchTensor tensor)
            {
                var res = THSNN_Dropout2d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Dropout2d_ctor(double probability, bool inPlace, out IntPtr pBoxedModule);

            /// <summary>
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            static public Dropout2d Dropout2d(double probability = 0.5, bool inPlace = false)
            {
                var handle = THSNN_Dropout2d_ctor(probability, inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Dropout2d(handle, boxedHandle);
            }
        }

        public static partial class functional
        {
            /// <summary>
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="x">Input tensor</param>
            /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            static public TorchTensor Dropout2d(TorchTensor x, double probability = 0.5, bool inPlace = false)
            {
                using (var d = nn.Dropout2d(probability, inPlace)) {
                    return d.forward(x);
                }
            }
        }
    }
}
