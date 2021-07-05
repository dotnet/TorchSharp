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
        /// This class is used to represent a dropout module.
        /// </summary>
        public class Dropout : torch.nn.Module
        {
            internal Dropout(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Dropout_forward(torch.nn.Module.HType module, IntPtr tensor);

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Dropout_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_Dropout_ctor(double probability, bool inPlace, out IntPtr pBoxedModule);

            /// <summary>
            /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
            /// Each channel will be zeroed out independently on every forward call.
            /// </summary>
            /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            static public Dropout Dropout(double probability = 0.5, bool inPlace = false)
            {
                var handle = THSNN_Dropout_ctor(probability, inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Dropout(handle, boxedHandle);
            }
        }

        public static partial class functional
        {
            /// <summary>
            /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
            /// Each channel will be zeroed out independently on every forward call.
            /// </summary>
            /// <param name="x">Input tensor</param>
            /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            static public Tensor Dropout(Tensor x, double probability = 0.5, bool inPlace = false)
            {
                using (var d = nn.Dropout(probability, inPlace)) {
                    return d.forward(x);
                }
            }
        }
    }
}
