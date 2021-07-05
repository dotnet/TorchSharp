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
        /// Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with zero mean and unit standard deviation,
        /// the output of Alpha Dropout maintains the original mean and standard deviation of the input.
        /// Alpha Dropout goes hand-in-hand with SELU activation function, which ensures that the outputs have zero mean and unit standard deviation.
        /// During training, it randomly masks some of the elements of the input tensor with probability p using samples from a bernoulli distribution.
        /// The elements to masked are randomized on every forward call, and scaled and shifted to maintain zero mean and unit standard deviation.
        /// During evaluation the module simply computes an identity function.
        /// </summary>
        public class AlphaDropout : torch.nn.Module
        {
            internal AlphaDropout(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AlphaDropout_forward(torch.nn.Module.HType module, IntPtr tensor);

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AlphaDropout_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_AlphaDropout_ctor(double probability, bool inPlace, out IntPtr pBoxedModule);

            /// <summary>
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j] ).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            static public AlphaDropout AlphaDropout(double probability = 0.5, bool inPlace = false)
            {
                var handle = THSNN_AlphaDropout_ctor(probability, inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AlphaDropout(handle, boxedHandle);
            }
        }

        public static partial class functional
        {
            /// <summary>
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j] ).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="x">Input tensor</param>
            /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            static public Tensor AlphaDropout(Tensor x, double probability = 0.5, bool inPlace = false)
            {
                using (var d = nn.AlphaDropout(probability, inPlace)) {
                    return d.forward(x);
                }
            }
        }
    }
}
