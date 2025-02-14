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
        /// Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with zero mean and unit standard deviation,
        /// the output of Alpha Dropout maintains the original mean and standard deviation of the input.
        /// Alpha Dropout goes hand-in-hand with SELU activation function, which ensures that the outputs have zero mean and unit standard deviation.
        /// During training, it randomly masks some of the elements of the input tensor with probability p using samples from a bernoulli distribution.
        /// The elements to masked are randomized on every forward call, and scaled and shifted to maintain zero mean and unit standard deviation.
        /// During evaluation the module simply computes an identity function.
        /// </summary>
        public sealed class AlphaDropout : ParameterLessModule<Tensor, Tensor>
        {
            internal AlphaDropout(double p = 0.5, bool inplace = false) : base(nameof(Dropout1d))
            {
                this.p = p;
                this.inplace = inplace;
            }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.alpha_dropout(tensor, this.p, this.training, inplace);
            }

            public bool inplace { get; set; }
            public double p { get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j] ).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="p">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inplace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            public static AlphaDropout AlphaDropout(double p = 0.5, bool inplace = false)
            {
                return new AlphaDropout(p, inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j] ).
                /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
                /// </summary>
                /// <returns></returns>
                public static Tensor alpha_dropout(Tensor input, double p = 0.5, bool training = false, bool inplace = false)
                {
                    var res = THSNN_alpha_dropout(input.Handle, p, training, inplace);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
