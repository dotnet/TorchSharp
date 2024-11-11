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
        /// This class is used to represent a Dropout2d module.
        /// </summary>
        public sealed class Dropout2d : ParameterLessModule<Tensor, Tensor>
        {
            internal Dropout2d(double p = 0.5, bool inplace = false) : base(nameof(Dropout2d))
            {
                this.p = p;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.dropout2d(input, this.p, this.training, this.inplace);
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
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="p">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inplace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            public static Dropout2d Dropout2d(double p = 0.5, bool inplace = false)
            {
                return new Dropout2d(p, inplace);
            }

            public static partial class functional
            {

                /// <summary>
                /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor).
                /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
                /// </summary>
                /// <returns></returns>
                public static Tensor dropout2d(Tensor input, double p = 0.5, bool training = true, bool inplace = false)
                {
                    var res = THSNN_dropout2d(input.Handle, p, training, inplace);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
