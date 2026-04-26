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
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public sealed class FeatureAlphaDropout : ParameterLessModule<Tensor, Tensor>
        {
            internal FeatureAlphaDropout(double p = 0.5, bool inplace = false) : base(nameof(FeatureAlphaDropout))
            {
                this.p = p;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.feature_alpha_dropout(input, this.p, this.training, this.inplace);
            }

            public bool inplace { get; set; }
            public double p { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Randomly masks out entire channels (a channel is a feature map, e.g. the j-th channel of the i-th sample in the batch input is a tensor input[i,j]) of the input tensor.
            /// Instead of setting activations to zero, as in regular Dropout, the activations are set to the negative saturation value of the SELU activation function.
            /// Each element will be masked independently on every forward call with probability p using samples from a Bernoulli distribution.The elements to be masked are
            /// randomized on every forward call, and scaled and shifted to maintain zero mean and unit variance.
            /// </summary>
            /// <param name="p">Dropout probability of a channel to be zeroed. Default: 0.5</param>
            /// <param name="inplace">If set to true, will do this operation in-place. Default: false</param>
            public static FeatureAlphaDropout FeatureAlphaDropout(double p, bool inplace)
            {
                return new FeatureAlphaDropout(p, inplace);
            }

            /// <summary>
            /// Randomly masks out entire channels (a channel is a feature map, e.g. the j-th channel of the i-th sample in the batch input is a tensor input[i,j]) of the input tensor.
            /// Instead of setting activations to zero, as in regular Dropout, the activations are set to the negative saturation value of the SELU activation function.
            /// Each element will be masked independently on every forward call with probability p using samples from a Bernoulli distribution.The elements to be masked are
            /// randomized on every forward call, and scaled and shifted to maintain zero mean and unit variance.
            /// </summary>
            /// <param name="p">Dropout probability of a channel to be zeroed. Default: 0.5</param>
            public static FeatureAlphaDropout FeatureAlphaDropout(double p = 0.5)
            {
                return new FeatureAlphaDropout(p, false);
            }

            public static partial class functional
            {
                /// <summary>
                /// Randomly masks out entire channels (a channel is a feature map, e.g. the j-th channel of the i-th sample in the batch input is a tensor input[i,j]) of the input tensor.
                /// Instead of setting activations to zero, as in regular Dropout, the activations are set to the negative saturation value of the SELU activation function.
                /// Each element will be masked independently on every forward call with probability p using samples from a Bernoulli distribution.The elements to be masked are
                /// randomized on every forward call, and scaled and shifted to maintain zero mean and unit variance.
                /// </summary>
                public static Tensor feature_alpha_dropout(Tensor input, double p = 0.5, bool training = false, bool inplace = false)
                {
                    var res = THSNN_feature_alpha_dropout(input.Handle, p, training, inplace);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
