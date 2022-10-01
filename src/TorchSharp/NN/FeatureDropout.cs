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
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public class FeatureAlphaDropout : torch.nn.Module<Tensor, Tensor>
        {
            internal FeatureAlphaDropout(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_FeatureAlphaDropout_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_FeatureAlphaDropout_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_FeatureAlphaDropout_ctor(double p, out IntPtr pBoxedModule);

            /// <summary>
            /// Randomly masks out entire channels (a channel is a feature map, e.g. the j-th channel of the i-th sample in the batch input is a tensor input[i,j]) of the input tensor.
            /// Instead of setting activations to zero, as in regular Dropout, the activations are set to the negative saturation value of the SELU activation function.
            /// Each element will be masked independently on every forward call with probability p using samples from a Bernoulli distribution.The elements to be masked are
            /// randomized on every forward call, and scaled and shifted to maintain zero mean and unit variance.
            /// </summary>
            /// <param name="p">Dropout probability of a channel to be zeroed. Default: 0.5</param>
            static public FeatureAlphaDropout FeatureAlphaDropout(double p = 0.5)
            {
                var handle = THSNN_FeatureAlphaDropout_ctor(p, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new FeatureAlphaDropout(handle, boxedHandle);
            }

            public static partial class functional
            {
                [DllImport("LibTorchSharp")]
                extern static IntPtr THSNN_feature_alpha_dropout(IntPtr input, double p, [MarshalAs(UnmanagedType.U1)] bool training, [MarshalAs(UnmanagedType.U1)] bool inplace);

                /// <summary>
                /// Randomly masks out entire channels (a channel is a feature map, e.g. the j-th channel of the i-th sample in the batch input is a tensor input[i,j]) of the input tensor.
                /// Instead of setting activations to zero, as in regular Dropout, the activations are set to the negative saturation value of the SELU activation function.
                /// Each element will be masked independently on every forward call with probability p using samples from a Bernoulli distribution.The elements to be masked are
                /// randomized on every forward call, and scaled and shifted to maintain zero mean and unit variance.
                /// </summary>
                static public Tensor feature_alpha_dropout(Tensor input, double p = 0.5, bool training = false, bool inplace = false)
                {
                    var res = THSNN_feature_alpha_dropout(input.Handle, p, training, inplace);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
