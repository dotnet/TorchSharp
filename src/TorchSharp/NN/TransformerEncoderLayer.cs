// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp
{
    using impl;

    namespace impl
    {
        public class TransformerEncoderLayer : torch.nn.Module
        {
            private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

            internal TransformerEncoderLayer(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_TransformerEncoderLayer_forward(torch.nn.Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

            /// <summary>
            /// Pass the input through the encoder layer.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="src_mask">The additive mask for the src sequence (optional).</param>
            /// <param name="src_key_padding_mask">The ByteTensor mask for src keys per batch (optional).</param>
            /// <returns></returns>
            public TorchTensor forward(TorchTensor src, TorchTensor? src_mask = null, TorchTensor? src_key_padding_mask = null)
            {
                var res = THSNN_TransformerEncoderLayer_forward(handle,
                    src.Handle,
                    src_mask?.Handle ?? IntPtr.Zero,
                    src_key_padding_mask?.Handle ?? IntPtr.Zero);
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
            private static extern IntPtr THSNN_TransformerEncoderLayer_ctor(long d_model, long nhead, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

            /// <summary>
            /// TransformerEncoderLayer is made up of self-attn and feedforward network. This standard encoder layer is based on the paper “Attention Is All You Need”. 
            /// </summary>
            /// <param name="d_model">The number of expected features in the input (required).</param>
            /// <param name="nhead">The number of heads in the multiheadattention models (required).</param>
            /// <param name="dim_feedforward">The dimension of the feedforward network model (default=2048).</param>
            /// <param name="dropout">The dropout value (default=0.1).</param>
            /// <param name="activation">The activation function of intermediate layer, relu or gelu (default=relu).</param>
            /// <returns></returns>
            static public TransformerEncoderLayer TransformerEncoderLayer(long d_model = 512, long nhead = 8, long dim_feedforward = 2048, double dropout = 0.1, Activations activation = nn.Activations.ReLU)
            {
                var res = THSNN_TransformerEncoderLayer_ctor(d_model, nhead, dim_feedforward, dropout, (long)activation, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TransformerEncoderLayer(res, boxedHandle);
            }
        }
    }
}
