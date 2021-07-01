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
        public class Transformer : torch.nn.Module
        {
            private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

            internal Transformer(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Transformer_forward(torch.nn.Module.HType module, IntPtr src, IntPtr tgt, IntPtr src_mask, IntPtr tgt_mask, IntPtr memory_mask, IntPtr src_key_padding_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

            /// <summary>
            /// Take in and process masked source/target sequences.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="tgt">The sequence to the decoder (required).</param>
            /// <param name="src_mask">The additive mask for the src sequence (optional).</param>
            /// <param name="tgt_mask">The additive mask for the tgt sequence (optional).</param>
            /// <param name="memory_mask">The additive mask for the encoder output (optional).</param>
            /// <param name="src_key_padding_mask">The ByteTensor mask for src keys per batch (optional).</param>
            /// <param name="tgt_key_padding_mask">The ByteTensor mask for tgt keys per batch (optional).</param>
            /// <param name="memory_key_padding_mask">The ByteTensor mask for memory keys per batch (optional).</param>
            /// <returns></returns>
            public TorchTensor forward(TorchTensor src, TorchTensor tgt, TorchTensor? src_mask = null, TorchTensor? tgt_mask = null, TorchTensor? memory_mask = null, TorchTensor? src_key_padding_mask = null, TorchTensor? tgt_key_padding_mask = null, TorchTensor? memory_key_padding_mask = null)
            {
                var res = THSNN_Transformer_forward(handle,
                    src.Handle,
                    tgt.Handle,
                    src_mask?.Handle ?? IntPtr.Zero,
                    tgt_mask?.Handle ?? IntPtr.Zero,
                    memory_mask?.Handle ?? IntPtr.Zero,
                    src_key_padding_mask?.Handle ?? IntPtr.Zero,
                    tgt_key_padding_mask?.Handle ?? IntPtr.Zero,
                    memory_key_padding_mask?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public enum Activations
            {
                ReLU = 0,
                GELU = 1
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Transformer_ctor(long d_model, long nhead, long num_encoder_layers, long num_decoder_layers, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

            /// <summary>
            /// A transformer model. User is able to modify the attributes as needed. The architecture is based on the paper “Attention Is All You Need”.
            /// </summary>
            /// <param name="d_model">The number of expected features in the encoder/decoder inputs (default=512).</param>
            /// <param name="nhead">The number of heads in the multiheadattention models (default=8).</param>
            /// <param name="num_encoder_layers">The number of sub-encoder-layers in the encoder (default=6).</param>
            /// <param name="num_decoder_layers">The number of sub-decoder-layers in the decoder (default=6).</param>
            /// <param name="dim_feedforward">The dimension of the feedforward network model (default=2048).</param>
            /// <param name="dropout">The dropout value (default=0.1).</param>
            /// <param name="activation">The activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).</param>
            /// <returns></returns>
            static public Transformer Transformer(long d_model = 512, long nhead = 8, long num_encoder_layers = 6, long num_decoder_layers = 6, long dim_feedforward = 2048, double dropout = 0.1, Activations activation = nn.Activations.ReLU)
            {
                var res = THSNN_Transformer_ctor(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, (long)activation, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Transformer(res, boxedHandle);
            }
        }
    }
}
