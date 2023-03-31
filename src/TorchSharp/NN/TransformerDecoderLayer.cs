// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class TransformerDecoderLayer : torch.nn.Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>, torch.nn.IModule<Tensor, Tensor, Tensor>
        {
            internal TransformerDecoderLayer(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Pass the inputs (and mask) through the decoder layer.
            /// </summary>
            /// <param name="tgt">The sequence to the decoder layer (required).</param>
            /// <param name="memory">The sequence from the last layer of the encoder (required).</param>
            /// <param name="tgt_mask">The mask for the tgt sequence (optional).</param>
            /// <param name="memory_mask">The mask for the memory sequence (optional).</param>
            /// <param name="tgt_key_padding_mask">The mask for the tgt keys per batch (optional).</param>
            /// <param name="memory_key_padding_mask">The mask for the memory keys per batch (optional).</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tgt, Tensor memory, Tensor tgt_mask, Tensor memory_mask, Tensor tgt_key_padding_mask, Tensor memory_key_padding_mask)
            {
                var res = THSNN_TransformerDecoderLayer_forward(handle,
                    tgt.Handle,
                    memory.Handle,
                    tgt_mask?.Handle ?? IntPtr.Zero,
                    memory_mask?.Handle ?? IntPtr.Zero,
                    tgt_key_padding_mask?.Handle ?? IntPtr.Zero,
                    memory_key_padding_mask?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public new Tensor call(Tensor tgt, Tensor memory, Tensor tgt_mask, Tensor memory_mask = null, Tensor tgt_key_padding_mask = null, Tensor memory_key_padding_mask = null)
            {
                return base.call(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask);
            }

            /// <summary>
            /// Pass the inputs (and mask) through the decoder layer.
            /// </summary>
            /// <param name="tgt">The sequence to the decoder layer (required).</param>
            /// <param name="memory">The sequence from the last layer of the encoder (required).</param>
            public Tensor call(Tensor tgt, Tensor memory)
            {
                return base.call(tgt, memory, null, null, null, null);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network. This standard decoder layer is based on the paper “Attention Is All You Need”.
            /// </summary>
            /// <param name="d_model">The number of expected features in the input (required).</param>
            /// <param name="nhead">The number of heads in the multiheadattention models (required).</param>
            /// <param name="dim_feedforward">The dimension of the feedforward network model (default=2048).</param>
            /// <param name="dropout">The dropout value (default=0.1).</param>
            /// <param name="activation">The activation function of intermediate layer, relu or gelu (default=relu).</param>
            /// <returns></returns>
            public static TransformerDecoderLayer TransformerDecoderLayer(long d_model = 512, long nhead = 8, long dim_feedforward = 2048, double dropout = 0.1, Activations activation = nn.Activations.ReLU)
            {
                var res = THSNN_TransformerDecoderLayer_ctor(d_model, nhead, dim_feedforward, dropout, (long)activation, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TransformerDecoderLayer(res, boxedHandle);
            }
        }
    }
}
