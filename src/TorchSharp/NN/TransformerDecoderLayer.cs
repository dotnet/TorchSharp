// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class TransformerDecoderLayer : torch.nn.Module
        {
            private Tensor NullTensor = new Tensor(IntPtr.Zero);

            internal TransformerDecoderLayer(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_TransformerDecoderLayer_forward(torch.nn.Module.HType module, IntPtr tgt, IntPtr memory, IntPtr tgt_mask, IntPtr memory_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

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
            public Tensor forward(Tensor tgt, Tensor memory, Tensor tgt_mask, Tensor memory_mask = null, Tensor tgt_key_padding_mask = null, Tensor memory_key_padding_mask = null)
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

            /// <summary>
            /// Pass the inputs (and mask) through the decoder layer.
            /// </summary>
            /// <param name="tgt">The sequence to the decoder layer (required).</param>
            /// <param name="memory">The sequence from the last layer of the encoder (required).</param>
            public override Tensor forward(Tensor tgt, Tensor memory)
            {
                var res = THSNN_TransformerDecoderLayer_forward(handle,
                    tgt.Handle,
                    memory.Handle,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    IntPtr.Zero);
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
            private static extern IntPtr THSNN_TransformerDecoderLayer_ctor(long d_model, long nhead, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

            /// <summary>
            /// TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network. This standard decoder layer is based on the paper “Attention Is All You Need”.
            /// </summary>
            /// <param name="d_model">The number of expected features in the input (required).</param>
            /// <param name="nhead">The number of heads in the multiheadattention models (required).</param>
            /// <param name="dim_feedforward">The dimension of the feedforward network model (default=2048).</param>
            /// <param name="dropout">The dropout value (default=0.1).</param>
            /// <param name="activation">The activation function of intermediate layer, relu or gelu (default=relu).</param>
            /// <returns></returns>
            static public TransformerDecoderLayer TransformerDecoderLayer(long d_model = 512, long nhead = 8, long dim_feedforward = 2048, double dropout = 0.1, Activations activation = nn.Activations.ReLU)
            {
                var res = THSNN_TransformerDecoderLayer_ctor(d_model, nhead, dim_feedforward, dropout, (long)activation, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TransformerDecoderLayer(res, boxedHandle);
            }
        }
    }
}
