// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class TransformerDecoder : torch.nn.Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
        {
            internal TransformerDecoder(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Pass the inputs (and mask) through the decoder layers in turn.
            /// </summary>
            /// <param name="tgt">The sequence to the decoder layer (required).</param>
            /// <param name="memory">The sequence from the last layer of the encoder (required).</param>
            /// <param name="tgt_mask">The mask for the tgt sequence (optional).</param>
            /// <param name="memory_mask">The mask for the memory sequence (optional).</param>
            /// <param name="tgt_key_padding_mask">The mask for the tgt keys per batch (optional).</param>
            /// <param name="memory_key_padding_mask">The mask for the memory keys per batch (optional).</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tgt, Tensor memory, Tensor tgt_mask, Tensor memory_mask = null, Tensor tgt_key_padding_mask = null, Tensor memory_key_padding_mask = null)
            {
                var res = THSNN_TransformerDecoder_forward(handle,
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
            /// Pass the inputs (and mask) through the decoder layers in turn.
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
            /// TransformerDecoder is a stack of N decoder layers
            /// </summary>
            /// <param name="decoder_layer">An instance of the TransformerDecoderLayer class (required).</param>
            /// <param name="num_layers">The number of sub-decoder-layers in the decoder (required).</param>
            /// <returns></returns>
            public static TransformerDecoder TransformerDecoder(TransformerDecoderLayer decoder_layer, long num_layers)
            {
                var res = THSNN_TransformerDecoder_ctor(decoder_layer.handle, num_layers, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TransformerDecoder(res, boxedHandle);
            }
        }
    }
}
