// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using System.Runtime.InteropServices;
    using Modules;

    namespace Modules
    {
        public sealed class Transformer : torch.nn.Module<Tensor, Tensor, Tensor>
        {
            internal Transformer(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

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
            public Tensor call(Tensor src, Tensor tgt, Tensor src_mask, Tensor? tgt_mask = null, Tensor? memory_mask = null, Tensor? src_key_padding_mask = null, Tensor? tgt_key_padding_mask = null, Tensor? memory_key_padding_mask = null)
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
                return new Tensor(res);
            }

            /// <summary>
            /// Take in and process masked source/target sequences.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="tgt">The sequence to the decoder (required).</param>
            public override Tensor forward(Tensor src, Tensor tgt)
            {
                var res = THSNN_Transformer_forward(handle,
                    src.Handle,
                    tgt.Handle,
                    IntPtr.Zero,
                    IntPtr.Zero,
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
            public enum Activations
            {
                ReLU = 0,
                GELU = 1
            }

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
            public static Transformer Transformer(long d_model = 512, long nhead = 8, long num_encoder_layers = 6, long num_decoder_layers = 6, long dim_feedforward = 2048, double dropout = 0.1, Activations activation = nn.Activations.ReLU)
            {
                var res = THSNN_Transformer_ctor(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, (long)activation, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Transformer(res, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
                /// </summary>
                /// <param name="query">Query tensor, shaped (N, ..., L, E)</param>
                /// <param name="key">Key tensor, shaped (N, ..., S, E)</param>
                /// <param name="value">Value tensor, shaped (N, ..., S, Ev)</param>
                /// <param name="attn_mask">
                /// Attention mask, shaped (N, ..., L, S).
                /// Two types of masks are supported:
                /// A boolean mask where a value of True indicates that the element should take part in attention.
                /// A float mask of the same type as query, key, value that is added to the attention score.
                /// </param>
                /// <param name="p">Dropout probability</param>
                /// <param name="is_causal">If true, assumes causal attention masking and errors if both attn_mask and is_causal are set.</param>
                /// <param name="scale">Scaling factor applied prior to softmax. If null, 1/sqrt(E) is used.</param>
                /// <param name="enable_gqa">If true, enable Group Query Attention</param>
                /// <returns></returns>
                public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask = null, double p = 0.0, [MarshalAs(UnmanagedType.U1)] bool is_causal = false, double? scale=null, bool enable_gqa=false)
                {
                    if (p < 0) throw new ArgumentException("Dropout probability must be greater than or equal to zero.");
                    if (is_causal && attn_mask is not null) throw new ArgumentException("Casual attention masking cannot pass a mask.");
                    if (query.dim() < 2 || key.dim() < 2 || value.dim() < 2) throw new ArgumentException("Query, key, and value must have at least 2 dimensions.");
                    if (!enable_gqa && (query.size(1) != key.size(1) || query.size(1) != value.size(1))) throw new InvalidOperationException("Query and key/value heads must be equal when Group Query Attention is not enabled.");

                    var _scale = scale.HasValue ? new double[] { scale.Value } : null;

                    unsafe {
                        fixed (double* scalePtr = _scale) {
                            var res = THSNN_scaled_dot_product_attention(query.Handle, key.Handle, value.Handle, attn_mask is null ? IntPtr.Zero : attn_mask.Handle, p, is_causal, (IntPtr)scalePtr, enable_gqa);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }
            }
        }
    }
}
