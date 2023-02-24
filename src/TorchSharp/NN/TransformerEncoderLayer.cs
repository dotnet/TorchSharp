// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class TransformerEncoderLayer : torch.nn.Module<Tensor, Tensor>, torch.nn.IModule<Tensor, Tensor, Tensor>, torch.nn.IModule<Tensor, Tensor, Tensor, Tensor>
        {
            internal TransformerEncoderLayer(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Pass the input through the encoder layer.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="src_mask">The additive mask for the src sequence (optional).</param>
            /// <param name="src_key_padding_mask">The ByteTensor mask for src keys per batch (optional).</param>
            /// <returns></returns>
            public Tensor call(Tensor src, Tensor src_mask, Tensor src_key_padding_mask)
            {
                var res = THSNN_TransformerEncoderLayer_forward(handle,
                    src.Handle,
                    src_mask?.Handle ?? IntPtr.Zero,
                    src_key_padding_mask?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Pass the input through the encoder layer.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="src_mask">The additive mask for the src sequence (optional).</param>
            public Tensor call(Tensor src, Tensor src_mask)
            {
                var res = THSNN_TransformerEncoderLayer_forward(handle,
                    src.Handle,
                    src_mask?.Handle ?? IntPtr.Zero,
                    IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Pass the input through the encoder layer.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            public override Tensor forward(Tensor src)
            {
                var res = THSNN_TransformerEncoderLayer_forward(handle,
                    src.Handle,
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
            /// <summary>
            /// TransformerEncoderLayer is made up of self-attn and feedforward network. This standard encoder layer is based on the paper “Attention Is All You Need”.
            /// </summary>
            /// <param name="d_model">The number of expected features in the input (required).</param>
            /// <param name="nhead">The number of heads in the multiheadattention models (required).</param>
            /// <param name="dim_feedforward">The dimension of the feedforward network model (default=2048).</param>
            /// <param name="dropout">The dropout value (default=0.1).</param>
            /// <param name="activation">The activation function of intermediate layer, relu or gelu (default=relu).</param>
            /// <returns></returns>
            public static TransformerEncoderLayer TransformerEncoderLayer(long d_model = 512, long nhead = 8, long dim_feedforward = 2048, double dropout = 0.1, Activations activation = nn.Activations.ReLU)
            {
                var res = THSNN_TransformerEncoderLayer_ctor(d_model, nhead, dim_feedforward, dropout, (long)activation, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TransformerEncoderLayer(res, boxedHandle);
            }
        }
    }
}
