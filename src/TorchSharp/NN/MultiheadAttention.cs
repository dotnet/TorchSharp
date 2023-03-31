// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using static TorchSharp.PInvoke.LibTorchSharp;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class MultiheadAttention : torch.nn.Module<Tensor, Tensor, Tensor, Tensor?, bool, Tensor?, Tuple<Tensor,Tensor>>
        {
            internal MultiheadAttention(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Applies the MultiheadAttention function element-wise.
            /// </summary>
            /// <param name="query">map a query and a set of key-value pairs to an output. See “Attention Is All You Need” for more details</param>
            /// <param name="key"></param>
            /// <param name="value"></param>
            /// <param name="key_padding_mask">if provided, specified padding elements in the key will be ignored by the attention. When given a binary mask and a value is True, the corresponding value on the attention layer will be ignored. When given a byte mask and a value is non-zero, the corresponding value on the attention layer will be ignored</param>
            /// <param name="need_weights">output attn_output_weights</param>
            /// <param name="attn_mask">2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all the batches while a 3D mask allows to specify a different mask for the entries of each batch</param>
            /// <returns>attn_output, attn_ouput_weights</returns>

            public override Tuple<Tensor,Tensor> forward(Tensor query, Tensor key, Tensor value, Tensor? key_padding_mask, bool need_weights, Tensor? attn_mask)
            {
                THSNN_MultiheadAttention_forward(handle,
                    query.Handle,
                    key.Handle,
                    value.Handle,
                    key_padding_mask?.Handle ?? IntPtr.Zero,
                    need_weights,
                    attn_mask?.Handle ?? IntPtr.Zero,
                    out var res1,
                    out var res2);
                if (res1 == IntPtr.Zero || (need_weights && res2 == IntPtr.Zero)) { torch.CheckForErrors(); }
                return Tuple.Create(new Tensor(res1), new Tensor(res2));
            }

            public new Tuple<Tensor, Tensor> call(Tensor query, Tensor key, Tensor value, Tensor? key_padding_mask = null, bool need_weights = true, Tensor? attn_mask = null)
            {
                return base.call(query, key, value, key_padding_mask, need_weights, attn_mask);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Allows the model to jointly attend to information from different representation subspaces (based on the paper “Attention Is All You Need”).
            /// </summary>
            /// <param name="embedded_dim">total dimension of the model</param>
            /// <param name="num_heads">parallel attention heads</param>
            /// <param name="dropout">a Dropout layer on attn_output_weights. Default: 0.0</param>
            /// <param name="bias">add bias as module parameter. Default: true</param>
            /// <param name="add_bias_kv">add bias to the key and value sequences at dim=0</param>
            /// <param name="add_zero_attn">add a new batch of zeros to the key and value sequences at dim=1</param>
            /// <param name="kdim">total number of features in key</param>
            /// <param name="vdim">total number of features in value</param>
            /// <returns></returns>
            public static MultiheadAttention MultiheadAttention(long embedded_dim, long num_heads, double dropout = 0.0, bool bias = true, bool add_bias_kv = false, bool add_zero_attn = false, long? kdim=null, long? vdim=null)
            {
                var _kdim = kdim.HasValue ? kdim.Value : embedded_dim;
                var _vdim = vdim.HasValue ? vdim.Value : embedded_dim;
                var res = THSNN_MultiheadAttention_ctor(embedded_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, _kdim, _vdim, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new MultiheadAttention(res, boxedHandle);
            }
        }
    }
}
