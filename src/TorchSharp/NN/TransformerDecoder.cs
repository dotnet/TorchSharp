// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp
{
    public class TransformerDecoder : nn.Module
    {
        private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

        internal TransformerDecoder (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerDecoder_forward (nn.Module.HType module, IntPtr tgt, IntPtr memory, IntPtr tgt_mask, IntPtr memory_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

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
        public TorchTensor forward (TorchTensor tgt, TorchTensor memory, TorchTensor? tgt_mask = null, TorchTensor? memory_mask = null, TorchTensor? tgt_key_padding_mask = null, TorchTensor? memory_key_padding_mask = null)
        {
            var res = THSNN_TransformerDecoder_forward(handle,
                tgt.Handle,
                memory.Handle,
                tgt_mask?.Handle ?? IntPtr.Zero,
                memory_mask?.Handle ?? IntPtr.Zero,
                tgt_key_padding_mask?.Handle ?? IntPtr.Zero,
                memory_key_padding_mask?.Handle ?? IntPtr.Zero);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerDecoder_ctor (Module.HType decoder_layer, long num_layers, out IntPtr pBoxedModule);

        /// <summary>
        /// TransformerDecoder is a stack of N decoder layers
        /// </summary>
        /// <param name="decoder_layer">An instance of the TransformerDecoderLayer class (required).</param>
        /// <param name="num_layers">The number of sub-decoder-layers in the decoder (required).</param>
        /// <returns></returns>
        static public TransformerDecoder TransformerDecoder (TransformerDecoderLayer decoder_layer, long num_layers)
        {
            var res = THSNN_TransformerDecoder_ctor (decoder_layer.handle, num_layers, out var boxedHandle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TransformerDecoder (res, boxedHandle);
        }
    }
}
