// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class TransformerDecoder : Module
    {
        private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

        internal TransformerDecoder (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerDecoder_forward (Module.HType module, IntPtr tgt, IntPtr memory, IntPtr tgt_mask, IntPtr memory_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

        public TorchTensor forward (TorchTensor tgt, TorchTensor memory, TorchTensor? tgt_mask = null, TorchTensor? memory_mask = null, TorchTensor? tgt_key_padding_mask = null, TorchTensor? memory_key_padding_mask = null)
        {
            var res = THSNN_TransformerDecoder_forward(handle,
                tgt.Handle,
                memory.Handle,
                tgt_mask?.Handle ?? IntPtr.Zero,
                memory_mask?.Handle ?? IntPtr.Zero,
                tgt_key_padding_mask?.Handle ?? IntPtr.Zero,
                memory_key_padding_mask?.Handle ?? IntPtr.Zero);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerDecoder_ctor (Module.HType decoder_layer, long num_layers, out IntPtr pBoxedModule);

        static public TransformerDecoder TransformerDecoder (TransformerDecoderLayer decoder_layer, long num_layers)
        {
            var res = THSNN_TransformerDecoder_ctor (decoder_layer.handle, num_layers, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TransformerDecoder (res, boxedHandle);
        }
    }
}
