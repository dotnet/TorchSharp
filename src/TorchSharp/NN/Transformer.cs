// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class Transformer : Module
    {
        public enum Activations
        {
            ReLU = 0,
            GELU = 1
        }

        private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

        internal Transformer (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Transformer_forward (Module.HType module, IntPtr src, IntPtr tgt, IntPtr src_mask, IntPtr tgt_mask, IntPtr memory_mask, IntPtr src_key_padding_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

        public TorchTensor forward (TorchTensor src, TorchTensor tgt, TorchTensor? src_mask = null, TorchTensor? tgt_mask = null, TorchTensor? memory_mask = null, TorchTensor? src_key_padding_mask = null, TorchTensor? tgt_key_padding_mask = null, TorchTensor? memory_key_padding_mask = null)
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Transformer_ctor (long d_model, long nhead, long num_encoder_layers, long num_decoder_layers, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

        static public Transformer Transformer (long d_model = 512, long nhead = 8, long num_encoder_layers = 6, long num_decoder_layers = 6, long dim_feedforward = 2048, double dropout = 0.1, Transformer.Activations activation = NN.Transformer.Activations.ReLU)
        {
            var res = THSNN_Transformer_ctor (d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, (long)activation, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Transformer (res, boxedHandle);
        }
    }
}
