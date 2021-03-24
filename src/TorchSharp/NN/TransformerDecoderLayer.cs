// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class TransformerDecoderLayer : Module
    {
        private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

        internal TransformerDecoderLayer (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerDecoderLayer_forward (Module.HType module, IntPtr tgt, IntPtr memory, IntPtr tgt_mask, IntPtr memory_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

        public TorchTensor forward (TorchTensor tgt, TorchTensor memory, TorchTensor? tgt_mask = null, TorchTensor? memory_mask = null, TorchTensor? tgt_key_padding_mask = null, TorchTensor? memory_key_padding_mask = null)
        {
            var res = THSNN_TransformerDecoderLayer_forward(handle,
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
        private static extern IntPtr THSNN_TransformerDecoderLayer_ctor (long d_model, long nhead, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

        static public TransformerDecoderLayer TransformerDecoderLayer (long d_model = 512, long nhead = 8, long dim_feedforward = 2048, double dropout = 0.1, Transformer.Activations activation = NN.Transformer.Activations.ReLU)
        {
            var res = THSNN_TransformerDecoderLayer_ctor (d_model, nhead, dim_feedforward, dropout, (long)activation, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TransformerDecoderLayer (res, boxedHandle);
        }
    }
}
