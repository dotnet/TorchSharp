// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class TransformerEncoderLayer : Module
    {
        private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

        internal TransformerEncoderLayer (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerEncoderLayer_forward (Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

        public TorchTensor forward (TorchTensor src, TorchTensor? src_mask = null, TorchTensor? src_key_padding_mask = null)
        {
            var res = THSNN_TransformerEncoderLayer_forward(handle,
                src.Handle,
                src_mask?.Handle ?? IntPtr.Zero,
                src_key_padding_mask?.Handle ?? IntPtr.Zero);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerEncoderLayer_ctor (long d_model, long nhead, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

        static public TransformerEncoderLayer TransformerEncoderLayer (long d_model = 512, long nhead = 8, long dim_feedforward = 2048, double dropout = 0.1, Transformer.Activations activation = NN.Transformer.Activations.ReLU)
        {
            var res = THSNN_TransformerEncoderLayer_ctor (d_model, nhead, dim_feedforward, dropout, (long)activation, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TransformerEncoderLayer (res, boxedHandle);
        }
    }
}
