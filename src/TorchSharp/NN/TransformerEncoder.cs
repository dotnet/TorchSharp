// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class TransformerEncoder : Module
    {
        public enum Activations
        {
            ReLU = 0,
            GELU = 1
        }

        private TorchTensor NullTensor = new TorchTensor(IntPtr.Zero);

        internal TransformerEncoder (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerEncoder_forward(Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

        public TorchTensor forward(TorchTensor src, TorchTensor? src_mask = null, TorchTensor? src_key_padding_mask = null)
        {
            var res = THSNN_TransformerEncoder_forward(handle,
                src.Handle,
                src_mask?.Handle ?? IntPtr.Zero,
                src_key_padding_mask?.Handle ?? IntPtr.Zero);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_TransformerEncoder_ctor (Module.HType encoder_layer, long num_layers, out IntPtr pBoxedModule);

        static public TransformerEncoder TransformerEncoder (TransformerEncoderLayer encoder_layer, long num_layers)
        {
            var res = THSNN_TransformerEncoder_ctor (encoder_layer.handle, num_layers, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TransformerEncoder (res, boxedHandle);
        }
    }
}
