// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class TransformerEncoder : torch.nn.Module<Tensor, Tensor>
        {
            public enum Activations
            {
                ReLU = 0,
                GELU = 1
            }

            internal TransformerEncoder(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_TransformerEncoder_forward(torch.nn.Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

            /// <summary>
            /// Pass the input through the encoder layers in turn.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="src_mask">The additive mask for the src sequence (optional).</param>
            /// <param name="src_key_padding_mask">The ByteTensor mask for src keys per batch (optional).</param>
            /// <returns></returns>
            public Tensor forward(Tensor src, Tensor src_mask, Tensor src_key_padding_mask)
            {
                var res = THSNN_TransformerEncoder_forward(handle,
                    src.Handle,
                    src_mask?.Handle ?? IntPtr.Zero,
                    src_key_padding_mask?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Pass the input through the encoder layers in turn.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <param name="src_mask">The additive mask for the src sequence (optional).</param>
            /// <returns></returns>
            public Tensor forward(Tensor src, Tensor src_mask)
            {
                var res = THSNN_TransformerEncoder_forward(handle,
                    src.Handle,
                    src_mask?.Handle ?? IntPtr.Zero,
                    IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Pass the input through the encoder layers in turn.
            /// </summary>
            /// <param name="src">The sequence to the encoder (required).</param>
            /// <returns></returns>
            public override Tensor forward(Tensor src)
            {
                var res = THSNN_TransformerEncoder_forward(handle,
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
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_TransformerEncoder_ctor(torch.nn.Module.HType encoder_layer, long num_layers, out IntPtr pBoxedModule);

            /// <summary>
            /// TransformerEncoder is a stack of N encoder layers
            /// </summary>
            /// <param name="encoder_layer">An instance of the TransformerEncoderLayer class (required).</param>
            /// <param name="num_layers">The number of sub-encoder-layers in the encoder (required).</param>
            /// <returns></returns>
            static public TransformerEncoder TransformerEncoder(TransformerEncoderLayer encoder_layer, long num_layers)
            {
                var res = THSNN_TransformerEncoder_ctor(encoder_layer.handle, num_layers, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TransformerEncoder(res, boxedHandle);
            }
        }
    }
}
