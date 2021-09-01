// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class RNN : torch.nn.Module
        {
            internal RNN(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_RNN_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0, out IntPtr h_n);

            /// <summary>
            /// Applies a multi-layer Elman RNN with \tanhtanh or \text{ReLU}ReLU non-linearity to an input sequence.
            /// </summary>
            /// <param name="input">Tensor of shape (seq_len, batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (num_layers * num_directions, batch, hidden_size)containing the initial hidden state for each element in the batch.
            /// Defaults to 0 if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.</param>
            /// <returns></returns>
            public new (Tensor, Tensor) forward(Tensor input, Tensor h0 = null)
            {
                var res = THSNN_RNN_forward(handle, input.Handle, h0?.Handle ?? IntPtr.Zero, out IntPtr hN);
                if (res == IntPtr.Zero || hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(hN));
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public enum NonLinearities
            {
                ReLU = 0,
                Tanh = 1
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_RNN_ctor(long input_size, long hidden_size, long num_layers, long nonlinearity, bool bias, bool batchFirst, double dropout, bool bidirectional, out IntPtr pBoxedModule);

            /// <summary>
            /// Creates an Elman RNN module with tanh or ReLU non-linearity.
            /// </summary>
            /// <param name="inputSize">The number of expected features in the input x</param>
            /// <param name="hiddenSize">The number of features in the hidden state h</param>
            /// <param name="numLayers">Number of recurrent layers. Default: 1</param>
            /// <param name="nonLinearity">The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'</param>
            /// <param name="bias">If False, then the layer does not use bias weights b_ih and b_hh. Default: True</param>
            /// <param name="batchFirst">if true, then the input and output tensors are provided as (batch, seq, feature). Default: False</param>
            /// <param name="dropout">If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0</param>
            /// <param name="bidirectional">if true, becomes a bidirectional RNN. Default: False</param>
            /// <returns></returns>
            static public RNN RNN(long inputSize, long hiddenSize, long numLayers = 1, NonLinearities nonLinearity = nn.NonLinearities.Tanh, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false)
            {
                var res = THSNN_RNN_ctor(inputSize, hiddenSize, numLayers, (long)nonLinearity, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new RNN(res, boxedHandle);
            }
        }
    }
}
