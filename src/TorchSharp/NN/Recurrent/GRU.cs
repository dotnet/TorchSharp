// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class GRU : Module
    {
        internal GRU (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_GRU_forward (Module.HType module, IntPtr input, IntPtr h_0, out IntPtr h_n);

        /// <summary>
        /// Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        /// </summary>
        /// <param name="input">Tensor of shape (seq_len, batch, input_size) containing the features of the input sequence.</param>
        /// <param name="h0">Tensor of shape (num_layers * num_directions, batch, hidden_size)containing the initial hidden state for each element in the batch.
        /// Defaults to 0 if not provided. If the GRU is bidirectional, num_directions should be 2, else it should be 1.</param>
        /// <returns></returns>
        /// <returns></returns>
        public (TorchTensor,TorchTensor) forward (TorchTensor input, TorchTensor? h0 = null)
        {
            var res = THSNN_GRU_forward (handle, input.Handle, h0?.Handle ?? IntPtr.Zero, out IntPtr hN);
            if (res == IntPtr.Zero || hN == IntPtr.Zero) { torch.CheckForErrors(); }
            return (new TorchTensor (res), new TorchTensor(hN));
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_GRU_ctor (long input_size, long hidden_size, long num_layers, bool bias, bool batchFirst, double dropout, bool bidirectional, out IntPtr pBoxedModule);

        /// <summary>
        /// Creates multi-layer gated recurrent unit (GRU) RNN module.
        /// </summary>
        /// <param name="inputSize">The number of expected features in the input x</param>
        /// <param name="hiddenSize">The number of features in the hidden state h</param>
        /// <param name="numLayers">Number of recurrent layers. Default: 1</param>
        /// <param name="bias">If False, then the layer does not use bias weights b_ih and b_hh. Default: True</param>
        /// <param name="batchFirst">if true, then the input and output tensors are provided as (batch, seq, feature). Default: False</param>
        /// <param name="dropout">If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0</param>
        /// <param name="bidirectional">if true, becomes a bidirectional RNN. Default: False</param>
        /// <returns></returns>
        static public GRU GRU (long inputSize, long hiddenSize, long numLayers = 1, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false)
        {
            var res = THSNN_GRU_ctor(inputSize, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new GRU (res, boxedHandle);
        }
    }
}
