// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;
using static TorchSharp.torch.nn;


#nullable enable
namespace TorchSharp
{
    public class LSTM : torch.nn.Module
    {
        internal LSTM (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        public new static LSTM Load (String modelPath)
        {
            var res = Module.Load (modelPath);
            return new LSTM (res.handle.DangerousGetHandle(), IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_LSTM_forward (torch.nn.Module.HType module, IntPtr input, IntPtr h_0, IntPtr c_0, out IntPtr h_n, out IntPtr c_n);

        /// <summary>
        /// Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        /// </summary>
        /// <param name="input">Tensor of shape (seq_len, batch, input_size) containing the features of the input sequence.</param>
        /// <param name="h0_c0">Tensors of shape (num_layers * num_directions, batch, hidden_size) containing the initial hidden and cell state for each element in the batch</param>
        /// <returns></returns>
        public (TorchTensor,TorchTensor,TorchTensor) forward (TorchTensor input, (TorchTensor, TorchTensor)? h0_c0 = null)
        {
            var res = THSNN_LSTM_forward (handle, input.Handle, h0_c0?.Item1.Handle ?? IntPtr.Zero, h0_c0?.Item2.Handle ?? IntPtr.Zero, out IntPtr hN, out IntPtr cN);
            if (res == IntPtr.Zero || hN == IntPtr.Zero || cN == IntPtr.Zero) { torch.CheckForErrors(); }
            return (new TorchTensor (res), new TorchTensor(hN), new TorchTensor(cN));
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_LSTM_ctor (long input_size, long hidden_size, long num_layers, bool bias, bool batchFirst, double dropout, bool bidirectional, out IntPtr pBoxedModule);

        /// <summary>
        /// Creates a multi-layer long short-term memory (LSTM) RNN module.
        /// </summary>
        /// <param name="inputSize">The number of expected features in the input x</param>
        /// <param name="hiddenSize">The number of features in the hidden state h</param>
        /// <param name="numLayers">Number of recurrent layers. Default: 1</param>
        /// <param name="bias">If False, then the layer does not use bias weights b_ih and b_hh. Default: True</param>
        /// <param name="batchFirst">if true, then the input and output tensors are provided as (batch, seq, feature). Default: False</param>
        /// <param name="dropout">If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0</param>
        /// <param name="bidirectional">if true, becomes a bidirectional RNN. Default: False</param>
        /// <returns></returns>
        static public LSTM LSTM (long inputSize, long hiddenSize, long numLayers = 1, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false)
        {
            var res = THSNN_LSTM_ctor(inputSize, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new LSTM (res, boxedHandle);
        }
    }
}
