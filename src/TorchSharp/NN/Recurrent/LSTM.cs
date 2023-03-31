// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class LSTM : torch.nn.Module<Tensor, (Tensor, Tensor)?, (Tensor, Tensor, Tensor)>
        {
            internal LSTM(IntPtr handle, IntPtr boxedHandle, long hiddenSize, long numLayers, bool batchFirst, bool bidirectional) : base(handle, boxedHandle)
            {
                _hidden_size = hiddenSize;
                _num_layers = numLayers;
                _bidirectional = bidirectional;
                _batch_first = batchFirst;
            }

            private long _hidden_size;
            private long _num_layers;
            private bool _bidirectional;
            private bool _batch_first;

            /// <summary>
            /// Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
            /// </summary>
            /// <param name="input">Tensor of shape (seq_len, batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0_c0">Tensors of shape (num_layers * num_directions, batch, hidden_size) containing the initial hidden and cell state for each element in the batch</param>
            /// <returns></returns>
            public override (Tensor, Tensor, Tensor) forward(Tensor input, (Tensor, Tensor)? h0_c0 = null)
            {
                Tensor c0, h0;

                if (h0_c0 == null) {
                    var N = _batch_first ? input.shape[0] : input.shape[1];
                    var D = _bidirectional ? 2 : 1;

                    h0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: input.dtype, device: input.device);
                    c0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: input.dtype, device: input.device);
                } else {
                    h0 = h0_c0.Value.Item1;
                    c0 = h0_c0.Value.Item2;
                }

                var res = THSNN_LSTM_forward(handle, input.Handle, h0.Handle, c0.Handle, out IntPtr hN, out IntPtr cN);
                if (res == IntPtr.Zero || hN == IntPtr.Zero || cN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(hN), new Tensor(cN));
            }

            public new (Tensor, Tensor, Tensor) call(Tensor input, (Tensor, Tensor)? h0_c0 = null) => base.call(input, h0_c0);

            /// <summary>
            /// Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
            /// </summary>
            /// <param name="input">PackedSequence containing the features of the input sequence.</param>
            /// <param name="h0_c0">Tensors of shape (num_layers * num_directions, batch, hidden_size) containing the initial hidden and cell state for each element in the batch</param>
            /// <returns></returns>
            public (torch.nn.utils.rnn.PackedSequence, Tensor, Tensor) call(torch.nn.utils.rnn.PackedSequence input, (Tensor, Tensor)? h0_c0 = null)
            {
                Tensor c0, h0;

                if (h0_c0 == null) {
                    var data = input.data;
                    var batch_sizes = input.batch_sizes;
                    var N = batch_sizes[0].item<long>();
                    var D = _bidirectional ? 2 : 1;

                    h0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: data.dtype, device: data.device);
                    c0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: data.dtype, device: data.device);
                } else {
                    h0 = h0_c0.Value.Item1;
                    c0 = h0_c0.Value.Item2;
                }

                var res = THSNN_LSTM_forward_with_packed_input(handle, input.Handle, h0.Handle, c0.Handle, out IntPtr hN, out IntPtr cN);
                if (res.IsInvalid || hN == IntPtr.Zero || cN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new torch.nn.utils.rnn.PackedSequence(res), new Tensor(hN), new Tensor(cN));
            }

            public void flatten_parameters()
            {
                THSNN_LSTM_flatten_parameters(handle);
                torch.CheckForErrors();
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
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
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static LSTM LSTM(long inputSize, long hiddenSize, long numLayers = 1, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_LSTM_ctor(inputSize, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LSTM(res, boxedHandle, hiddenSize, numLayers, batchFirst, bidirectional).MoveModule<LSTM>(device, dtype);
            }
        }
    }
}
