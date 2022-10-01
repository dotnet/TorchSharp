// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class GRU : torch.nn.Module
        {
            internal GRU(IntPtr handle, IntPtr boxedHandle, long hiddenSize, long numLayers, bool batchFirst, bool bidirectional) : base(handle, boxedHandle)
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

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_GRU_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr input, IntPtr h_0, out IntPtr h_n);

            /// <summary>
            /// Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
            /// </summary>
            /// <param name="input">Tensor of shape (seq_len, batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (num_layers * num_directions, batch, hidden_size)containing the initial hidden state for each element in the batch.
            /// Defaults to 0 if not provided. If the GRU is bidirectional, num_directions should be 2, else it should be 1.</param>
            /// <returns></returns>
            /// <returns></returns>
            public (Tensor, Tensor) forward(Tensor input, Tensor h0 = null)
            {
                if (h0 is null) {
                    var N = _batch_first ? input.shape[0] : input.shape[1];
                    var D = _bidirectional ? 2 : 1;

                    h0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: input.dtype, device: input.device);
                }

                var res = THSNN_GRU_forward(handle, input.Handle, h0.Handle, out IntPtr hN);
                if (res == IntPtr.Zero || hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(hN));
            }

            [DllImport("LibTorchSharp")]
            extern static torch.nn.utils.rnn.PackedSequence.HType THSNN_GRU_forward_with_packed_input(torch.nn.Module<Tensor, Tensor>.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, out IntPtr h_n);

            /// <summary>
            /// Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
            /// </summary>
            /// <param name="input">PackedSequence containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (num_layers * num_directions, batch, hidden_size)containing the initial hidden state for each element in the batch.
            /// Defaults to 0 if not provided. If the GRU is bidirectional, num_directions should be 2, else it should be 1.</param>
            /// <returns></returns>
            /// <returns></returns>
            public (torch.nn.utils.rnn.PackedSequence, Tensor) forward(torch.nn.utils.rnn.PackedSequence input, Tensor h0 = null)
            {
                if (h0 is null) {
                    var data = input.data;
                    var batch_sizes = input.batch_sizes;
                    var N = batch_sizes[0].item<long>();
                    var D = _bidirectional ? 2 : 1;

                    h0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: data.dtype, device: data.device);
                }

                var res = THSNN_GRU_forward_with_packed_input(handle, input.Handle, h0.Handle, out IntPtr hN);
                if (res.IsInvalid || hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new torch.nn.utils.rnn.PackedSequence(res), new Tensor(hN));
            }

            [DllImport("LibTorchSharp")]
            extern static void THSNN_GRU_flatten_parameters(torch.nn.Module<Tensor, Tensor>.HType module);

            public void flatten_parameters()
            {
                THSNN_GRU_flatten_parameters(handle);
                torch.CheckForErrors();
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_GRU_ctor(long input_size, long hidden_size, long num_layers, bool bias, bool batchFirst, double dropout, bool bidirectional, out IntPtr pBoxedModule);

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
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            static public GRU GRU(long inputSize, long hiddenSize, long numLayers = 1, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false, Device device = null, ScalarType? dtype = null)
            {
                var res = THSNN_GRU_ctor(inputSize, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new GRU(res, boxedHandle, hiddenSize, numLayers, batchFirst, bidirectional).MoveModule<GRU>(device, dtype);
            }
        }
    }
}
