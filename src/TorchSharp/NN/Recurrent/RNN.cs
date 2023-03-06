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
        public sealed class RNN : torch.nn.Module<Tensor, Tensor?, (Tensor, Tensor)>
        {
            internal RNN(IntPtr handle, IntPtr boxedHandle, long hiddenSize, long numLayers, bool batchFirst, bool bidirectional) : base(handle, boxedHandle)
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
            /// Applies a multi-layer Elman RNN with \tanhtanh or \text{ReLU}ReLU non-linearity to an input sequence.
            /// </summary>
            /// <param name="input">Tensor of shape (seq_len, batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (num_layers * num_directions, batch, hidden_size)containing the initial hidden state for each element in the batch.
            /// Defaults to 0 if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.</param>
            /// <returns></returns>
            public override (Tensor, Tensor) forward(Tensor input, Tensor? h0)
            {
                if (h0 is null) {
                    var N = _batch_first ? input.shape[0] : input.shape[1];
                    var D = _bidirectional ? 2 : 1;

                    h0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: input.dtype, device: input.device);
                }

                var res = THSNN_RNN_forward(handle, input.Handle, h0.Handle, out IntPtr hN);
                if (res == IntPtr.Zero || hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(hN));
            }

            public new (Tensor, Tensor) call(Tensor input, Tensor? h0 = null) => base.call(input, h0);

            /// <summary>
            /// Applies a multi-layer Elman RNN with \tanhtanh or \text{ReLU}ReLU non-linearity to an input sequence.
            /// </summary>
            /// <param name="input">PackedSequence containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (num_layers * num_directions, batch, hidden_size)containing the initial hidden state for each element in the batch.
            /// Defaults to 0 if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.</param>
            /// <returns></returns>
            public (torch.nn.utils.rnn.PackedSequence, Tensor) call(torch.nn.utils.rnn.PackedSequence input, Tensor? h0 = null)
            {
                if (h0 is null) {
                    var data = input.data;
                    var batch_sizes = input.batch_sizes;
                    var N = batch_sizes[0].item<long>();
                    var D = _bidirectional ? 2 : 1;

                    h0 = torch.zeros(D * _num_layers, N, _hidden_size, dtype: data.dtype, device: data.device);
                }

                var res = THSNN_RNN_forward_with_packed_input(handle, input.Handle, h0.Handle, out IntPtr hN);
                if (res.IsInvalid || hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new torch.nn.utils.rnn.PackedSequence(res), new Tensor(hN));
            }

            public void flatten_parameters()
            {
                THSNN_RNN_flatten_parameters(handle);
                torch.CheckForErrors();
            }

            public Parameter? get_bias_ih(long idx)
            {
                var res = THSNN_RNN_bias_ih(handle, idx);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return ((res == IntPtr.Zero) ? null : new Parameter(res));
            }

            public Parameter? get_bias_hh(long idx)
            {
                var res = THSNN_RNN_bias_hh(handle, idx);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return ((res == IntPtr.Zero) ? null : new Parameter(res));
            }

#if false   // Disabled until we can figure out how to set the native code parameters.

            public void set_bias_ih(Tensor? value, long idx)
            {
                THSNN_RNN_set_bias_ih(handle, (value is null ? IntPtr.Zero : value.Handle), idx);
                torch.CheckForErrors();
                ConditionallyRegisterParameter($"bias_ih_l{idx}", value);
            }

            public void set_bias_hh(Tensor? value, long idx)
            {
                THSNN_RNN_set_bias_hh(handle, (value is null ? IntPtr.Zero : value.Handle), idx);
                torch.CheckForErrors();
                ConditionallyRegisterParameter($"bias_hh_l{idx}", value);
            }
#endif

            public Parameter? get_weight_ih(long idx)
            {
                var res = THSNN_RNN_weight_ih(handle, idx);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return (res == IntPtr.Zero) ? null : new Parameter(res);
            }

            public Parameter? get_weight_hh(long idx)
            {
                var res = THSNN_RNN_weight_hh(handle, idx);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return (res == IntPtr.Zero) ? null : new Parameter(res);
            }

#if false   // Disabled until we can figure out how to set the native code parameters.
            public void set_weight_ih(Tensor? value, long idx)
            {
                THSNN_RNN_set_weight_ih(handle, (value is null ? IntPtr.Zero : value.Handle), idx);
                torch.CheckForErrors();
                ConditionallyRegisterParameter($"weight_ih_l{idx}", value);
            }

            public void set_weight_hh(Tensor? value, long idx)
            {
                THSNN_RNN_set_weight_hh(handle, (value is null ? IntPtr.Zero : value.Handle), idx);
                torch.CheckForErrors();
                ConditionallyRegisterParameter($"weight_hh_l{idx}", value);
            }
#endif
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
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static RNN RNN(long inputSize, long hiddenSize, long numLayers = 1, NonLinearities nonLinearity = nn.NonLinearities.Tanh, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_RNN_ctor(inputSize, hiddenSize, numLayers, (long)nonLinearity, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new RNN(res, boxedHandle, hiddenSize, numLayers, batchFirst, bidirectional).MoveModule<RNN>(device, dtype);
            }
        }
    }
}
