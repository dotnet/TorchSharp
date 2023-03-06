// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class LSTMCell : torch.nn.Module<Tensor, (Tensor, Tensor)?, (Tensor, Tensor)>
        {
            internal LSTMCell(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public new static LSTMCell Load(string modelPath)
            {
                var res = Module<Tensor, Tensor>.Load(modelPath);
                return new LSTMCell(res.handle.DangerousGetHandle(), IntPtr.Zero);
            }

            /// <summary>
            /// Apply the RNN cell to an input tensor.
            /// </summary>
            /// <param name="input">Tensor of shape (batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0_c0">Tensors of shape (batch, hidden_size) containing the initial hidden and cell state for each element in the batch.</param>
            /// <returns></returns>
            public override (Tensor, Tensor) forward(Tensor input, (Tensor, Tensor)? h0_c0)
            {
                var hN = THSNN_LSTMCell_forward(handle, input.Handle, h0_c0?.Item1.Handle ?? IntPtr.Zero, h0_c0?.Item2.Handle ?? IntPtr.Zero, out IntPtr cN);
                if (hN == IntPtr.Zero || cN == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(hN), new Tensor(cN));
            }

            public new (Tensor, Tensor) call(Tensor input, (Tensor, Tensor)? h0_c0 = null) => base.call(input, h0_c0);

            public Parameter? bias_ih {
                get {
                    var res = THSNN_LSTMCell_bias_ih(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_LSTMCell_set_bias_ih(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias_ih", value);
                }
            }

            public Parameter? bias_hh {
                get {
                    var res = THSNN_LSTMCell_bias_hh(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_LSTMCell_set_bias_hh(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias_hh", value);
                }
            }

            public Parameter? weight_ih {
                get {
                    var res = THSNN_LSTMCell_weight_ih(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_LSTMCell_set_weight_ih(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight_ih", value);
                }
            }

            public Parameter? weight_hh {
                get {
                    var res = THSNN_LSTMCell_weight_hh(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_LSTMCell_set_weight_hh(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight_hh", value);
                }
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// A long short-term memory (LSTM) cell.
            /// </summary>
            /// <param name="inputSize">The number of expected features in the input x</param>
            /// <param name="hiddenSize">The number of features in the hidden state h</param>
            /// <param name="bias">If False, then the layer does not use bias weights b_ih and b_hh. Default: True</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            public static LSTMCell LSTMCell(long inputSize, long hiddenSize, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_LSTMCell_ctor(inputSize, hiddenSize, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LSTMCell(res, boxedHandle).MoveModule<LSTMCell>(device, dtype);
            }
        }
    }
}
