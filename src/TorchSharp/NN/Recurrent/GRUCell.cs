// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics.CodeAnalysis;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class GRUCell : torch.nn.Module<Tensor, Tensor, Tensor>
        {
            internal GRUCell(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Apply the GRU cell to an input tensor.
            /// </summary>
            /// <param name="input">Tensor of shape (batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (batch, hidden_size) containing the initial hidden state for each element in the batch.</param>
            /// <returns></returns>
            public override Tensor forward(Tensor input, Tensor? h0 = null)
            {
                var hN = THSNN_GRUCell_forward(handle, input.Handle, h0?.Handle ?? IntPtr.Zero);
                if (hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(hN);
            }

            [DisallowNull]
            public Parameter? bias_ih {
                get {
                    var res = THSNN_GRUCell_bias_ih(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_GRUCell_set_bias_ih(handle, value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias_ih", value);
                }
            }

            [DisallowNull]
            public Parameter? bias_hh {
                get {
                    var res = THSNN_GRUCell_bias_hh(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_GRUCell_set_bias_hh(handle, value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias_hh", value);
                }
            }

            [DisallowNull]
            public Parameter? weight_ih {
                get {
                    var res = THSNN_GRUCell_weight_ih(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_GRUCell_set_weight_ih(handle, value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight_ih", value);
                }
            }

            [DisallowNull]
            public Parameter? weight_hh {
                get {
                    var res = THSNN_GRUCell_weight_hh(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_GRUCell_set_weight_hh(handle, value.Handle);
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
            /// A gated recurrent unit (GRU) cell
            /// </summary>
            /// <param name="inputSize">The number of expected features in the input x</param>
            /// <param name="hiddenSize">The number of features in the hidden state h</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <param name="bias">If False, then the layer does not use bias weights b_ih and b_hh. Default: True</param>
            public static GRUCell GRUCell(long inputSize, long hiddenSize, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_GRUCell_ctor(inputSize, hiddenSize, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new GRUCell(res, boxedHandle).MoveModule<GRUCell>(device, dtype);
            }
        }
    }
}
