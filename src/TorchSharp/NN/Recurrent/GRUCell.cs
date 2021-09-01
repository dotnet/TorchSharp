// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class GRUCell : torch.nn.Module
        {
            internal GRUCell(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public new static GRUCell Load(String modelPath)
            {
                var res = Module.Load(modelPath);
                return new GRUCell(res.handle.DangerousGetHandle(), IntPtr.Zero);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_GRUCell_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0);

            /// <summary>
            /// Apply the GRU cell to an input tensor.
            /// </summary>
            /// <param name="input">Tensor of shape (batch, input_size) containing the features of the input sequence.</param>
            /// <param name="h0">Tensor of shape (batch, hidden_size) containing the initial hidden state for each element in the batch.</param>
            /// <returns></returns>
            public override Tensor forward(Tensor input, Tensor h0 = null)
            {
                var hN = THSNN_GRUCell_forward(handle, input.Handle, h0?.Handle ?? IntPtr.Zero);
                if (hN == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(hN);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_GRUCell_ctor(long input_size, long hidden_size, bool bias, out IntPtr pBoxedModule);

            /// <summary>
            /// A gated recurrent unit (GRU) cell
            /// </summary>
            /// <param name="inputSize">The number of expected features in the input x</param>
            /// <param name="hiddenSize">The number of features in the hidden state h</param>
            /// <param name="bias">If False, then the layer does not use bias weights b_ih and b_hh. Default: True</param>
            static public GRUCell GRUCell(long inputSize, long hiddenSize, bool bias = true)
            {
                var res = THSNN_GRUCell_ctor(inputSize, hiddenSize, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new GRUCell(res, boxedHandle);
            }
        }
    }
}
