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

        public new static GRU Load (String modelPath)
        {
            var res = Module.Load (modelPath);
            return new GRU (res.handle.DangerousGetHandle(), IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_GRU_forward (Module.HType module, IntPtr input, IntPtr h_0, out IntPtr h_n);

        public (TorchTensor,TorchTensor) forward (TorchTensor input, TorchTensor? h0 = null)
        {
            var res = THSNN_GRU_forward (handle, input.Handle, h0?.Handle ?? IntPtr.Zero, out IntPtr hN);
            if (res == IntPtr.Zero || hN == IntPtr.Zero) { Torch.CheckForErrors(); }
            return (new TorchTensor (res), new TorchTensor(hN));
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_GRU_ctor (long input_size, long hidden_size, long num_layers, bool bias, bool batchFirst, double dropout, bool bidirectional, out IntPtr pBoxedModule);

        static public GRU GRU (long inputSize, long hiddenSize, long numLayers = 1, bool bias = true, bool batchFirst = false, double dropout = 0.0, bool bidirectional = false)
        {
            var res = THSNN_GRU_ctor(inputSize, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new GRU (res, boxedHandle);
        }
    }
}
