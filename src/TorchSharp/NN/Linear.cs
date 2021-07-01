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
    public class Linear : torch.nn.Module
    {
        internal Linear (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        public new static Linear Load (String modelPath)
        {
            var res = Module.Load (modelPath);
            return new Linear (res.handle.DangerousGetHandle(), IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_forward (torch.nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Linear_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_bias (torch.nn.Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Linear_set_bias (torch.nn.Module.HType module, IntPtr tensor);

        public TorchTensor? Bias {
            get {
                var res = THSNN_Linear_bias (handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return ((res == IntPtr.Zero) ? null : new TorchTensor (res));
            }
            set {
                THSNN_Linear_set_bias (handle, (value is null ? IntPtr.Zero : value.Handle));
                torch.CheckForErrors ();
            }
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_weight (torch.nn.Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Linear_set_weight (torch.nn.Module.HType module, IntPtr tensor);

        public TorchTensor Weight {
            get {
                var res = THSNN_Linear_weight (handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor (res);
            }
            set {
                THSNN_Linear_set_weight (handle, value.Handle);
                torch.CheckForErrors ();
            }
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Linear_ctor (long input_size, long output_size, bool bias, out IntPtr pBoxedModule);

        static public Linear Linear (long inputSize, long outputSize, bool hasBias = true)
        {
            var res = THSNN_Linear_ctor (inputSize, outputSize, hasBias, out var boxedHandle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Linear (res, boxedHandle);
        }
    }
    public static partial class functional
    {
        static public TorchTensor Linear (TorchTensor x, long inputSize, long outputSize, bool hasBias = true)
        {
            using (var d =nn.Linear (inputSize, outputSize, hasBias)) {
                return d.forward (x);
            }
        }
    }

}
