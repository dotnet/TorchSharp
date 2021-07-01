// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class Bilinear : Module
    {
        internal Bilinear (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        public new static Bilinear Load (String modelPath)
        {
            var res = Module.Load (modelPath);
            return new Bilinear (res.handle.DangerousGetHandle(), IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Bilinear_forward (Module.HType module, IntPtr input1, IntPtr input2);

        public TorchTensor forward (TorchTensor input1, TorchTensor input2)
        {
            var res = THSNN_Bilinear_forward (handle, input1.Handle, input2.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Bilinear_bias (Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Bilinear_set_bias (Module.HType module, IntPtr tensor);

        public TorchTensor? Bias {
            get {
                var res = THSNN_Bilinear_bias (handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return ((res == IntPtr.Zero) ? null : new TorchTensor (res));
            }
            set {
                THSNN_Bilinear_set_bias (handle, (value is null ? IntPtr.Zero : value.Handle));
                torch.CheckForErrors ();
            }
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Bilinear_weight (Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Bilinear_set_weight (Module.HType module, IntPtr tensor);

        public TorchTensor Weight {
            get {
                var res = THSNN_Bilinear_weight (handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor (res);
            }
            set {
                THSNN_Bilinear_set_weight (handle, value.Handle);
                torch.CheckForErrors ();
            }
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Bilinear_ctor (long in1_features, long in2_features, long output_size, bool bias, out IntPtr pBoxedModule);

        static public Bilinear Bilinear (long in1Features, long in2Features, long outputSize, bool hasBias = true)
        {
            var res = THSNN_Bilinear_ctor (in1Features, in2Features, outputSize, hasBias, out var boxedHandle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Bilinear (res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Bilinear (TorchTensor x1, TorchTensor x2, long in1Features, long in2Features, long outputSize, bool hasBias = true)
        {
            using (var d = Modules.Bilinear(in1Features, in2Features, outputSize, hasBias)) { 
                return d.forward(x1, x2);
            }
        }
    }
}