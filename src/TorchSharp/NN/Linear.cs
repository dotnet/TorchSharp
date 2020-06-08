// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        public new static Linear Load (String modelPath)
        {
            var res = Module.Load (modelPath);
            Torch.CheckForErrors ();
            return new Linear (res.handle.DangerousGetHandle(), IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Linear_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_bias (Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Linear_set_bias (Module.HType module, IntPtr tensor);

        public TorchTensor? Bias {
            get {
                var res = THSNN_Linear_bias (handle);
                Torch.CheckForErrors ();
                return ((res == IntPtr.Zero) ? null : new TorchTensor (res));
            }
            set {
                THSNN_Linear_set_bias (handle, (value is null ? IntPtr.Zero : value.Handle));
                Torch.CheckForErrors ();
            }
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_weight (Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Linear_set_weight (Module.HType module, IntPtr tensor);

        public TorchTensor Weight {
            get {
                var res = THSNN_Linear_weight (handle);
                Torch.CheckForErrors ();
                return new TorchTensor (res);
            }
            set {
                THSNN_Linear_set_weight (handle, value.Handle);
                Torch.CheckForErrors ();
            }
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Linear_ctor (long input_size, long output_size, bool bias, out IntPtr pBoxedModule);

        static public Linear Linear (long inputSize, long outputSize, bool hasBias = true)
        {
            var res = THSNN_Linear_ctor (inputSize, outputSize, hasBias, out var boxedHandle);
            Torch.CheckForErrors ();
            return new Linear (res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Linear (TorchTensor x, long inputSize, long outputSize, bool hasBias = true)
        {
            using (var d = Modules.Linear (inputSize, outputSize, hasBias)) {
                return d.Forward (x);
            }
        }
    }

}
