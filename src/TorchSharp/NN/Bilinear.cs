// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Bilinear : torch.nn.Module
        {
            internal Bilinear(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public new static Bilinear Load(String modelPath)
            {
                var res = Module.Load(modelPath);
                return new Bilinear(res.handle.DangerousGetHandle(), IntPtr.Zero);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Bilinear_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

            public Tensor forward(Tensor input1, Tensor input2)
            {
                var res = THSNN_Bilinear_forward(handle, input1.Handle, input2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Bilinear_bias(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Bilinear_set_bias(torch.nn.Module.HType module, IntPtr tensor);

            public Tensor? Bias {
                get {
                    var res = THSNN_Bilinear_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Tensor(res));
                }
                set {
                    THSNN_Bilinear_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                }
            }
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Bilinear_weight(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Bilinear_set_weight(torch.nn.Module.HType module, IntPtr tensor);

            public Tensor Weight {
                get {
                    var res = THSNN_Bilinear_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSNN_Bilinear_set_weight(handle, value.Handle);
                    torch.CheckForErrors();
                }
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Bilinear_ctor(long in1_features, long in2_features, long output_size, bool bias, out IntPtr pBoxedModule);

            static public Bilinear Bilinear(long in1Features, long in2Features, long outputSize, bool hasBias = true)
            {
                var res = THSNN_Bilinear_ctor(in1Features, in2Features, outputSize, hasBias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Bilinear(res, boxedHandle);
            }
        }
        public static partial class functional
        {
            static public Tensor Bilinear(Tensor x1, Tensor x2, long in1Features, long in2Features, long outputSize, bool hasBias = true)
            {
                using (var d = nn.Bilinear(in1Features, in2Features, outputSize, hasBias)) {
                    return d.forward(x1, x2);
                }
            }
        }
    }
}