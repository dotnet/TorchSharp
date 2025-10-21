// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
#pragma warning disable CA2101
    internal static partial class NativeMethods
    {
        // torch.export support via AOTInductor (INFERENCE-ONLY)
        // Models must be compiled with torch._inductor.aoti_compile_and_package() in Python

        // Load ExportedProgram from .pt2 file
        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSExport_load(string filename);

        // Dispose ExportedProgram module
        [DllImport("LibTorchSharp")]
        internal static extern void THSExport_Module_dispose(IntPtr handle);

        // Execute forward pass (inference only)
        [DllImport("LibTorchSharp")]
        internal static extern void THSExport_Module_run(
            IntPtr module,
            IntPtr[] input_tensors,
            int input_length,
            out IntPtr result_tensors,
            out int result_length);
    }
#pragma warning restore CA2101
}
