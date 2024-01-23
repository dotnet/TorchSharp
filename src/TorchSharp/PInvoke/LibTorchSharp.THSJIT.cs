// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
#pragma warning disable CA2101
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern void THSJIT_CompilationUnit_Invoke(IntPtr module, string name, IntPtr tensors, int length, AllocateIndexedNativeTensorOrScalarArray allocator, out sbyte typeCode, int idx);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_CompilationUnit_dispose(IntPtr handle);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSJIT_compile(string script);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_dispose(torch.nn.Module.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_named_parameters(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_named_buffers(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_named_attributes(torch.nn.Module.HType module, [MarshalAs(UnmanagedType.U1)] bool recurse, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_set_attribute(torch.nn.Module.HType module, [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_named_modules(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_named_children(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern long THSJIT_getNumModules(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern int THSJIT_Module_num_inputs(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern int THSJIT_Module_num_outputs(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_train(torch.nn.Module.HType module, [MarshalAs(UnmanagedType.U1)] bool on);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_eval(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSJIT_Module_is_training(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_to_device(torch.nn.Module.HType module, long deviceType, long deviceIndex);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_to_device_dtype(torch.nn.Module.HType module, sbyte dtype, long deviceType, long deviceIndex);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_to_dtype(torch.nn.Module.HType module, sbyte dtype);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSJIT_Module_getInputType(torch.nn.Module.HType module, int index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSJIT_getOutputType(torch.jit.Type.HType module, int index);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Module_forward(torch.nn.Module.HType module, IntPtr tensors, int length, AllocateIndexedNativeTensorOrScalarArray allocator, out sbyte typeCode, int idx);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern void THSJIT_Module_invoke(torch.nn.Module.HType module, string name, IntPtr tensors, int length, AllocateIndexedNativeTensorOrScalarArray allocator, out sbyte typeCode, int idx);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSJIT_load(string filename, long deviceType, long deviceIndex);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSJIT_load_byte_array(IntPtr bytes, long size, long deviceType, long deviceIndex);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern void THSJIT_save(torch.nn.Module.HType handle, string filename);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_save_byte_array(torch.nn.Module.HType handle, IntPtr bytes, long size);

        [DllImport("LibTorchSharp")]
        internal static extern long THSJIT_TensorType_sizes(torch.jit.Type.HType handle, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern int THSJIT_getDimensionedTensorTypeDimensions(torch.jit.Type.HType handle);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern string THSJIT_getDimensionedTensorDevice(torch.jit.Type.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSJIT_TensorType_dtype(torch.jit.Type.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_Type_dispose(torch.jit.Type.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_TensorType_dispose(torch.jit.Type.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSJIT_Type_kind(torch.jit.Type.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSJIT_Type_cast(torch.jit.Type.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSJIT_AllocateTensorOrScalarArray(int size);
        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_FreeTensorOrScalarArray(IntPtr ptr);
        [DllImport("LibTorchSharp")]
        internal static extern void THSJIT_SetTensorOrScalar(IntPtr array, int index, long type_code, long array_index, IntPtr handle);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSJIT_GetTensorOrScalar(IntPtr array, int index);
    }
#pragma warning restore CA2101
}
