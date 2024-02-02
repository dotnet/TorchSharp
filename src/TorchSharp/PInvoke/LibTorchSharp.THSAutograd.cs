// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_isGradEnabled();

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_setGrad([MarshalAs(UnmanagedType.U1)] bool enabled);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_isInferenceModeEnabled();

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAutograd_getInferenceModeGuard([MarshalAs(UnmanagedType.U1)] bool mode);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_deleteInferenceModeGuard(IntPtr guard);
        
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_isAnomalyEnabled();

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSAutograd_shouldCheckNaN();

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_setAnomaly([MarshalAs(UnmanagedType.U1)] bool enabled, [MarshalAs(UnmanagedType.U1)] bool check_nan);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_grad(
            IntPtr outputs, long oLength,
            IntPtr inputs, long iLength,
            IntPtr grad_outs, long gLength,
            [MarshalAs(UnmanagedType.U1)] bool retain_graph,
            [MarshalAs(UnmanagedType.U1)] bool create_graph,
            [MarshalAs(UnmanagedType.U1)] bool allow_unused,
            AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_backward(
            IntPtr tensors, long tLength,
            IntPtr grad_tensors, long gtLength,
            [MarshalAs(UnmanagedType.U1)] bool retain_graph,
            [MarshalAs(UnmanagedType.U1)] bool create_graph,
            IntPtr inputs, long iLength);

        [StructLayout(LayoutKind.Sequential)]
        internal struct ArrayWithSize {
            public IntPtr Array;
            public long Size;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NodeUnmanagedPtr {
            public IntPtr sharedPtr;
            public IntPtr weakPtr;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate ArrayWithSize ApplyFunc([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] IntPtr[] array, int size);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ManagedDeleteNode();

        [DllImport("LibTorchSharp")]
        internal static extern NodeUnmanagedPtr THSAutograd_CSharpNode_ctor(ApplyFunc applyFunc, ManagedDeleteNode managedDeleteNode);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_CSharpNode_disposeSharedPtr(NodeUnmanagedPtr node);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_CSharpNode_disposeWeakPtr(NodeUnmanagedPtr node);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_CSharpNode_setNextEdges(NodeUnmanagedPtr node, ArrayWithSize vars, [MarshalAs(UnmanagedType.U1)] bool is_executable);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_CSharpNode_clearInputMetadata(NodeUnmanagedPtr node);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_Function_wrapOutputs(ArrayWithSize vars, ArrayWithSize nonDiff, ArrayWithSize dirty, ArrayWithSize outputs, NodeUnmanagedPtr node, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAutograd_SavedVariable_ctor(IntPtr variable, NodeUnmanagedPtr nodeRef, [MarshalAs(UnmanagedType.U1)] bool is_inplace_on_view);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_SavedVariable_dispose(IntPtr saved_variable);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSAutograd_SavedVariable_unpack(IntPtr saved_variable, NodeUnmanagedPtr node_saved_for);

        [DllImport("LibTorchSharp")]
        internal static extern void THSAutograd_SavedVariable_reset_data(IntPtr saved_variable);
    }
}
