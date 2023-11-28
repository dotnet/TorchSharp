// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate IntPtr AllocateIndexedNativeTensorOrScalarArray(int id, IntPtr length);

    /// <summary>
    /// Allocator of TensorOrScalar[] that is used by the native runtime to allocate and register
    /// native memory for processing TorchScript arguments and return values.
    /// </summary>
    internal sealed class NativeTensorOrScalarIndexedArray : IDisposable
    {
        private List<IntPtr> arrays = new List<IntPtr>();
        private List<int> sizes = new List<int>();

        private int _allocated = 0;

        public int Count {
            get { return arrays.Count; }
        }

        public IntPtr this[int idx] {
            get {
                return arrays[idx];
            }
        }

        public IntPtr this[int idx, int size] {
            set {
                ExtendHandlesList(idx);
                arrays[idx] = value;
                sizes[idx] = size;
            }
        }
        public IntPtr CreateArray(int index, int length)
        {
            var result = THSJIT_AllocateTensorOrScalarArray(length);
            torch.CheckForErrors();
            _allocated += 1;
            System.Diagnostics.Debug.Assert(result != IntPtr.Zero);

            this[index, length] = result;
            return result;
        }

        internal unsafe static TensorOrScalar ToTOS(IntPtr handle, int index)
        {
            TensorOrScalar* ptr = (TensorOrScalar*)THSJIT_GetTensorOrScalar(handle, index);
            torch.CheckForErrors();
            var result = new TensorOrScalar();
            result.Handle = ptr->Handle;
            result.ArrayIndex = ptr->ArrayIndex;
            result.TypeCode = ptr->TypeCode;
            return result;
        }

        public TensorOrScalar[] ToToSArray(int index)
        {
            var size = sizes[index];
            var ptr = arrays[index];

            var result = new TensorOrScalar[size];

            for (int i = 0; i < size; i++) {
                result[i] = ToTOS(ptr, i);
            }
            return result;
        }

        public IntPtr CreateArray(int index, IntPtr length)
        {
            return CreateArray(index, (int)length);
        }

        public void Dispose()
        {
            FreeHandles();
        }

        ~NativeTensorOrScalarIndexedArray()
        {
            FreeHandles();
        }

        private void ExtendHandlesList(int idx)
        {
            if (idx >= arrays.Count) {
                var extras = idx - arrays.Count + 1;
                for (var i = 0; i < extras; i++) {
                    arrays.Add(IntPtr.Zero);
                    sizes.Add(0);
                }
            }
        }

        private void FreeHandles()
        {
            foreach (var handle in arrays) {
                if (handle != IntPtr.Zero) {
                    THSJIT_FreeTensorOrScalarArray(handle);
                    torch.CheckForErrors();
                    _allocated -= 1;
                }
            }
            arrays.Clear();

            System.Diagnostics.Debug.Assert(_allocated == 0);
        }
    }
}