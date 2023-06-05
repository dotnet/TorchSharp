// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    /// <summary>
    /// Allocator of T[] that pins the memory and handles unpinning.
    /// Modified version of PinnedArray, this class allows multiple arrays to be handled
    /// and unpinned together. Each array is identified by an integer index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal sealed class IndexedPinnedArrays<T> : IDisposable where T : struct
    {
        private List<GCHandle> handles = new List<GCHandle>();
        private List<T[]> arrays = new List<T[]>();

        public int Count {
            get { return arrays.Count; }
        }

        public T[] this[int idx] {
            get {
                return arrays[idx];
            }
            set {
                ExtendHandlesList(idx);
                arrays[idx] = value;
            }
        }

        public IntPtr CreateArray(int index, int length)
        {
            this[index] = new T[length];

            // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
            try {
            } finally {
                handles[index] = GCHandle.Alloc(this[index], GCHandleType.Pinned);
            }

            return handles[index].AddrOfPinnedObject();
        }

        public IntPtr CreateArray(int index, IntPtr length)
        {
            return CreateArray(index, (int)length);
        }

        public IntPtr CreateArray(int index, T[] array)
        {
            this[index] = array;

            // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
            try {
            } finally {
                handles[index] = GCHandle.Alloc(array, GCHandleType.Pinned);
            }

            return handles[index].AddrOfPinnedObject();
        }

        public void Dispose()
        {
            foreach (var array in arrays)
                foreach (var val in array) {
                    (val as IDisposable)?.Dispose();
                }
            FreeHandles();
        }

        ~IndexedPinnedArrays()
        {
            FreeHandles();
        }

        private void ExtendHandlesList(int idx)
        {
            if (idx >= handles.Count) {
                var extras = idx - handles.Count + 1;
                for (var i = 0; i < extras; i++) {
                    handles.Add(default(GCHandle));
                    arrays.Add(default(T[]));
                }
            }
        }

        private void FreeHandles()
        {
            foreach (var handle in handles) {
                if (handle.IsAllocated) {
                    handle.Free();
                }
            }
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate IntPtr AllocateIndexedPinnedArray(int id, IntPtr length);
}