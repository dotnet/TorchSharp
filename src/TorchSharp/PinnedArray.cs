// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;

/// <summary>
/// Allocator of T[] that pins the memory and handles unpinning.
/// (taken from StackOverflow)
/// </summary>
/// <typeparam name="T"></typeparam>
internal sealed class PinnedArray<T> : IDisposable where T : struct
{
    private GCHandle handle;

    public T[] Array { get; private set; }

    public IntPtr CreateArray(int length)
    {
        FreeHandle();

        Array = new T[length];

        // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
        try
        {
        }
        finally
        {
            handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
        }

        return handle.AddrOfPinnedObject();
    }

    public IntPtr CreateArray(IntPtr length)
    {
        return CreateArray((int)length);
    }

    public IntPtr CreateArray(T[] array)
    {
        FreeHandle();

        Array = array;

        // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
        try
        {
        }
        finally
        {
            handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
        }

        return handle.AddrOfPinnedObject();
    }

    public void Dispose()
    {
        if (Array != null)
        {
            foreach (var val in Array)
            {
                (val as IDisposable)?.Dispose();
            }
        }
        FreeHandle();
    }

    ~PinnedArray()
    {
        FreeHandle();
    }

    private void FreeHandle()
    {
        if (handle.IsAllocated)
        {
            handle.Free();
        }
    }
}

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public delegate IntPtr AllocatePinnedArray(IntPtr length);