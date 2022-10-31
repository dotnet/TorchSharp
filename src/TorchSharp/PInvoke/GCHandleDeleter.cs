using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GCHandleDeleter(IntPtr memory);
}