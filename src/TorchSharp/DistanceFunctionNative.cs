#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate IntPtr DistanceFunctionNative(IntPtr x, IntPtr y);
}