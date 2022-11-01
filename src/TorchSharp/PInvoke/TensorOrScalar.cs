using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct TensorOrScalar
    {
        public long TypeCode;
        public IntPtr Handle;
    }
}