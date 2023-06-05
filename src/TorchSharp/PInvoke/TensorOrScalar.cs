using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct TensorOrScalar
    {
        public long TypeCode;
        public long ArrayIndex;
        public IntPtr Handle;
    }
}