/*using System.Runtime.InteropServices;
using TorchSharp.PInvoke;

namespace System
{
    [StructLayout(LayoutKind.Sequential,Pack=2)]
    public struct BFloat16
    {
        [MarshalAs(UnmanagedType.U2)]
        public ushort x;
        public struct from_bits_t{};

        public BFloat16(float value)
        {
            var bf = NativeMethods.THSBFloat16_ctor(value);
            this.x = bf.x;
        }

        public float ToFloat()
        {
            return NativeMethods.THSBFloat16_op_float(this);
        }
    }
}
*/