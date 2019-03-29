using System;

namespace TorchSharp.JIT
{
    public sealed class DynamicType : Type
    {
        internal DynamicType(IntPtr handle) : base(handle)
        {
            this.handle = new HType(handle, true);
        }

        internal DynamicType(Type type) : base()
        {
            handle = type.handle;
            type.handle = new HType(IntPtr.Zero, true);
            type.Dispose();
        }
    }
}
