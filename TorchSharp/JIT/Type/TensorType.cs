using System;
using System.Runtime.InteropServices;

namespace TorchSharp.JIT
{
    public sealed class TensorType : Type
    {
        internal TensorType(IntPtr handle) : base(handle)
        {
            this.handle = new HType(handle, true);
        }

        internal TensorType(Type type) : base()
        {
            handle = type.handle;
            type.handle = new HType(IntPtr.Zero, true);
            type.Dispose();
        }

        [DllImport("libTorchSharp")]
        extern static short JIT_TensorType_getScalar(HType handle);

        public Tensor.ATenScalarMapping GetScalarType()
        {
            return (Tensor.ATenScalarMapping)JIT_TensorType_getScalar(handle);
        }

        [DllImport("libTorchSharp")]
        extern static int JIT_TensorType_getDimensions(HType handle);

        public int GetDimensions()
        {
            return JIT_TensorType_getDimensions(handle);
        }

        [DllImport("libTorchSharp")]
        extern static string JIT_TensorType_getDevice(HType handle);

        public string GetDevice()
        {
            return JIT_TensorType_getDevice(handle);
        }
    }
}
