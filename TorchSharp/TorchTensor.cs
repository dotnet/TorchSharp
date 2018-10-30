using System;
using System.Numerics.Tensors;

namespace TorchSharp
{
    public class FloatTorchTensor : DenseTensor<float>
    {
        FloatTensor inner;

        public static unsafe FloatTorchTensor Create(int length)
        {
            var inner = new FloatTensor(length);
            var mem = new NativeMemory<float>((void*)inner.Data, length);
            return new FloatTorchTensor(mem.Memory, new int[] { length }, inner);
        }

        public FloatTorchTensor(Memory<float> memory, ReadOnlySpan<int> dimensions, FloatTensor inner) : base(memory, dimensions)
        {
            this.inner = inner;
        }

        public void Fill(int value)
        {
            inner.Fill(value);
        }
    }
}
