using System;
using System.Linq;
using System.Numerics.Tensors;

namespace TorchSharp
{
    public class FloatTorchTensor : DenseTensor<float>
    {
        FloatTensor inner;

        /// <summary>
        ///   Utility methos to create a TorchTensor.
        /// </summary>
        /// <param name="length">The desired length for the dimension of the tensor.</param>

        public static unsafe FloatTorchTensor Create(params int[] sizes)
        {
            var inner = new FloatTensor(sizes.Select(x => (long)x).ToArray());
            var mem = new NativeMemory<float>((void*)inner.Data, sizes.Aggregate((a, b) => a * b));
            return new FloatTorchTensor(mem.Memory, sizes, inner);
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
