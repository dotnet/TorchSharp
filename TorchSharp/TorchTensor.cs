using System;
using System.Linq;
using System.Numerics.Tensors;

namespace TorchSharp
{
    public class FloatTorchTensor : DenseTensor<float>
    {
        FloatTensor inner;

        /// <summary>
        ///   Utility method to create an empty TorchTensor.
        /// </summary>
        public static unsafe FloatTorchTensor Create()
        {
            var inner = new FloatTensor();
            // Note that for the moment we need to create a tensors of size 1 because
            // SNT does not support zero-size tensors (for the moment)
            return new FloatTorchTensor(new Memory<float>(), new int[1] { 1 }, inner);
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static unsafe FloatTorchTensor Create(params int[] sizes)
        {
            if (sizes.Length == 0)
            {
                return Create();
            }

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
