using System;
using System.Linq;
using System.Numerics.Tensors;

namespace TorchSharp
{
    public sealed class FloatTorchTensor : DenseTensor<float>
    {
        private readonly FloatTensor inner;

        /// <summary>
        ///   Utility method to create an empty TorchTensor.
        ///   This is currently failing because SNT does not support zero-size tensors
        /// </summary>
        public static unsafe FloatTorchTensor Create()
        {
            var inner = new FloatTensor();
            return new FloatTorchTensor(new Memory<float>(), new int[1] { 0 }, inner);
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

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<float> Clone()
        {
            var innerClone = new FloatTensor(inner.Shape);
            return new FloatTorchTensor(Buffer.ToArray(), Dimensions, inner);
        }
    }
}
