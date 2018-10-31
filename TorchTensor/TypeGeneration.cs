using System;
using System.Linq;
using System.Numerics.Tensors;
using TorchSharp;

namespace TorchTensor
{
    public sealed class FloatTorchTensor : DenseTensor<float>
    {
        internal readonly FloatTensor inner;

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static unsafe FloatTorchTensor Create(params int[] sizes)
        {
            var memLen = 0;
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }
            else
            {
                memLen = sizes.Aggregate((a, b) => a * b);
            }

            var inner = new FloatTensor(sizes.Select(x => (long)x).ToArray());
            var mem = new NativeMemory<float>((void*)inner.Data, memLen);

            return new FloatTorchTensor(mem.Memory, shape, inner);
        }

        public FloatTorchTensor(Memory<float> memory, ReadOnlySpan<int> dimensions, FloatTensor inner) : base(memory, dimensions)
        {
            this.inner = inner;
        }
    }
}
