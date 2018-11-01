using System;
using System.Linq;
using System.Numerics.Tensors;
using TorchSharp;

namespace TorchTensor
{
    public sealed class FloatTorchTensor : DenseTensor<float>
    {
        internal readonly object inner;

        public FloatTensor TorchSharpTensor =>  inner as FloatTensor;

        public FloatTorchTensor(Memory<float> memory, ReadOnlySpan<int> dimensions, FloatTensor inner) : base(memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
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
            var mem = new NativeMemory<float>(inner.Data, memLen);

            return new FloatTorchTensor(mem.Memory, shape, inner);
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public unsafe override Tensor<float> Clone()
        {
            var typedInner = inner as FloatTensor;
            var innerClone = typedInner.Clone();
            var mem = new NativeMemory<float>(innerClone.Data, Buffer.Length);

            return new FloatTorchTensor(mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override unsafe Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as FloatTensor;

            switch (true)
            {
                case bool _ when typeof(TResult) == typeof(float):
                    var innerClone = new FloatTensor(typedInner.Shape);
                    innerClone.Fill(default);
                    var mem = new NativeMemory<float>(innerClone.Data, Buffer.Length);

                    return new FloatTorchTensor(mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException("Only cloning floats is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<float> Reshape(ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as FloatTensor;
            // This requires newWithStorage on the torchSharp side
            return null;
        }
    }
}
