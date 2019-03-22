using System;

namespace TorchSharp.Tensor
{
    public interface ITorchTensor<T> : IDisposable
    {
        IntPtr Handle { get; }

        int Dimensions { get; }

        long[] Shape { get; }

        long NumberOfElements { get; }

        string Device { get; }

        ITorchTensor<T> Cpu();

        ITorchTensor<T> Cuda();

        Span<T> Data { get; }

        T Item { get; }

        long GetTensorDimension(int dim);

        long GetTensorStride(int dim);

        void Backward();

        ITorchTensor<float> Grad();

        ITorchTensor<T> View(params long[] shape);

        ITorchTensor<U> Eq<U>(ITorchTensor<U> target);

        ITorchTensor<T> SubInPlace(ITorchTensor<T> target, bool noGrad = true);

        ITorchTensor<T> Mul(T scalar, bool noGrad = true);

        ITorchTensor<T> Sum();

        ITorchTensor<T> Argmax(long dimension, bool keepDimension = false);
    }
}
