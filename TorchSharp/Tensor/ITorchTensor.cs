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

        Span<T> Data { get; }

        T Item { get; }

        long GetTensorDimension(int dim);

        long GetTensorStride(int dim);

        void Backward();

        ITorchTensor<float> Grad();

        ITorchTensor<T> View(params long[] shape);

        ITorchTensor<T> SubInPlace(ITorchTensor<T> target, bool no_grad = true);

        ITorchTensor<T> Mul(T scalar, bool no_grad = true);
    }
}
