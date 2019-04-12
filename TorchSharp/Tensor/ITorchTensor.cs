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

        bool IsSparse { get; }

        bool IsVariable { get; }

        long GetTensorDimension(int dim);

        long GetTensorStride(int dim);

        ITorchTensor<T> Cpu();

        ITorchTensor<T> Cuda();

        void Backward();

        ITorchTensor<float> Grad();

        ITorchTensor<T> Reshape(params long[] shape);

        ITorchTensor<T> View(params long[] shape);

        ITorchTensor<U> Eq<U>(ITorchTensor<U> target);

        bool Equal<U>(ITorchTensor<U> target);

        ITorchTensor<T> Add(ITorchTensor<T> target, int scalar);

        void AddInPlace(ITorchTensor<T> target, int scalar);

        ITorchTensor<T> Addbmm(ITorchTensor<T> batch1, ITorchTensor<T> batch2, float beta, float alpha);

        ITorchTensor<T> Argmax(long dimension, bool keepDimension = false);

        ITorchTensor<T> Baddbmm(ITorchTensor<T> batch2, ITorchTensor<T> mat, float beta, float alpha);

        ITorchTensor<T> Bmm(ITorchTensor<T> batch2);

        ITorchTensor<T> Exp();

        ITorchTensor<T> MatMul(ITorchTensor<T> target);

        ITorchTensor<T> Mean();

        ITorchTensor<T> Mm(ITorchTensor<T> target);

        ITorchTensor<T> Mul(ITorchTensor<T> target);

        ITorchTensor<T> Mul(T scalar);

        void MulInPlace(ITorchTensor<T> target);

        ITorchTensor<T> Pow(float scalar);

        ITorchTensor<T> Sigmoid();

        ITorchTensor<T> Sub(ITorchTensor<T> target);

        void SubInPlace(ITorchTensor<T> target);

        ITorchTensor<T> Sum();
    }
}
