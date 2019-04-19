using System;

namespace TorchSharp.Tensor
{
    public interface ITorchTensor<U> : IDisposable
    {
        IntPtr Handle { get; }

        int Dimensions { get; }

        long[] Shape { get; }

        long NumberOfElements { get; }

        string Device { get; }

        Span<U> Data { get; }

        U DataItem { get; }

        ITorchTensor<U> this[long i1] { get; }

        ITorchTensor<U> this[long i1, long i2] { get; }

        ITorchTensor<U> this[long i1, long i2, long i3] { get; }

        bool IsSparse { get; }

        bool IsVariable { get; }

        long GetTensorDimension(int dim);

        long GetTensorStride(int dim);

        ITorchTensor<U> Cpu();

        ITorchTensor<U> Cuda();

        void Backward();

        ITorchTensor<float> Grad();

        ITorchTensor<U> Reshape(params long[] shape);

        ITorchTensor<U> T();

        ITorchTensor<U> Transpose(long dimension1, long dimension2);

        void TransposeInPlace(long dimension1, long dimension2);

        ITorchTensor<U> View(params long[] shape);

        ITorchTensor<U> Eq<U>(ITorchTensor<U> target);

        bool Equal<U>(ITorchTensor<U> target);

        ITorchTensor<U> Add(ITorchTensor<U> target, int scalar = 1);

        void AddInPlace(ITorchTensor<U> target, int scalar);

        ITorchTensor<U> Addbmm(ITorchTensor<U> batch1, ITorchTensor<U> batch2, float beta, float alpha);

        ITorchTensor<U> Argmax(long dimension, bool keepDimension = false);

        ITorchTensor<U> Baddbmm(ITorchTensor<U> batch2, ITorchTensor<U> mat, float beta, float alpha);

        ITorchTensor<U> Bmm(ITorchTensor<U> batch2);

        ITorchTensor<U> Exp();

        ITorchTensor<U> MatMul(ITorchTensor<U> target);

        ITorchTensor<U> Mean();

        ITorchTensor<U> Mm(ITorchTensor<U> target);

        ITorchTensor<U> Mul(ITorchTensor<U> target);

        ITorchTensor<U> Mul(U scalar);

        void MulInPlace(ITorchTensor<U> target);

        ITorchTensor<U> Pow(float scalar);

        ITorchTensor<U> Sigmoid();

        ITorchTensor<U> Sub(ITorchTensor<U> target);

        void SubInPlace(ITorchTensor<U> target);

        ITorchTensor<U> Sum();
    }
}
