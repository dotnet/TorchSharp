using System;

namespace TorchSharp.Tensor
{
    public interface ITorchTensor : IDisposable
    {
        IntPtr Handle { get; }

        long Dimensions { get; }

        long[] Shape { get; }

        long NumberOfElements { get; }

        string Device { get; }

        ITorchTensor this[long i1] { get; }

        ITorchTensor this[long i1, long i2] { get; }

        ITorchTensor this[long i1, long i2, long i3] { get; }

        bool IsSparse { get; }

        bool IsVariable { get; }

        Span<T> Data<T>();

        T DataItem<T>();

        long GetTensorDimension(int dim);

        long GetTensorStride(int dim);

        ITorchTensor Cpu();

        ITorchTensor Cuda();

        void Backward();

        ITorchTensor Grad();

        ITorchTensor Reshape(params long[] shape);

        ITorchTensor T();

        ITorchTensor Transpose(long dimension1, long dimension2);

        void TransposeInPlace(long dimension1, long dimension2);

        ITorchTensor View(params long[] shape);

        ITorchTensor Eq(ITorchTensor target);

        bool Equal(ITorchTensor target);

        //ITorchTensor Add<T>(ITorchTensor target, T scalar = default);

        //void AddInPlace<T>(ITorchTensor target, T scalar = default);

        ITorchTensor Add(ITorchTensor target, int scalar = 1);

        void AddInPlace(ITorchTensor target, int scalar = 1);

        ITorchTensor Addbmm(ITorchTensor batch1, ITorchTensor batch2, float beta, float alpha);

        ITorchTensor Argmax(long dimension, bool keepDimension = false);

        ITorchTensor Baddbmm(ITorchTensor batch2, ITorchTensor mat, float beta, float alpha);

        ITorchTensor Bmm(ITorchTensor batch2);

        ITorchTensor Exp();

        ITorchTensor MatMul(ITorchTensor target);

        ITorchTensor Mean();

        ITorchTensor Mm(ITorchTensor target);

        ITorchTensor Mul(ITorchTensor target);

        // ITorchTensor Mul<T>(T scalar);

        ITorchTensor Mul(float scalar);

        void MulInPlace(ITorchTensor target);

        ITorchTensor Pow(float scalar);

        ITorchTensor Sigmoid();

        ITorchTensor Sub(ITorchTensor target);

        void SubInPlace(ITorchTensor target);

        ITorchTensor Sum();
    }
}
