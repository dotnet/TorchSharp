using System;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a functional module (e.g., ReLU).
    /// </summary>
    public abstract class ProvidedModule : Module
    {
        internal ProvidedModule() : base(IntPtr.Zero)
        {
        }

        internal ProvidedModule(IntPtr handle) : base(handle)
        {
        }

        public override ITorchTensor<float> Forward<T>(params ITorchTensor<T>[] tensors)
        {
            if (tensors.Length != 1)
            {
                throw new ArgumentException(nameof(tensors));
            }

            return Forward(tensors[0]);
        }

        public abstract ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor);
    }
}
