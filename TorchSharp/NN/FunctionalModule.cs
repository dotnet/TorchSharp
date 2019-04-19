using System;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a functional module (e.g., ReLU).
    /// </summary>
    public abstract class FunctionalModule<T> : ProvidedModule
    {
        internal FunctionalModule() : base(IntPtr.Zero)
        {
        }

        public override void RegisterModule(Module module)
        {
        }

        public override void ZeroGrad()
        {
        }

        public override IEnumerable<(string name, ITorchTensor parameter)> NamedParameters()
        {
            return new List<(string, ITorchTensor)>();
        }

        public override IEnumerable<ITorchTensor> Parameters()
        {
            return new List<ITorchTensor>();
        }

        public override IEnumerable<string> GetModules()
        {
            return new string[0];
        }

        public override string GetName()
        {
            return typeof(T).Name;
        }
    }
}
