using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a functional module (e.g., ReLU).
    /// </summary>
    public abstract class FunctionalModule : Module
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

        public override IEnumerable<ITorchTensor<float>> Parameters()
        {
            return new List<ITorchTensor<float>>();
        }

        public override IEnumerable<string> GetModules()
        {
            return new string[0];
        }
    }
}
