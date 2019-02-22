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

        public override void ZeroGrad()
        {
        }

        public override bool IsTraining()
        {
            return true;
        }

        public override IEnumerable<ITorchTensor<float>> Parameters()
        {
            return new List<ITorchTensor<float>>();
        }

        public override string[] GetModules()
        {
            return new string[0];
        }
    }
}
