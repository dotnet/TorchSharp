
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Sequential : Module
    {
        internal IList<Module> Modules = new List<Module>();

        internal Sequential(IntPtr handle) : base(handle)
        {
        }

        public Sequential(params Module[] modules) : base(IntPtr.Zero)
        {
            foreach (var module in modules)
            {
                Modules.Add(module);
            }
        }

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            if (Modules.Count < 1)
            {
                throw new ArgumentException("Cannot do forward pass over empty Sequence module.");
            }

            var (head, tail) = Modules;
            ITorchTensor<float> result = head.Forward(tensor);

            foreach (var module in tail)
            {
                result = module.Forward(result);
            }

            return result;
        }

        public override void ZeroGrad()
        {
            foreach (var module in Modules)
            {
                module.ZeroGrad();
            }
        }

        public override IEnumerable<ITorchTensor<float>> Parameters()
        {
            IEnumerable<ITorchTensor<float>> result = Enumerable.Empty<ITorchTensor<float>>();

            foreach (var module in Modules)
            {
                result = result.Concat(module.Parameters());
            }

            return result;
        }

        public override string[] GetModules()
        {
            string[] result = new string[Modules.Count];

            for (int i = 0; i < Modules.Count; i++)
            {
                result[i] = Modules[i].GetName();
            }

            return result;
        }
    }
}
