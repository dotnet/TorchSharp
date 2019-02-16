
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Sequential : Module
    {
        private IList<Module> _modules = new List<Module>();

        internal Sequential(IntPtr handle) : base(handle)
        {
        }

        public Sequential(params Module[] modules) : base(IntPtr.Zero)
        {
            foreach (var module in modules)
            {
                _modules.Add(module);
            }
        }

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            if (_modules.Count < 1)
            {
                throw new ArgumentException("Cannot do forward pass over empty Sequence module.");
            }

            var (head, tail) = _modules;
            ITorchTensor<float> result = head.Forward(tensor);

            foreach (var module in tail)
            {
                result = module.Forward(result);
            }

            return result;
        }

        public override void ZeroGrad()
        {
            foreach (var module in _modules)
            {
                module.ZeroGrad();
            }
        }

        public override IEnumerable<ITorchTensor<float>> Parameters()
        {
            IEnumerable<ITorchTensor<float>> result = Enumerable.Empty<ITorchTensor<float>>();

            foreach (var module in _modules)
            {
                result = result.Concat(module.Parameters());
            }

            return result;
        }

        public override string[] GetModules()
        {
            string[] result = new string[_modules.Count];

            for (int i = 0; i < _modules.Count; i++)
            {
                result[i] = _modules[i].GetName();
            }

            return result;
        }
    }
}
