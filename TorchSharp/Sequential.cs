
using System;
using System.Collections.Generic;

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

        public override FloatTensor Forward(FloatTensor tensor)
        {
            FloatTensor result = tensor;

            foreach (var module in _modules)
            {
                result = module.Forward(result);
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
