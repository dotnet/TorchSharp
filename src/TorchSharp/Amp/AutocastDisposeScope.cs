using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Amp
{
    public sealed class AutocastDisposeScope : IDisposable
    {
        //private AutocastDisposeManager autocastDisposeManager;
        public bool IsEnabled;
        /*internal AutocastMode autocastMode = AutocastMode.GetInstance();
        internal HashSet<torch.Tensor> Tensors = new HashSet<torch.Tensor>();
        public AutocastDisposeScope(AutocastDisposeManager autocastDisposeManager)
        {
            this.autocastDisposeManager = autocastDisposeManager;
            IsEnabled = true;
        }*/
        public void Dispose()
        {
            IsEnabled = false;
        }
    }
}
