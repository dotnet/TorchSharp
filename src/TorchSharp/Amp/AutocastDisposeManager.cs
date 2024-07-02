using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Amp
{
    public class AutocastDisposeManager
    {

        /*[ThreadStatic] private static AutocastDisposeManager _threadAutocastSingleton;

        internal static AutocastDisposeManager ThreadAutocastSingleton => _threadAutocastSingleton ??= new AutocastDisposeManager();

        internal AutocastDisposeScope CurrentAutocastDispose;
        //internal HashSet<torch.nn.Module> Modules = new List<torch.nn.Module>();
        public AutocastDisposeManager()
        {
            CurrentAutocastDispose = new AutocastDisposeScope(this);
        }
        internal AutocastDisposeScope RegisterTensorAutocastScope(torch.Tensor t)
        {
            if (CurrentAutocastDispose == null)
                return null;
            CurrentAutocastDispose.Tensors.Add(t);
            return CurrentAutocastDispose;
        }*/

    }
}
