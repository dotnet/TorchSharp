
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a Sequential module.
    /// </summary>
    public class Sequential : Module
    {
        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Sequential_push_back(Module.HType module, string name, Module.HType boxedSubModule);

        public void Add (string name, Module submodule)
        {
            Debug.Assert (!handle.IsInvalid);
            Debug.Assert (!submodule.boxedHandle.IsInvalid);
            THSNN_Sequential_push_back (handle, name, submodule.boxedHandle);
            Torch.CheckForErrors ();
        }

        internal Sequential (IntPtr handle) : base (handle, IntPtr.Zero)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Sequential_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Sequential_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }


    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Sequential_ctor ();

        static public Sequential Sequential (params (string name, Module module)[] modules)
        {
            var handle = THSNN_Sequential_ctor ();
            Torch.CheckForErrors ();
            var res = new Sequential (handle);
            foreach (var module in modules)
                res.Add(module.name, module.module);
            return res;
        }

    }

}
