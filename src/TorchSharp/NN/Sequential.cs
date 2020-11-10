
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
        private static extern void THSNN_Sequential_push_back(Module.HType module,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            BoxedModule.HType boxedSubModule);

        public void Add (string name, Module submodule)
        {
            Debug.Assert (!handle.IsInvalid);
            if (submodule.BoxedModule == null)
                throw new InvalidOperationException ("A Sequential or loaded module may not be added to a Sequential");

            THSNN_Sequential_push_back (handle, name, submodule.BoxedModule.handle);
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }


    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Sequential_ctor ();

        static public Sequential Sequential (params (string name, Module submodule)[] modules)
        {
            var handle = THSNN_Sequential_ctor ();
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            var res = new Sequential (handle);
            foreach (var module in modules)
                res.Add(module.name, module.submodule);
            return res;
        }

    }

}
