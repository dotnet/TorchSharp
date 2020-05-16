
using System;
using System.Collections.Generic;
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
        internal Sequential (IntPtr handle, IEnumerable<Module> modules) : base (handle)
        {
            foreach (var module in modules) {
                RegisterModule (module);
            }
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

        static public Sequential Sequential (params Module[] modules)
        {
            var handle = THSNN_Sequential_ctor ();
            Torch.CheckForErrors ();
            return new Sequential (handle, modules);
        }
    }

}
