
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a Sequential module.
    /// </summary>
    public class Sequential : torch.nn.Module
    {
        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Sequential_push_back(torch.nn.Module.HType module,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            torch.nn.BoxedModule.HType boxedSubModule);

        public void Add (string name, torch.nn.Module submodule)
        {
            Debug.Assert (!handle.IsInvalid);
            if (submodule.BoxedModule == null)
                throw new InvalidOperationException ("A Sequential or loaded module may not be added to a Sequential");

            THSNN_Sequential_push_back (handle, name, submodule.BoxedModule.handle);
            torch.CheckForErrors ();
        }

        internal Sequential (IntPtr handle) : base (handle, IntPtr.Zero)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Sequential_forward (torch.nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Sequential_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Sequential_ctor ();

        /// <summary>
        /// A sequential container. Modules will be added to it in the order they are passed in the constructor.
        /// Alternatively, an OrderedDict of modules can be passed in. The forward() method of Sequential accepts any input and forwards it to the first module it contains.
        /// It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
        /// The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module,
        /// such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).
        /// </summary>
        /// <param name="modules">An ordered list of the contained modules.</param>
        /// <returns></returns>
        static public Sequential Sequential (params (string name, torch.nn.Module submodule)[] modules)
        {
            var handle = THSNN_Sequential_ctor ();
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            var res = new Sequential (handle);
            foreach (var module in modules)
                res.Add(module.name, module.submodule);
            return res;
        }

    }

}
