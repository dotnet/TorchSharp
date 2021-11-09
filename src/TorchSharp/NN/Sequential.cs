// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        public class Sequential : torch.nn.Module
        {
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_Sequential_push_back(torch.nn.Module.HType module,
                [MarshalAs(UnmanagedType.LPStr)] string name,
                torch.nn.BoxedModule.HType boxedSubModule);

            public void Add(string name, torch.nn.Module submodule)
            {
                Debug.Assert(!handle.IsInvalid);
                if (submodule.BoxedModule == null)
                    throw new InvalidOperationException("A Sequential or loaded module may not be added to a Sequential");

                THSNN_Sequential_push_back(handle, name, submodule.BoxedModule.handle);
                torch.CheckForErrors();
                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(submodule);
                _names.Add(name);
            }

            public void Add(torch.nn.Module module)
            {
                //if (module.GetType() == typeof(nn.Module)) throw new Exception(module.GetType().ToString());
                //throw new Exception(module.GetType().ToString());

                var type = module.GetType().ToString().Split('.')[^1].TrimEnd(Enumerable.Range('0', 10).Select(x => (char)x).ToArray()).ToLower().Replace('+', '_');
                //throw new Exception(type + (_countType.ContainsKey(type) ? ++_countType[type] : _countType[type] = 1));
                Add(type + (_countType.ContainsKey(type) ? ++_countType[type] : _countType[type] = 1), module);
            }

            internal Sequential(IntPtr handle) : base(handle, IntPtr.Zero)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Sequential_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                if (_modules.Count == 0) return tensor.clone();

                // Note: we have not been able to detect any significant performance difference between
                // implementing forward() in native or managed code.

                // Using an for loop helps debugging, since we can know the ordinal of the submodule.

                for (var idx = 0; idx < _modules.Count; idx++) {
                    tensor = _modules[idx].forward(tensor);
                }
                return tensor;
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) m.apply(fn);
                fn(this);
                return this;

            }

            // Currently, Sequential is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
            private Dictionary<string, int> _countType = new Dictionary<string, int>();
        }
    }

    public static partial class torch
    {

        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Sequential_ctor();

            /// <summary>
            /// Get empty sequential
            /// </summary>
            /// <returns></returns>
            static public Sequential Sequential()
            {
                var handle = THSNN_Sequential_ctor();
                if(handle == IntPtr.Zero) {torch.CheckForErrors();}

                return new Sequential(handle);
            }

            /// <summary>
            /// A sequential container. Modules will be added to it in the order they are passed in the constructor.
            /// Alternatively, an OrderedDict of modules can be passed in. The forward() method of Sequential accepts any input and forwards it to the first module it contains.
            /// It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
            /// The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module,
            /// such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            /// <returns></returns>
            static public Sequential Sequential(params (string name, torch.nn.Module submodule)[] modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module.name, module.submodule);
                return res;
            }

            /// <summary>
            /// A sequential container. Modules will be added to it in the order they are passed in the constructor.
            /// It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
            /// The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module,
            /// such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            /// <returns></returns>
            static public Sequential Sequential(params torch.nn.Module[] modules)
            {
                var res = Sequential();
                foreach(var m in modules)
                    res.Add(m);
                return res;
            }

            /// <summary>
            /// A sequential container. Modules will be added to it in the order they are passed in the constructor.
            /// Alternatively, an OrderedDict of modules can be passed in. The forward() method of Sequential accepts any input and forwards it to the first module it contains.
            /// It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
            /// The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module,
            /// such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            static public Sequential Sequential(IEnumerable<(string name, torch.nn.Module submodule)> modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module.name, module.submodule);
                return res;
            }

            /// <summary>
            /// A sequential container. Modules will be added to it in the order they are passed in the constructor.
            /// It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
            /// The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module,
            /// such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            /// <returns></returns>
            static public Sequential Sequential(IEnumerable<torch.nn.Module> modules)
            {
                var res = Sequential();
                foreach(var module in modules)
                    res.Add(module);
                return res;
            }
        }
    }
}
