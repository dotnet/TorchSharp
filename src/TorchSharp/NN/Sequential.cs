// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            public void append(string name, torch.nn.Module module)
            {
                Add(name, module);
            }

            internal void Add(string name, torch.nn.Module submodule)
            {
                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(submodule);
                _names.Add(name);
            }

            public void append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            internal void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in _modules[i].named_parameters(true)) {
                        if (seen.Contains(p.Handle)) continue;
                        seen.Add(p.Handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true)
            {
                if (!recurse) yield break;

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in _modules[i].named_buffers(true)) {
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, torch.nn.Module module)> named_children()
            {
                for (var i = 0; i < _names.Count; i++) {
                    yield return ($"{_names[i]}", _modules[i]);
                }
            }

            public override IEnumerable<(string name, torch.nn.Module module)> named_modules()
            {
                for (var i = 0; i < _names.Count; i++) {
                    yield return ($"{_names[i]}", _modules[i]);
                }

                for (var i = 0; i < _names.Count; i++) {
                    var sm = _modules[i];
                    var name = _names[i];
                    foreach (var (n, p) in sm.named_modules()) {
                        yield return ($"{name}.{n}", p);
                    }
                }

            }
            internal Sequential(IntPtr handle) : base(handle, IntPtr.Zero)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                // If there are no modules, just return a fresh handle to the input.
                if (_modules.Count == 0) return tensor.alias();

                // The loop-based logic below only works for n > 1, so here's another special case.
                if (_modules.Count == 1) return _modules[0].forward(tensor);

                // Note: we have not been able to detect any significant performance difference between
                // implementing forward() in native or managed code.

                // Using an for loop helps debugging, since we can know the ordinal of the submodule.

                var t0 = _modules[0].forward(tensor);

                for (var idx = 1; idx < _modules.Count - 1; idx++) {
                    var t1 = _modules[idx].forward(t0);
                    t0.Dispose();
                    t0 = t1;
                }

                var result = _modules[_modules.Count - 1].forward(t0);
                t0.Dispose();

                return result;
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) m.apply(fn);
                fn(this);
                return this;

            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    foreach (var m in _modules) { m.Dispose(); }
                }
                base.Dispose(disposing);
            }

            /// <summary>
            /// Sets the module in training mode.
            /// </summary>
            /// <remarks>
            /// This has any effect only on certain modules.See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.Dropout, BatchNorm, etc.
            /// </remarks>
            public override void train(bool on = true)
            {
                foreach (var m in _modules) { m.train(on); }
            }

            /// <summary>
            /// Sets the module in evaluation mode.
            /// </summary>
            /// <remarks>
            /// This has any effect only on certain modules.See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.Dropout, BatchNorm, etc.
            /// </remarks>
            public override void eval()
            {
                foreach (var m in _modules) { m.eval(); }
            }

            internal protected override nn.Module _to(ScalarType dtype)
            {
                foreach (var m in _modules) { m._to(dtype); }
                return this;
            }

            internal protected override nn.Module _to(Device device, ScalarType dtype)
            {
                foreach (var m in _modules) { m._to(device, dtype); }
                return this;
            }

            internal protected override nn.Module _to(DeviceType deviceType, int deviceIndex = -1)
            {
                foreach (var m in _modules) { m._to(deviceType, deviceIndex); }
                return this;
            }

            // Currently, Sequential is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
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
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

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
            /// <remarks>Sequential will take ownership of the modules and dispose of them when disposed.</remarks>
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
            /// <remarks>Sequential will take ownership of the modules and dispose of them when disposed.</remarks>
            static public Sequential Sequential(params torch.nn.Module[] modules)
            {
                var res = Sequential();
                foreach (var m in modules)
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
            /// <remarks>Sequential will take ownership of the modules and dispose of them when disposed.</remarks>
            static public Sequential Sequential(params System.Tuple<string, torch.nn.Module>[] modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module.Item1, module.Item2);
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
            /// <remarks>Sequential will take ownership of the modules and dispose of them when disposed.</remarks>
            static public Sequential Sequential(IEnumerable<(string name, torch.nn.Module submodule)> modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module.name, module.submodule);
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
            /// <remarks>Sequential will take ownership of the modules and dispose of them when disposed.</remarks>
            static public Sequential Sequential(IEnumerable<System.Tuple<string, torch.nn.Module>> modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module.Item1, module.Item2);
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
            /// <remarks>Sequential will take ownership of the modules and dispose of them when disposed.</remarks>
            static public Sequential Sequential(IEnumerable<torch.nn.Module> modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module);
                return res;
            }
        }
    }
}
