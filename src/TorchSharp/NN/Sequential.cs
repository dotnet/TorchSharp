// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.PInvoke.LibTorchSharp;

using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Runtime.CompilerServices;
    using Modules;
    using TorchSharp.PInvoke;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        public class Sequential : torch.nn.Module<Tensor, Tensor>
        {
            public Sequential append(string name, torch.nn.IModule<Tensor, Tensor> module)
            {
                Add(name, module);
                return this;
            }

            internal void Add(string name, torch.nn.IModule<Tensor, Tensor> sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential append(torch.nn.IModule<Tensor, Tensor> module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal void Add(torch.nn.IModule<Tensor, Tensor> module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
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
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true)) {
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, torch.nn.Module module)> named_children()
            {
                for (var i = 0; i < _names.Count; i++) {
                    yield return ($"{_names[i]}", ((torch.nn.Module)_modules[i]));
                }
            }

            public override IEnumerable<(string name, torch.nn.Module module)> named_modules()
            {
                for (var i = 0; i < _names.Count; i++) {
                    yield return ($"{_names[i]}", ((torch.nn.Module)_modules[i]));
                }

                for (var i = 0; i < _names.Count; i++) {
                    var sm = (torch.nn.Module)_modules[i];
                    var name = _names[i];
                    foreach (var (n, p) in sm.named_modules()) {
                        yield return ($"{name}.{n}", p);
                    }
                }

            }
            internal Sequential(IntPtr handle) : base(handle, IntPtr.Zero)
            {
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module<Tensor, Tensor> submodule)[] modules) : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);

                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module<Tensor, Tensor>[] modules) : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);

                foreach (var m in modules) Add(m);
            }

            public override Tensor forward(Tensor tensor)
            {
                // If there are no modules, just return a fresh handle to the input.
                if (_modules.Count == 0) return tensor.alias();

                // The loop-based logic below only works for n > 1, so here's another special case.
                if (_modules.Count == 1) return _modules[0].call(tensor);

                // Note: we have not been able to detect any significant performance difference between
                // implementing forward() in native or managed code.

                // Using an for loop helps debugging, since we can know the ordinal of the submodule.

                var t0 = _modules[0].call(tensor);

                for (var idx = 1; idx < _modules.Count - 1; idx++) {
                    var t1 = _modules[idx].call(t0);
                    t0.Dispose();
                    t0 = t1;
                }

                var result = _modules[_modules.Count - 1].call(t0);
                t0.Dispose();

                return result;
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the Sequential collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.IModule<Tensor,Tensor> this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            private Sequential Slice(int start, int end)
            {
                if (start < 0 || start > _modules.Count) throw new IndexOutOfRangeException($"{start} is not a valid index.");
                if (end < 0 || end > _modules.Count) throw new IndexOutOfRangeException($"{end} is not a valid index.");

                var stop = Math.Min(_modules.Count, end);

                var result = new Sequential(Array.Empty<torch.nn.Module<Tensor, Tensor>>());

                for (var i = start; i < stop; i++) {
                    result.Add(_names[i], _modules[i]);
                }
                return result;
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential this[System.Range index] {
                get {
                    var start = index.Start.IsFromEnd ? _modules.Count - index.Start.Value : index.Start.Value;
                    var end = index.End.IsFromEnd ? _modules.Count - index.End.Value : index.End.Value;

                    return this[(start, end)];
                }
            }
#endif
            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    foreach (var m in _modules) { ((torch.nn.Module)m).Dispose(); }
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
                foreach (var m in _modules) { ((torch.nn.Module)m).train(on); }
            }

            /// <summary>
            /// Sets the module in evaluation mode.
            /// </summary>
            /// <remarks>
            /// This has any effect only on certain modules.See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.Dropout, BatchNorm, etc.
            /// </remarks>
            public override void eval()
            {
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex = -1)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex); }
                return this;
            }

            // Currently, Sequential is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.IModule<Tensor, Tensor>> _modules = new List<nn.IModule<Tensor, Tensor>>();
            private List<string> _names = new List<string>();
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Get empty sequential
            /// </summary>
            /// <returns></returns>
            public static Sequential Sequential()
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
            public static Sequential Sequential(params (string name, torch.nn.Module<Tensor, Tensor> submodule)[] modules)
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
            public static Sequential Sequential(params torch.nn.Module<Tensor, Tensor>[] modules)
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
            public static Sequential Sequential(params System.Tuple<string, torch.nn.Module<Tensor, Tensor>>[] modules)
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
            public static Sequential Sequential(IEnumerable<(string name, torch.nn.Module<Tensor, Tensor> submodule)> modules)
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
            public static Sequential Sequential(IEnumerable<System.Tuple<string, torch.nn.Module<Tensor, Tensor>>> modules)
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
            public static Sequential Sequential(IEnumerable<torch.nn.Module<Tensor, Tensor>> modules)
            {
                var res = Sequential();
                foreach (var module in modules)
                    res.Add(module);
                return res;
            }
        }
    }
}
