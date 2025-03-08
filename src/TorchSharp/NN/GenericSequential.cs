// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.PInvoke.NativeMethods;

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
        /// <remarks>
        /// This version of Sequential accepts modules of any type.
        /// The submodules are not statically typed, so any `forward` implementation will have to validate the module
        /// types dynamically and cast to the right type in order to invoke them.
        /// The constructor of a derived type should validate that the entered modules are of the expected types, as should
        /// the `forward` implementation. Without dynamic type validation, there is a significant risk of silent failure.
        ///
        /// Please note that if all submodules have forward methods taking a single Tensor and returning a single Tensor, then
        /// using the non-generic Sequential will be better -- it means not having to derive a class or implement the forward
        /// method, and the implementation is more efficient since there is no need for dynamic type checking of the submodules.
        /// </remarks>
        public abstract class Sequential<T, TResult> : nn.Module<T, TResult>
        {
            public Sequential<T, TResult> append(string name, torch.nn.Module module)
            {
                Add(name, module);
                return this;
            }

            internal protected void Add(string name, torch.nn.Module sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential<T, TResult> append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal protected void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid parameters

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_persistent = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid buffers

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true, include_persistent)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
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

            internal Sequential() : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module submodule)[] modules) : this()
            {
                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module[] modules) : this()
            {
                foreach (var m in modules) Add(m);
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the SequentialAbstractBase collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.Module this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T, TResult> this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            /// <summary>
            /// Slicing the Sequential from `start` to `end'
            /// </summary>
            /// <remarks>
            /// Any implementation of `Slice` must take into consideration that the first layer
            /// in the slice must adhere to the same forward semantics as the containing class itself.
            /// </remarks>
            protected virtual Sequential<T, TResult> Slice(int start, int end)
            {
                throw new System.NotImplementedException();
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T, TResult> this[System.Range index] {
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
                base.train(on);
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
                base.eval();
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex, non_blocking); }
                return this;
            }

            // Currently, SequentialAbstractBase is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
        }

        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        /// <remarks>
        /// This version of Sequential accepts modules of any type.
        /// The submodules are not statically typed, so any `forward` implementation will have to validate the module
        /// types dynamically and cast to the right type in order to invoke them.
        /// The constructor of a derived type should validate that the entered modules are of the expected types, as should
        /// the `forward` implementation. Without dynamic type validation, there is a significant risk of silent failure.
        /// </remarks>
        public abstract class Sequential<T1, T2, TResult> : nn.Module<T1, T2, TResult>
        {
            public Sequential<T1, T2, TResult> append(string name, torch.nn.Module module)
            {
                Add(name, module);
                return this;
            }

            internal protected void Add(string name, torch.nn.Module sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential<T1, T2, TResult> append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal protected void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid parameters

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_persistent = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid buffers

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true, include_persistent)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
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

            internal Sequential() : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module submodule)[] modules) : this()
            {
                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module[] modules) : this()
            {
                foreach (var m in modules) Add(m);
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the SequentialAbstractBase collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.Module this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, TResult> this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            /// <summary>
            /// Slicing the Sequential from `start` to `end'
            /// </summary>
            /// <remarks>
            /// Any implementation of `Slice` must take into consideration that the first layer
            /// in the slice must adhere to the same forward semantics as the containing class itself.
            /// </remarks>
            protected virtual Sequential<T1, T2, TResult> Slice(int start, int end)
            {
                throw new System.NotImplementedException();
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, TResult> this[System.Range index] {
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
                base.train(on);
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
                base.eval();
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex, non_blocking); }
                return this;
            }

            // Currently, SequentialAbstractBase is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
        }

        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        /// <remarks>
        /// This version of Sequential accepts modules of any type.
        /// The submodules are not statically typed, so any `forward` implementation will have to validate the module
        /// types dynamically and cast to the right type in order to invoke them.
        /// The constructor of a derived type should validate that the entered modules are of the expected types, as should
        /// the `forward` implementation. Without dynamic type validation, there is a significant risk of silent failure.
        /// </remarks>
        public abstract class Sequential<T1, T2, T3, TResult> : nn.Module<T1, T2, T3, TResult>
        {
            public Sequential<T1, T2, T3, TResult> append(string name, torch.nn.Module module)
            {
                Add(name, module);
                return this;
            }

            internal protected void Add(string name, torch.nn.Module sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential<T1, T2, T3, TResult> append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal protected void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid parameters

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_persistent = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid buffers

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true, include_persistent)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
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

            internal Sequential() : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module submodule)[] modules) : this()
            {
                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module[] modules) : this()
            {
                foreach (var m in modules) Add(m);
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the SequentialAbstractBase collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.Module this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, TResult> this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            protected virtual Sequential<T1, T2, T3, TResult> Slice(int start, int end)
            {
                throw new System.NotImplementedException();
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, TResult> this[System.Range index] {
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
                base.train(on);
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
                base.eval();
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex, non_blocking); }
                return this;
            }

            // Currently, SequentialAbstractBase is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
        }

        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        /// <remarks>
        /// This version of Sequential accepts modules of any type.
        /// The submodules are not statically typed, so any `forward` implementation will have to validate the module
        /// types dynamically and cast to the right type in order to invoke them.
        /// The constructor of a derived type should validate that the entered modules are of the expected types, as should
        /// the `forward` implementation. Without dynamic type validation, there is a significant risk of silent failure.
        /// </remarks>
        public abstract class Sequential<T1, T2, T3, T4, TResult> : nn.Module<T1, T2, T3, T4,TResult>
        {
            public Sequential<T1, T2, T3, T4,TResult> append(string name, torch.nn.Module module)
            {
                Add(name, module);
                return this;
            }

            internal protected void Add(string name, torch.nn.Module sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential<T1, T2, T3, T4,TResult> append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal protected void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid parameters

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_persistent = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid buffers

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true, include_persistent)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
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

            internal Sequential() : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module submodule)[] modules) : this()
            {
                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module[] modules) : this()
            {
                foreach (var m in modules) Add(m);
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the SequentialAbstractBase collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.Module this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, T4, TResult> this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            protected virtual Sequential<T1, T2, T3, T4, TResult> Slice(int start, int end)
            {
                throw new System.NotImplementedException();
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, T4, TResult> this[System.Range index] {
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
                base.train(on);
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
                base.eval();
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex, non_blocking); }
                return this;
            }

            // Currently, SequentialAbstractBase is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
        }

        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        /// <remarks>
        /// This version of Sequential accepts modules of any type.
        /// The submodules are not statically typed, so any `forward` implementation will have to validate the module
        /// types dynamically and cast to the right type in order to invoke them.
        /// The constructor of a derived type should validate that the entered modules are of the expected types, as should
        /// the `forward` implementation. Without dynamic type validation, there is a significant risk of silent failure.
        /// </remarks>
        public abstract class Sequential<T1, T2, T3, T4, T5, TResult> : nn.Module<T1, T2, T3, T4, T5, TResult>
        {
            public Sequential<T1, T2, T3, T4, T5, TResult> append(string name, torch.nn.Module module)
            {
                Add(name, module);
                return this;
            }

            internal protected void Add(string name, torch.nn.Module sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential<T1, T2, T3, T4, T5, TResult> append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal protected void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid parameters

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_persistent = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid buffers

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true, include_persistent)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
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

            internal Sequential() : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module submodule)[] modules) : this()
            {
                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module[] modules) : this()
            {
                foreach (var m in modules) Add(m);
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the SequentialAbstractBase collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.Module this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, T4, T5, TResult> this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            protected virtual Sequential<T1, T2, T3, T4, T5, TResult> Slice(int start, int end)
            {
                throw new System.NotImplementedException();
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, T4, T5, TResult> this[System.Range index] {
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
                base.train(on);
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
                base.eval();
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex, non_blocking); }
                return this;
            }

            // Currently, SequentialAbstractBase is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
        }

        /// <summary>
        /// This class is used to represent a Sequential module.
        /// </summary>
        /// <remarks>
        /// This version of Sequential accepts modules of any type.
        /// The submodules are not statically typed, so any `forward` implementation will have to validate the module
        /// types dynamically and cast to the right type in order to invoke them.
        /// The constructor of a derived type should validate that the entered modules are of the expected types, as should
        /// the `forward` implementation. Without dynamic type validation, there is a significant risk of silent failure.
        /// </remarks>
        public abstract class Sequential<T1, T2, T3, T4, T5, T6, TResult> : nn.Module<T1, T2, T3, T4, T5, T6, TResult>
        {
            public Sequential<T1, T2, T3, T4, T5, T6, TResult> append(string name, torch.nn.Module module)
            {
                Add(name, module);
                return this;
            }

            internal protected void Add(string name, torch.nn.Module sm)
            {
                var submodule = (torch.nn.Module)sm;

                Debug.Assert(!handle.IsInvalid);
                Debug.Assert(!submodule.handle.IsInvalid);

                // Keep the sub-module alive for at least as long as the Sequential object is alive.
                _modules.Add(sm);
                _names.Add(name);
            }

            public Sequential<T1, T2, T3, T4, T5, T6, TResult> append(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
                return this;
            }

            internal protected void Add(torch.nn.Module module)
            {
                var name = _modules.Count.ToString();
                Add(name, module);
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid parameters

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_parameters(true)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
                        yield return ($"{_names[i]}.{n}", p);
                    }
                }
            }

            public override IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_persistent = true)
            {
                if (!recurse) yield break;

                var seen = new HashSet<IntPtr>();
                seen.Add(IntPtr.Zero);           // Ignore invalid buffers

                for (var i = 0; i < _names.Count; i++) {
                    foreach (var (n, p) in ((torch.nn.Module)_modules[i]).named_buffers(true, include_persistent)) {
                        if (seen.Contains(p.handle)) continue;
                        seen.Add(p.handle);
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

            internal Sequential() : base(IntPtr.Zero, IntPtr.Zero)
            {
                var handle = THSNN_Sequential_ctor();
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                this.handle = new HType(handle, true);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params (string name, torch.nn.Module submodule)[] modules) : this()
            {
                foreach (var module in modules) Add(module.name, module.submodule);
            }

            /// <summary>
            /// Constructor, intended for derived modules.
            /// </summary>
            /// <param name="modules">An ordered list of the contained modules.</param>
            protected Sequential(params torch.nn.Module[] modules) : this()
            {
                foreach (var m in modules) Add(m);
            }

            public override nn.Module apply(Action<nn.Module> fn)
            {
                // More efficient than asking C++ for the children. We already have the list, after all.
                foreach (var m in _modules) ((torch.nn.Module)m).apply(fn);
                fn(this);
                return this;

            }

            /// <summary>
            /// The number of modules in the SequentialAbstractBase collection.
            /// </summary>
            public int Count => _modules.Count;

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public nn.Module this[int index] {
                get {
                    return _modules[index];
                }
            }

            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, T4, T5, T6, TResult> this[(int? start, int? end) index] {
                get {
                    var start = index.start.HasValue ? index.start.Value : 0;
                    var end = index.end.HasValue ? index.end.Value : _modules.Count;

                    return Slice(start, end);
                }
            }

            protected virtual Sequential<T1, T2, T3, T4, T5, T6, TResult> Slice(int start, int end)
            {
                throw new System.NotImplementedException();
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Module indexer.
            /// </summary>
            [IndexerName("SequentialItems")]
            public Sequential<T1, T2, T3, T4, T5, T6, TResult> this[System.Range index] {
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
                base.train(on);
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
                base.eval();
                foreach (var m in _modules) { ((torch.nn.Module)m).eval(); }
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(device, dtype, non_blocking); }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                foreach (var m in _modules) { ((torch.nn.Module)m)._to(deviceType, deviceIndex, non_blocking); }
                return this;
            }

            // Currently, SequentialAbstractBase is implemented entirely in managed code, but if we go back to
            // using the native forward() implementation, this collection is still necessary.
            // The module handles are held in the native runtime, which calls back into managed code,
            // the .NET module instances need to stay alive, and keeping a list of them will do that.

            private List<torch.nn.Module> _modules = new List<nn.Module>();
            private List<string> _names = new List<string>();
        }
    }
}
