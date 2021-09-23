// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Holds modules in a list.
        /// ModuleList can be indexed like a regular list, but modules it contains are properly registered, and will be visible by all Module methods.
        /// </summary>
        public class ModuleList : Module, IList<Module>
        {
            public ModuleList(params Module[] modules) : base(nameof(ModuleList))
            {
                if (modules != null && modules.Length > 0) {
                    _list.AddRange(modules);
                }
            }

            protected override void RegisterComponents()
            {
                if (_registered) return;

                for (int i = 0; i < _list.Count; i++) {
                    register_module($"{i}", _list[i]);
                }
                _registered = true;
            }

            private bool _registered = false;

            public Module this[int index] {
                get => _list[index];
                set => _list[index] = value;
            }

            public int Count => _list.Count;

            public bool IsReadOnly => false;

            public void Add(Module item)
            {
                _list.Add(item);
            }

            /// <summary>
            /// Appends zero or more parameters to the end of the list.
            /// </summary>
            public void append(params Module[] parameters)
            {
                if (parameters != null && parameters.Length > 0) {
                    _list.AddRange(parameters);
                }
            }

            public void Clear()
            {
                _list.Clear();
            }

            public bool Contains(Module item)
            {
                return _list.Contains(item);
            }

            public void CopyTo(Module[] array, int arrayIndex)
            {
                _list.CopyTo(array, arrayIndex);
            }

            /// <summary>
            /// Appends parameters from an enumeration to the end of the list.
            /// </summary>
            /// <param name="parameters"></param>
            public void extend(IEnumerable<Module> parameters)
            {
                _list.AddRange(parameters);
            }

            public IEnumerator<Module> GetEnumerator()
            {
                return _list.GetEnumerator();
            }

            public int IndexOf(Module item)
            {
                return _list.IndexOf(item);
            }

            public void Insert(int index, Module item)
            {
                _list.Insert(index, item);
            }

            public bool Remove(Module item)
            {
                return _list.Remove(item);
            }

            public void RemoveAt(int index)
            {
                _list.RemoveAt(index);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            private List<Module> _list = new List<Module>();
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public static ModuleList ModuleList(params Module[] modules) => new ModuleList(modules);
        }
    }
}
