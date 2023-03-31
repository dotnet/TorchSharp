// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections;
using System.Collections.Generic;
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
        public class ModuleList<T> : Module, IList<T> where T: Module
        {
            public ModuleList(params T[] modules) : base(nameof(ModuleList))
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

            public T this[int index] {
                get => _list[index];
                set => _list[index] = value;
            }

            public int Count => _list.Count;

            public bool IsReadOnly => false;

            public void Add(T item)
            {
                _list.Add(item);
            }

            /// <summary>
            /// Appends zero or more parameters to the end of the list.
            /// </summary>
            public void append(params T[] parameters)
            {
                if (parameters != null && parameters.Length > 0) {
                    _list.AddRange(parameters);
                }
            }

            public void Clear()
            {
                _list.Clear();
            }

            public bool Contains(T item)
            {
                return _list.Contains(item);
            }

            public void CopyTo(T[] array, int arrayIndex)
            {
                _list.CopyTo(array, arrayIndex);
            }

            /// <summary>
            /// Appends parameters from an enumeration to the end of the list.
            /// </summary>
            /// <param name="modules"></param>
            public void extend(IEnumerable<T> modules)
            {
                _list.AddRange(modules);
            }

            public IEnumerator<T> GetEnumerator()
            {
                return _list.GetEnumerator();
            }

            public int IndexOf(T item)
            {
                return _list.IndexOf(item);
            }

            public void Insert(int index, T item)
            {
                _list.Insert(index, item);
            }

            public bool Remove(T item)
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

            private List<T> _list = new List<T>();
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Create a ModuleList instance from an array of modules.
            /// </summary>
            /// <param name="modules">A list of modules.</param>
            /// <remarks>
            /// ModuleList can be indexed like a regular list, but modules it contains are properly registered, and will be visible by all Module methods.
            /// </remarks>
            public static ModuleList<Module> ModuleList(params Module[] modules) => new ModuleList<Module>(modules);

            /// <summary>
            /// Create a ModuleList instance from an array of modules.
            /// </summary>
            /// <param name="modules">A list of modules.</param>
            /// <remarks>
            /// ModuleList can be indexed like a regular list, but modules it contains are properly registered, and will be visible by all Module methods.
            /// </remarks>
            public static ModuleList<T> ModuleList<T>(params T[] modules) where T : Module => new ModuleList<T>(modules);
        }
    }
}
