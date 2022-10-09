// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Holds parameters in a dictionary.
        /// 
        /// ModuleDict can be indexed like a regular dictionary, but the modules it
        /// contains are properly registered, and will be visible by all Module methods.
        ///
        /// ModuleDict is an ordered dictionary that respects the order of insertion, and
        /// in update(), the order of the merged OrderedDict or another ModuleDict (the argument to update()).
        /// </summary>
        public class ModuleDict<T> : Module, IDictionary<string, T>, IList<(string, T)> where T : Module
        {
            public ModuleDict() : base(nameof(ModuleDict))
            {
            }

            /// <summary>
            /// Remove all items from the ParameterDict.
            /// </summary>
            public void clear()
            {
                _list.Clear();
                _dict.Clear();
            }

            /// <summary>
            /// Return an enumeration of the ParameterDict key/value pairs.
            /// </summary>
            /// <returns></returns>
            public IEnumerator<(string, T)> items() => _list.GetEnumerator();

            /// <summary>
            /// Return the ParameterDict keys.
            /// </summary>
            /// <returns></returns>
            public IEnumerable<string> keys() => _dict.Keys;

            protected override void RegisterComponents()
            {
                if (_registered) return;

                for (int i = 0; i < _list.Count; i++) {
                    register_module($"{_list[i].Item1}", _list[i].Item2);
                }
                _registered = true;
            }

            private bool _registered = false;

            /// <summary>
            /// Return the ParameterDict values.
            /// </summary>
            /// <returns></returns>
            public IEnumerable<T> values() => _dict.Values;

            public (string, T) this[int index] {
                get => _list[index];
                set {
                    var name = value.Item1;
                    _list[index] = value;
                    _dict[name] = value.Item2;
                }
            }

            public bool IsReadOnly => false;

            public ICollection<string> Keys => _list.Select(kv => kv.Item1).ToList();

            public ICollection<T> Values => _list.Select(kv => kv.Item2).ToList();

            public int Count => _dict.Count;

            public T this[string key] {
                get => _dict[key];
                set {
                    _dict[key] = value;
                    var idx = _list.FindIndex(kv => kv.Item1.Equals(key));
                    _list[idx] = (key, value);
                }
            }

            public void Add((string, T) item)
            {
                _dict.Add(item.Item1, item.Item2);
                _list.Add(item);
            }

            public void Add(string key, T value)
            {
                _dict.Add(key, value);
                _list.Add((key, value));
            }

            public void Add(KeyValuePair<string, T> item)
            {
                _dict.Add(item.Key, item.Value);
                _list.Add((item.Key, item.Value));
            }

            public bool Contains((string, T) item)
            {
                return _list.Contains(item);
            }

            public void CopyTo((string, T)[] array, int arrayIndex)
            {
                _list.CopyTo(array, arrayIndex);
            }

            public int IndexOf((string, T) item)
            {
                return _list.IndexOf(item);
            }

            public void Insert(int index, (string, T) item)
            {
                _dict.Add(item.Item1, item.Item2);
                _list.Insert(index, item);
            }

            public bool Remove((string, T) item)
            {
                _dict.Remove(item.Item1);
                return _list.Remove(item);
            }

            public void RemoveAt(int index)
            {
                if (index >= _list.Count) throw new IndexOutOfRangeException();
                var (n, p) = _list[index];
                _list.RemoveAt(index);
                _dict.Remove(n);
            }

            public bool ContainsKey(string key)
            {
                return _dict.ContainsKey(key);
            }

            public bool Remove(string key)
            {
                var value = _dict[key];
                return _dict.Remove(key) && _list.Remove((key, value));
            }

            public bool TryGetValue(string key, [MaybeNullWhen(false)] out T value)
            {
                return _dict.TryGetValue(key, out value);
            }

            public void Clear()
            {
                _dict.Clear();
                _list.Clear();
            }

            public bool Contains(KeyValuePair<string, T> item)
            {
                return _dict.ContainsKey(item.Key);
            }

            public void CopyTo(KeyValuePair<string, T>[] array, int arrayIndex)
            {
                throw new NotImplementedException();
            }

            public bool Remove(KeyValuePair<string, T> item)
            {
                return _dict.Remove(item.Key);
            }

            public IEnumerator<(string, T)> GetEnumerator()
            {
                return _list.GetEnumerator();
            }

            IEnumerator<KeyValuePair<string, T>> IEnumerable<KeyValuePair<string, T>>.GetEnumerator()
            {
                throw new NotImplementedException();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return ((IEnumerable<KeyValuePair<string, T>>)this).GetEnumerator();
            }

            private List<(string, T)> _list = new List<(string, T)>();
            private Dictionary<string, T> _dict = new Dictionary<string, T>();
        };
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Create a ModuleDict instance from an array of modules.
            /// </summary>
            /// <param name="modules">A list of (name,module) tuples.</param>
            /// <returns></returns>
            /// <remarks>
            /// ModuleDict can be indexed like a regular dictionary, but the modules it
            /// contains are properly registered, and will be visible by all Module methods.
            ///
            /// ModuleDict is an ordered dictionary that respects the order of insertion, and
            /// in update(), the order of the merged OrderedDict or another ModuleDict (the argument to update()).
            /// </remarks>
            public static ModuleDict<Module> ModuleDict(params (string, Module)[] modules)
            {
                var result = new ModuleDict<Module>();
                foreach (var (n, m) in modules) {
                    result.Add((n, m));
                }
                return result;
            }

            /// <summary>
            /// Create a ModuleDict instance from an array of modules.
            /// </summary>
            /// <param name="modules">A list of (name,module) tuples.</param>
            /// <returns></returns>
            /// <remarks>
            /// ModuleDict can be indexed like a regular dictionary, but the modules it
            /// contains are properly registered, and will be visible by all Module methods.
            ///
            /// ModuleDict is an ordered dictionary that respects the order of insertion, and
            /// in update(), the order of the merged OrderedDict or another ModuleDict (the argument to update()).
            /// </remarks>
            public static ModuleDict<T> ModuleDict<T>(params (string, T)[] modules) where T: Module
            {
                var result = new ModuleDict<T>();
                foreach (var (n, m) in modules) {
                    result.Add((n, m));
                }
                return result;
            }
        }
    }
}
