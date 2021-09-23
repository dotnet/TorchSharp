// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

using static TorchSharp.torch;

namespace TorchSharp
{

    public static partial class torch
    {
        public static partial class nn
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
            public class ModuleDict : Module, IDictionary<string, Module>, IList<(string, Module)>
            {
                public ModuleDict(string name) : base(name)
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
                public IEnumerator<(string, Module)> items() => _list.GetEnumerator();

                /// <summary>
                /// Return the ParameterDict keys.
                /// </summary>
                /// <returns></returns>
                public IEnumerable<string> keys() => _dict.Keys;

                protected override void RegisterComponents()
                {
                    if (_registered) return;

                    for (int i = 0; i < _list.Count; i++) {
                        RegisterModule($"{_list[i].Item1}", _list[i].Item2);
                    }
                    _registered = true;
                }

                private bool _registered = false;

                /// <summary>
                /// Return the ParameterDict values.
                /// </summary>
                /// <returns></returns>
                public IEnumerable<Module> values() => _dict.Values;

                public (string, Module) this[int index] {
                    get => _list[index];
                    set => _list[index] = value; }

                public bool IsReadOnly => false;

                public ICollection<string> Keys => _list.Select(kv => kv.Item1).ToList();

                public ICollection<Module> Values => _list.Select(kv => kv.Item2).ToList();

                public int Count => _dict.Count;

                public Module this[string key] { get => _dict[key]; set => _dict[key] = value; }

                public void Add((string, Module) item)
                {
                    _dict.Add(item.Item1, item.Item2);
                    _list.Add(item);
                }

                public void Add(string key, Module value)
                {
                    _dict.Add(key, value);
                    _list.Add((key, value));
                }

                public void Add(KeyValuePair<string, Module> item)
                {
                    _dict.Add(item.Key, item.Value);
                    _list.Add((item.Key, item.Value));
                }

                public bool Contains((string, Module) item)
                {
                    return _list.Contains(item);
                }

                public void CopyTo((string, Module)[] array, int arrayIndex)
                {
                    _list.CopyTo(array, arrayIndex);
                }

                public int IndexOf((string, Module) item)
                {
                    return _list.IndexOf(item);
                }

                public void Insert(int index, (string, Module) item)
                {
                    _list.Insert(index, item);
                }

                public bool Remove((string, Module) item)
                {
                    return _list.Remove(item);
                }

                public void RemoveAt(int index)
                {
                    _list.RemoveAt(index);
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

                public bool TryGetValue(string key, [MaybeNullWhen(false)] out Module value)
                {
                    return _dict.TryGetValue(key, out value);
                }

                public void Clear()
                {
                    _dict.Clear();
                    _list.Clear();
                }

                public bool Contains(KeyValuePair<string, Module> item)
                {
                    return _dict.ContainsKey(item.Key);
                }

                public void CopyTo(KeyValuePair<string, Module>[] array, int arrayIndex)
                {
                    throw new NotImplementedException();
                }

                public bool Remove(KeyValuePair<string, Module> item)
                {
                    return _dict.Remove(item.Key);
                }

                public IEnumerator<(string, Module)> GetEnumerator()
                {
                    return _list.GetEnumerator();
                }

                IEnumerator<KeyValuePair<string, Module>> IEnumerable<KeyValuePair<string, Module>>.GetEnumerator()
                {
                    throw new NotImplementedException();
                }

                IEnumerator IEnumerable.GetEnumerator()
                {
                    return ((IEnumerable<KeyValuePair<string, Module>>)this).GetEnumerator();
                }

                private List<(string, Module)> _list = new List<(string, Module)>();
                private Dictionary<string, Module> _dict = new Dictionary<string, Module>();
            };
        }
    }
}
