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
        /// ParameterDict can be indexed like a regular dictionary, but the parameters it
        /// contains are properly registered, and will be visible by all Module methods.
        ///
        /// ParameterDict is an ordered dictionary that respects the order of insertion, and
        /// in update(), the order of the merged OrderedDict or another ParameterDict (the argument to update()).
        /// </summary>
        public class ParameterDict : Module, IDictionary<string, Parameter>, IList<(string, Parameter)>
        {
            public ParameterDict() : base(nameof(ParameterDict))
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
            public IEnumerator<(string, Parameter)> items() => _list.GetEnumerator();

            /// <summary>
            /// Return the ParameterDict keys.
            /// </summary>
            /// <returns></returns>
            public IEnumerable<string> keys() => _dict.Keys;

            protected override void RegisterComponents()
            {
                if (_registered) return;

                for (int i = 0; i < _list.Count; i++) {
                    register_parameter($"{_list[i].Item1}", _list[i].Item2);
                }
                _registered = true;
            }

            private bool _registered = false;

            protected internal override Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                base._to(deviceType, deviceIndex, non_blocking);
                _toEpilog();
                return this;
            }

            protected internal override Module _to(torch.Device device, torch.ScalarType dtype, bool non_blocking)
            {
                base._to(device, dtype, non_blocking);
                _toEpilog();
                return this;
            }

            protected internal override Module _to(torch.ScalarType dtype, bool non_blocking)
            {
                base._to(dtype, non_blocking);
                _toEpilog();
                return this;
            }

            void _toEpilog()
            {
                for (int i = 0; i < _list.Count; i++) {
                    string name = _list[i].Item1;
                    var param = base.get_parameter(name);
                    _list[i] = (name, param);
                    _dict[name] = param;
                }
            }

            /// <summary>
            /// Return the ParameterDict values.
            /// </summary>
            /// <returns></returns>
            public IEnumerable<Parameter> values() => _dict.Values;

            public (string, Parameter) this[int index] {
                get => _list[index];
                set {
                    var name = value.Item1;
                    _list[index] = value;
                    _dict[name] = value.Item2;
                }
            }

            public bool IsReadOnly => false;

            public ICollection<string> Keys => _list.Select(kv => kv.Item1).ToList();

            public ICollection<Parameter> Values => _list.Select(kv => kv.Item2).ToList();

            public int Count => _dict.Count;

            public Parameter this[string key] {
                get => _dict[key];
                set {
                    _dict[key] = value;
                    var idx = _list.FindIndex(kv => kv.Item1.Equals(key));
                    _list[idx] = (key, value);
                }
            }

            public override bool has_parameter(string target)
            {
                return _dict.ContainsKey(target);
            }

            public override Parameter get_parameter(string target)
            {
                if (_dict.TryGetValue(target, out var result)) {
                    return result;
                }
                return null;
            }

            public void Add((string, Parameter) item)
            {
                _dict.Add(item.Item1, item.Item2);
                _list.Add(item);
            }

            public void Add(string key, Parameter value)
            {
                _dict.Add(key, value);
                _list.Add((key, value));
            }

            public void Add(KeyValuePair<string, Parameter> item)
            {
                _dict.Add(item.Key, item.Value);
                _list.Add((item.Key, item.Value));
            }

            public bool Contains((string, Parameter) item)
            {
                return _list.Contains(item);
            }

            public void CopyTo((string, Parameter)[] array, int arrayIndex)
            {
                _list.CopyTo(array, arrayIndex);
            }

            public int IndexOf((string, Parameter) item)
            {
                return _list.IndexOf(item);
            }

            public void Insert(int index, (string, Parameter) item)
            {
                _dict.Add(item.Item1, item.Item2);
                _list.Insert(index, item);
            }

            public bool Remove((string, Parameter) item)
            {
                _dict.Remove(item.Item1);
                return _list.Remove(item);
            }

            public void RemoveAt(int index)
            {
                if (index >= _list.Count) throw new IndexOutOfRangeException();
                var (n,p) = _list[index];
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
                var idx = _list.FindIndex(kv => kv.Item1.Equals(key));
                _list.RemoveAt(idx);
                return _dict.Remove(key);
            }

            public bool TryGetValue(string key, [MaybeNullWhen(false)] out Parameter value)
            {
                return _dict.TryGetValue(key, out value);
            }

            public void Clear()
            {
                _dict.Clear();
                _list.Clear();
            }

            public bool Contains(KeyValuePair<string, Parameter> item)
            {
                return _dict.ContainsKey(item.Key);
            }

            public void CopyTo(KeyValuePair<string, Parameter>[] array, int arrayIndex)
            {
                throw new NotImplementedException();
            }

            public bool Remove(KeyValuePair<string, Parameter> item)
            {
                return _dict.Remove(item.Key);
            }

            public IEnumerator<(string, Parameter)> GetEnumerator()
            {
                return _list.GetEnumerator();
            }

            IEnumerator<KeyValuePair<string, Parameter>> IEnumerable<KeyValuePair<string, Parameter>>.GetEnumerator()
            {
                throw new NotImplementedException();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return ((IEnumerable<KeyValuePair<string, Parameter>>)this).GetEnumerator();
            }

            private List<(string, Parameter)> _list = new List<(string, Parameter)>();
            private Dictionary<string, Parameter> _dict = new Dictionary<string, Parameter>();
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {

            /// <summary>
            /// Create a ParameterList instance from an array of parameter tensors.
            /// </summary>
            /// <param name="parameters">A tuple of (name,parameter).</param>
            /// <remarks>
            /// ParameterDict can be indexed like a regular dictionary, but the parameters it
            /// contains are properly registered, and will be visible by all Module methods.
            ///
            /// ParameterDict is an ordered dictionary that respects the order of insertion, and
            /// in update(), the order of the merged OrderedDict or another ParameterDict (the argument to update()).
            /// </remarks>
            public static ParameterDict ParameterDict(params (string, Parameter)[] parameters)
            {
                var result = new ParameterDict();
                foreach (var (n, p) in parameters) {
                    result.Add(n, p);
                }
                return result;
            }
        }
    }
}
