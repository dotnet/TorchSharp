// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Holds parameters in a list.
            /// ParameterList can be indexed like a regular list, but parameters it contains are properly registered, and will be visible by all Module methods.
            /// </summary>
            public class ParameterList : Module, IList<parameter.Parameter>
            {
                public ParameterList(string name) : base(name)
                {
                }

                protected override void RegisterComponents()
                {
                    if (_registered) return;

                    for (int i = 0; i < _list.Count; i++) {
                        RegisterParameter($"{i}", _list[i]);
                    }
                    _registered = true;
                }

                private bool _registered = false;

                public parameter.Parameter this[int index] {
                    get => _list[index];
                    set => _list[index] = value;
                }

                public int Count => _list.Count;

                public bool IsReadOnly => false;

                public void Add(parameter.Parameter item)
                {
                    _list.Add(item);
                }

                /// <summary>
                /// Appends zero or more parameters to the end of the list.
                /// </summary>
                public void append(params parameter.Parameter[] parameters)
                {
                    if (parameters != null && parameters.Length > 0) {
                        _list.AddRange(parameters);
                    }
                }

                public void Clear()
                {
                    _list.Clear();
                }

                public bool Contains(parameter.Parameter item)
                {
                    return _list.Contains(item);
                }

                public void CopyTo(parameter.Parameter[] array, int arrayIndex)
                {
                    _list.CopyTo(array, arrayIndex);
                }

                /// <summary>
                /// Appends parameters from an enumeration to the end of the list.
                /// </summary>
                /// <param name="parameters"></param>
                public void extend(IEnumerable<parameter.Parameter> parameters)
                {
                    _list.AddRange(parameters);
                }

                public IEnumerator<parameter.Parameter> GetEnumerator()
                {
                    return _list.GetEnumerator();
                }

                public int IndexOf(parameter.Parameter item)
                {
                    return _list.IndexOf(item);
                }

                public void Insert(int index, parameter.Parameter item)
                {
                    _list.Insert(index, item);
                }

                public bool Remove(parameter.Parameter item)
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

                private List<parameter.Parameter> _list = new List<parameter.Parameter>();
            }
        }
    }
}
