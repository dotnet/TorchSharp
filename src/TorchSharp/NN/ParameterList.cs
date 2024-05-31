// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections;
using System.Collections.Generic;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    using System.Linq;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Holds parameters in a list.
        /// ParameterList can be indexed like a regular list, but parameters it contains are properly registered, and will be visible by all Module methods.
        /// </summary>
        public class ParameterList : Module, IList<Parameter>
        {
            public ParameterList(params Parameter[] parameters) : base(nameof(ParameterList))
            {
                if (parameters != null && parameters.Length > 0) {
                    _list.AddRange(parameters);
                }
            }

            protected override void RegisterComponents()
            {
                if (_registered) return;

                for (int i = 0; i < _list.Count; i++) {
                    register_parameter($"{i}", _list[i]);
                }
                _registered = true;
            }


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
                    _list[i] = base.get_parameter($"{i}");
                }
            }

            public override IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
            {
                return Enumerable.Range(0, _list.Count).Select(i => ($"{i}", _list[i]));
            }

            public override bool has_parameter(string target)
            {
                return int.TryParse(target, out int idx) && idx > -1 && idx < _list.Count && _list[idx] is not null;
            }

            public override Parameter get_parameter(string target)
            {
                if (int.TryParse(target, out int idx) && idx > -1 && idx < _list.Count) {
                    return _list[idx];
                }
                return null;
            }

            private bool _registered = false;

            public Parameter this[int index] {
                get => _list[index];
                set => _list[index] = value;
            }

            public int Count => _list.Count;

            public bool IsReadOnly => false;

            public void Add(Parameter item)
            {
                _list.Add(item);
            }

            // The following two funtions are here because PyTorch defines them.

            /// <summary>
            /// Appends parameters from an enumeration to the end of the list.
            /// </summary>
            /// <param name="parameters"></param>
            public void extend(IEnumerable<Parameter> parameters)
            {
                _list.AddRange(parameters);
            }

            /// <summary>
            /// Appends zero or more parameters to the end of the list.
            /// </summary>
            public void append(params Parameter[] parameters)
            {
                if (parameters != null && parameters.Length > 0) {
                    _list.AddRange(parameters);
                }
            }

            public void Clear()
            {
                _list.Clear();
            }

            public bool Contains(Parameter item)
            {
                return _list.Contains(item);
            }

            public void CopyTo(Parameter[] array, int arrayIndex)
            {
                _list.CopyTo(array, arrayIndex);
            }

            public IEnumerator<Parameter> GetEnumerator()
            {
                return _list.GetEnumerator();
            }

            public int IndexOf(Parameter item)
            {
                return _list.IndexOf(item);
            }

            public void Insert(int index, Parameter item)
            {
                _list.Insert(index, item);
            }

            public bool Remove(Parameter item)
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

            private List<Parameter> _list = new List<Parameter>();
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Create a ParameterList instance from an array of parameter tensors.
            /// </summary>
            /// <param name="parameters">A list of modules.</param>
            /// <remarks>
            /// ParameterList can be indexed like a regular list, but the parameters it contains are properly registered.
            /// </remarks>
            public static ParameterList ParameterList(params Parameter[] parameters) => new ParameterList(parameters);
        }
    }
}
