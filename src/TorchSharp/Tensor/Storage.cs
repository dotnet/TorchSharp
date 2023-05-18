// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// A torch.Storage is a contiguous, one-dimensional array of a single data type.
        /// Every tensor has a corresponding storage of the same data type.
        /// </summary>
        public abstract class Storage
        {
            protected Storage()
            {
            }

            protected Storage(Tensor tensor)
            {
                _tensor = tensor;
                data_ptr();
            }

            protected Storage(Tensor tensor, IntPtr data_ptr)
            {
                _tensor = tensor;
                _tensor_data_ptr = (data_ptr == IntPtr.Zero) ? this.data_ptr() : data_ptr;
            }

            internal static Storage<T> Create<T>(Tensor tensor) where T : unmanaged
            {
                var type = typeof(T);
                switch (type) {
                case Type _ when type == typeof(byte):
                    return new Storage<T>(tensor.@byte());
                case Type _ when type == typeof(bool):
                    return new Storage<T>(tensor.@bool());
                case Type _ when type == typeof(int):
                    return new Storage<T>(tensor.@int());
                case Type _ when type == typeof(long):
                    return new Storage<T>(tensor.@long());
                case Type _ when type == typeof(float):
                    return new Storage<T>(tensor.@float());
                case Type _ when type == typeof(double):
                    return new Storage<T>(tensor.@double());
                case Type _ when type == typeof((float,float)):
                    return new Storage<T>(tensor.to_type(ScalarType.ComplexFloat32));
                case Type _ when type == typeof(System.Numerics.Complex):
                    return new Storage<T>(tensor.to_type(ScalarType.ComplexFloat64));
                default:
                    throw new NotSupportedException();
                }
            }

            protected static Tensor CreateTypedTensor<T>(ScalarType dtype, IList<T> rawArray)
            {
                switch (dtype) {
                case ScalarType.Int8:
                    return torch.tensor(rawArray as IList<byte>);
                case ScalarType.Bool:
                    return torch.tensor(rawArray as IList<bool>);
                case ScalarType.Int32:
                    return torch.tensor(rawArray as IList<int>);
                case ScalarType.Int64:
                    return torch.tensor(rawArray as IList<long>);
                case ScalarType.Float32:
                    return torch.tensor(rawArray as IList<float>);
                case ScalarType.Float64:
                    return torch.tensor(rawArray as IList<double>);
                case ScalarType.ComplexFloat32:
                    return torch.tensor(rawArray as IList<(float, float)>);
                case ScalarType.ComplexFloat64:
                    return torch.tensor(rawArray as IList<System.Numerics.Complex>);
                default:
                    throw new NotSupportedException();
                }
            }

            protected torch.Tensor _tensor;   // Keeping it alive.
            protected IntPtr _tensor_data_ptr;

            /// <summary>
            /// Convert to bool storage.
            /// </summary>
            /// <returns></returns>
            public Storage<bool> @bool() => _tensor.to_type(ScalarType.Bool).storage<bool>();

            /// <summary>
            /// Convert to byte storage.
            /// </summary>
            public Storage<byte> @byte() => _tensor.to_type(ScalarType.Byte).storage<byte>();

            /// <summary>
            /// Convert to char storage.
            /// </summary>
            public Storage<char> @char() => _tensor.to_type(ScalarType.Int8).storage<char>();

            /// <summary>
            /// Convert to int storage.
            /// </summary>
            public Storage<int> @int() => _tensor.to_type(ScalarType.Int32).storage<int>();

            /// <summary>
            /// Convert to long storage.
            /// </summary>
            public Storage<long> @long() => _tensor.to_type(ScalarType.Int64).storage<long>();

            /// <summary>
            /// Convert to float storage.
            /// </summary>
            public Storage<float> @float() => _tensor.to_type(ScalarType.Float32).storage<float>();

            /// <summary>
            /// Convert to double storage.
            /// </summary>
            public Storage<double> @double() => _tensor.to_type(ScalarType.Float64).storage<double>();

            /// <summary>
            /// Convert to 32-bit complex storage.
            /// </summary>
            public Storage<(float,float)> complex_float() => _tensor.to_type(ScalarType.ComplexFloat32).storage<(float, float)>();

            /// <summary>
            /// Convert to 64-bit complex storage.
            /// </summary>
            public Storage<System.Numerics.Complex> complex_double() => _tensor.to_type(ScalarType.ComplexFloat64).storage<System.Numerics.Complex>();

            /// <summary>
            /// The size of each storage element.
            /// </summary>
            /// <returns></returns>
            public abstract int element_size();

            /// <summary>
            /// A pointer to the raw data in memory.
            /// </summary>
            /// <returns></returns>
            protected IntPtr data_ptr()
            {
                if (_tensor_data_ptr != IntPtr.Zero)
                    return _tensor_data_ptr;

                var res = THSStorage_data_ptr(_tensor.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                _tensor_data_ptr = res;
                return res;
            }

            /// <summary>
            /// The number of bytes allocated to the storage.
            /// </summary>
            /// <returns></returns>
            public ulong nbytes()
            {
                var res = THSStorage_nbytes(_tensor.Handle);
                CheckForErrors();
                return res;
            }
        }

        /// <summary>
        /// A torch.Storage is a contiguous, one-dimensional array of a single data type.
        /// Every tensor has a corresponding storage of the same data type.
        /// </summary>
        public sealed class Storage<T> : torch.Storage, IDisposable, IEnumerable<T> where T : unmanaged
        {
            internal Storage(torch.Tensor tensor, int count = -1) : base(tensor, IntPtr.Zero)
            {
                _count = count;
            }

            internal Storage(IntPtr data_ptr, ScalarType dtype, int count = -1) : base(null, data_ptr)
            {
                _count = count;
                _tensor = CreateTypedTensor<T>(dtype, this.ToArray());
            }

            internal Storage(T value) : base()
            {
                _scalarValue = value;
            }

            private T _scalarValue = default(T);

            /// <summary>
            /// Size of each storage element.
            /// </summary>
            /// <returns></returns>
            public override int element_size()
            {
                ValidateNotScalar();
                unsafe {
                    return sizeof(T);
                }
            }

            /// <summary>
            /// The device where the storage is located.
            /// </summary>
            public Device device {
                get {
                    ValidateNotScalar();
                    return _tensor.device;
                }
            }

            /// <summary>
            /// Move the storage to a CUDA device.
            /// </summary>
            public Storage<T> cuda(Device device = null)
            {
                ValidateNotScalar();
                return _tensor.cuda(device).storage<T>();
            }

            /// <summary>
            /// Move the storage to the CPU.
            /// </summary>
            public Storage<T> cpu()
            {
                ValidateNotScalar();
                return _tensor.cpu().storage<T>();
            }

            /// <summary>
            /// Convert the storage instance to a .NET array, copying its data.
            /// </summary>
            /// <returns></returns>
            public T[] ToArray()
            {
                ValidateNotScalar();
                var result = new T[Count];
                CopyTo(result, 0);
                return result;
            }

            /// <summary>
            /// Accesses a single element of a storage.
            /// </summary>
            /// <param name="index">The index of the element.</param>
            /// <returns></returns>
            public T this[int index] {
                get {
                    ValidateNotScalar();
                    CheckIndex(index);
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        return ptr[index];
                    }
                }
                set {
                    ValidateNotScalar();
                    CheckIndex(index);
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        ptr[index] = value;
                    }
                }
            }

            /// <summary>
            /// Accesses a slice of a storage instance.
            /// </summary>
            /// <param name="range">The range, expressed as a tuple [start,end)</param>
            public Storage<T> this[(int? start, int? end) range] {
                get {
                    ValidateNotScalar();
                    if (range.end < range.start)
                        throw new ArgumentException("end < start");
                    var start = CheckIndex(range.start.HasValue? range.start.Value : 0);
                    var end = CheckIndex(range.end.HasValue ? range.end.Value : Count);
                    var count = end - start;
                    unsafe {
                        var ptr = _tensor_data_ptr + sizeof(T) * start;
                        return new Storage<T>(ptr, _tensor.dtype, count);
                    }
                }
                set {
                    ValidateNotScalar();
                    if (value._tensor is not null || value._tensor_data_ptr != IntPtr.Zero)
                        throw new InvalidOperationException();
                    if (range.end < range.start)
                        throw new ArgumentException("end < start");
                    var v = value._scalarValue;
                    var start = CheckIndex(range.start.HasValue ? range.start.Value : 0);
                    var end = CheckIndex(range.end.HasValue ? range.end.Value : Count);
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        for (int i = start; i < end; i++)
                            ptr[i] = v;
                    }
                }
            }

#if !NETSTANDARD2_0_OR_GREATER
            /// <summary>
            /// Accesses a slice of a storage instance.
            /// </summary>
            /// <param name="range">The range.</param>
            public Storage<T> this[System.Range range] {
                get {
                    var start = CheckIndex(range.Start);
                    var end = CheckIndex(range.End);
                    return this[(start, end)];
                }
                set {
                    var start = CheckIndex(range.Start);
                    var end = CheckIndex(range.End);
                    this[(start, end)] = value;
                }
            }

            private int CheckIndex(System.Index index)
            {
                var i = index.IsFromEnd ? Count - index.Value : index.Value;
                if (i < 0 || i >= Count)
                    throw new IndexOutOfRangeException();
                return i;
            }
#endif

            private int CheckIndex(int index) {
                if (index < 0 || index >= Count)
                    throw new IndexOutOfRangeException();
                return index;
            }

            private void ValidateNotScalar()
            {
                if (_tensor is null &&_tensor_data_ptr == IntPtr.Zero)
                    throw new InvalidOperationException("Invalid use of a scalar Storage instance.");
            }

            /// <summary>
            /// The number of elements in the storage.
            /// </summary>
            public int Count {
                get {
                    ValidateNotScalar();
                    if (_count == -1) {
                        unsafe {
                            _count = (int)nbytes() / sizeof(T);
                        }
                    }
                    return _count;
                }
            }

            private int _count = -1;

            public static implicit operator Storage<T>(T value)
            {
                return new Storage<T>(value);
            }

            /// <summary>
            /// Copy data from an array into a storage instance.
            /// </summary>
            /// <param name="array"></param>
            public void copy_(IList<T> array)
            {
                ValidateNotScalar();
                CopyFrom(array, 0);
            }

            /// <summary>
            /// Fill a storage instance with a single value.
            /// </summary>
            /// <param name="value"></param>
            public void fill_(T value)
            {
                ValidateNotScalar();
                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = 0; i < Count; i++)
                        ((T*)ptr)[i] = value;
                }
            }

            /// <summary>
            /// Convert the storage to a .NET list.
            /// </summary>
            /// <returns></returns>
            public IList<T> tolist()
            {
                ValidateNotScalar();
                var result = new T[Count];
                CopyTo(result, 0);
                return result;
            }

            public bool IsReadOnly => false;

            public bool Contains(T item)
            {
                ValidateNotScalar();
                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = 0; i < Count; i++)
                        if (((T*)ptr)[i].Equals(item)) return true;
                }
                return false;
            }

            /// <summary>
            /// Copy the contents of the Storage instance into the provided array, starting at 'arrayIndex'
            /// </summary>
            /// <param name="array">A target array.</param>
            /// <param name="arrayIndex">The first offset in the array to write data to.</param>
            public void CopyTo(IList<T> array, int arrayIndex)
            {
                ValidateNotScalar();

                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = 0; i < Count && i < array.Count-arrayIndex; i++)
                        array[i + arrayIndex] = ((T*)ptr)[i];
                }
            }

            /// <summary>
            /// Copy the contents of the provided array into the Storage instance, starting at 'arrayIndex' in the array.
            /// </summary>
            /// <param name="array">A source array.</param>
            /// <param name="arrayIndex">The first offset in the array to read data from.</param>
            public void CopyFrom(IList<T> array, int arrayIndex)
            {
                ValidateNotScalar();

                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = 0; i < Count && i < array.Count-arrayIndex; i++)
                        ((T*)ptr)[i] = array[i+arrayIndex];
                }
            }

            /// <summary>
            /// Look up a value and return the first index where it can be found.
            /// </summary>
            /// <param name="item">The item to look for.</param>
            /// <returns></returns>
            public int IndexOf(T item)
            {
                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = 0; i < Count; i++)
                        if (((T*)ptr)[i].Equals(item)) return i;
                }
                return -1;
            }

            public IEnumerator<T> GetEnumerator()
            {
                ValidateNotScalar();
                return new StorageEnumerator(this);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                ValidateNotScalar();
                return new StorageEnumerator(this);
            }

            public void Dispose()
            {
                _tensor_data_ptr = IntPtr.Zero;
                _tensor = null;
            }

            private struct StorageEnumerator: IEnumerator<T>
            {
                private Storage<T> _span;
                private readonly long _count;

                // State.
                private long _index;
                private T _current;

                public StorageEnumerator(Storage<T> span)
                {
                    _span = span;
                    _count = span.Count;
                    Debug.Assert(_count > 0);
                    _index = -1;
                    _current = default;
                }

                public T Current => _current;

                object IEnumerator.Current => Current;

                public void Dispose()
                {
                    _span = null;
                    Reset();
                }

                public bool MoveNext()
                {
                    if (_index >= _count - 1) {
                        Reset();
                        return false;
                    }

                    _index += 1;

                    unsafe { _current = ((T*)_span._tensor_data_ptr)[_index]; }
                    return true;
                }

                public void Reset()
                {
                    _index = -1;
                    _current = default;
                }
            }
        }
    }
}
