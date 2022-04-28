using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;

namespace TorchSharp
{
    public static partial class torch
    {
        public abstract class Storage
        {
            internal Storage()
            {
            }

            internal Storage(Tensor tensor)
            {
                _tensor = tensor;
                data_ptr();
            }

            internal Storage(Tensor tensor, IntPtr data_ptr)
            {
                _tensor = tensor;
                _tensor_data_ptr = (data_ptr == IntPtr.Zero) ? this.data_ptr() : data_ptr;
            }

            internal static Storage<T> CreateTypedStorageInstance<T>(Tensor tensor) where T : unmanaged
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
                    throw new NotImplementedException();
                }
            }

            internal static Tensor CreateTypedTensor<T>(ScalarType dtype, IList<T> rawArray)
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
                    throw new NotImplementedException();
                }
            }

            protected torch.Tensor _tensor;   // Keeping it alive.
            protected IntPtr _tensor_data_ptr;

            public Storage<bool> @bool() => _tensor.to_type(ScalarType.Bool).storage<bool>();

            public Storage<byte> @byte() => _tensor.to_type(ScalarType.Byte).storage<byte>();

            public Storage<char> @char() => _tensor.to_type(ScalarType.Int8).storage<char>();

            public Storage<int> @int() => _tensor.to_type(ScalarType.Int32).storage<int>();

            public Storage<long> @long() => _tensor.to_type(ScalarType.Int64).storage<long>();

            public Storage<float> @float() => _tensor.to_type(ScalarType.Float32).storage<float>();

            public Storage<double> @double() => _tensor.to_type(ScalarType.Float64).storage<double>();

            public Storage<(float,float)> complex_float() => _tensor.to_type(ScalarType.ComplexFloat32).storage<(float, float)>();

            public Storage<System.Numerics.Complex> complex_double() => _tensor.to_type(ScalarType.ComplexFloat64).storage<System.Numerics.Complex>();

            public abstract int element_size();

            protected IntPtr data_ptr()
            {
                if (_tensor_data_ptr != IntPtr.Zero)
                    return _tensor_data_ptr;

                var res = THSStorage_data_ptr(_tensor.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                _tensor_data_ptr = res;
                return res;
            }

            public Device device { get { return _tensor.device; } }

            public UInt64 nbytes()
            {
                var res = THSStorage_nbytes(_tensor.Handle);
                CheckForErrors();
                return res;
            }

            [DllImport("LibTorchSharp")]
            protected static extern UInt64 THSStorage_nbytes(IntPtr tensor);
            [DllImport("LibTorchSharp")]
            protected static extern void THSStorage_set_nbytes(IntPtr tensor, UInt64 nbytes);
            [DllImport("LibTorchSharp")]
            protected static extern IntPtr THSStorage_data_ptr(IntPtr tensor);
        }

        public class Storage<T> : torch.Storage, IDisposable, IEnumerable<T> where T : unmanaged
        {
            public Storage(torch.Tensor tensor, int count = -1) : base(tensor, IntPtr.Zero)
            {
                _count = count;
            }

            public Storage(IntPtr data_ptr, ScalarType dtype, int count = -1) : base(null, data_ptr)
            {
                _count = count;
                _tensor = CreateTypedTensor<T>(dtype, this.ToArray());
            }

            internal Storage(T value) : base()
            {
                _scalarValue = value;
            }

            private T _scalarValue = default(T);

            public override int element_size()
            {
                ValidateNotScalar();
                unsafe {
                    return sizeof(T);
                }
            }

            public Storage<T> cuda()
            {
                ValidateNotScalar();
                return _tensor.cuda().storage<T>();
            }

            public Storage<T> cpu()
            {
                ValidateNotScalar();
                return _tensor.cpu().storage<T>();
            }

            public T[] ToArray()
            {
                ValidateNotScalar();
                var result = new T[Count];
                CopyTo(result, 0);
                return result;
            }

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

            public Storage<T> this[(int? start, int? end) range] {
                get {
                    ValidateNotScalar();
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
            public Storage<T> this[System.Range index] {
                get {
                    ValidateNotScalar();
                    var start = CheckIndex(index.Start);
                    var end = CheckIndex(index.End);
                    return this[(start, end)];
                }
                set {
                    ValidateNotScalar();
                    var start = CheckIndex(index.Start);
                    var end = CheckIndex(index.End);
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
                    throw new InvalidOperationException();
            }

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

            public void copy_(IList<T> array)
            {
                ValidateNotScalar();
                CopyFrom(array, 0);
            }

            public void fill_(T value)
            {
                ValidateNotScalar();
                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = 0; i < Count; i++)
                        ((T*)ptr)[i] = value;
                }
            }

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

            public void CopyTo(IList<T> array, int arrayIndex)
            {
                ValidateNotScalar();
                int idx = arrayIndex;

                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = arrayIndex; i < Count && i < array.Count-arrayIndex; i++)
                        array[i + arrayIndex] = ((T*)ptr)[i];
                }
            }

            public void CopyFrom(IList<T> array, int arrayIndex)
            {
                ValidateNotScalar();
                int idx = arrayIndex;

                unsafe {
                    var ptr = (T*)data_ptr();
                    for (int i = arrayIndex; i < Count && i < array.Count-arrayIndex; i++)
                        ((T*)ptr)[i] = array[i+arrayIndex];
                }
            }

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

            private class StorageEnumerator: IEnumerator<T>
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
                    Reset();
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
                    if (_index < 0) {
                        _index = 0;
                    } else if (++_index >= _count) {
                        Reset();
                        return false;
                    } else {
                        _index += 1;
                    }

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
