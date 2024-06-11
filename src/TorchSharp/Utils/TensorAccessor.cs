using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp.Utils
{
    /// <summary>
    /// TensorAccessor is used to present the contents of a tensor or tensor view to the .NET world as an ordered collection
    /// of values that integrates well with things like LINQ and foreach loops in the .NET world.
    /// </summary>
    /// <typeparam name="T">The type of the tensor elements.</typeparam>
    public sealed class TensorAccessor<T> : IDisposable, IEnumerable<T> where T : unmanaged
    {
        internal TensorAccessor(torch.Tensor tensor)
        {
            if (tensor.device_type != DeviceType.CPU) {
                throw new InvalidOperationException("Reading data from non-CPU memory is not supported. Move or copy the tensor to the cpu before reading.");
            }

            var strides = tensor.stride();
            for (var i = 0; i < strides.Length; i++) {
                if (strides[i] < 0)
                    throw new NotImplementedException($"Negative tensor strides are not currently supported. tensor.strides({i}) == {strides[i]}");
            }

            // Get the data from native code.

            unsafe {
                var res = THSTensor_data(tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                // NOTE: there is no safety here.
                _tensor_data_ptr = res;
            }

            _tensor = tensor; // Keep the tensor alive now that everything is alright.
        }

        public long Count => (_tensor is not null ? _tensor.numel() : 0);

        public bool IsReadOnly => false;

        public T[] ToArray()
        {
            if (_tensor.ndim < 2)
                return (T[])ToNDArray();

            var result = new T[Count];
            CopyTo(result);
            return result;
        }

        /// <summary>
        /// Extract tensor data as a multi-dimensional .NET array, with the same number of dimensions as the tensor.
        /// </summary>
        /// <returns>An array object, which should be cast to the concrete array type.</returns>
        public Array ToNDArray()
        {
            var shape = _tensor.shape;
            var strides = _tensor.stride();
            switch (_tensor.ndim) {
            default:
                return ToNDArray(shape, strides);
            case 0:
                unsafe {
                    var result = new T[1];
                    T* ptr = (T*)_tensor_data_ptr;
                    result[0] = ptr[0];
                    return result;
                }
            case 1:
                unsafe {
                    var result = new T[shape[0]];
                    T* ptr = (T*)_tensor_data_ptr;
                    for (long i0 = 0, off0 = 0; i0 < shape[0]; i0++, off0 += strides[0]) {
                        result[i0] = ptr[off0];
                    }
                    return result;
                }
            case 2:
                unsafe {
                    var result = new T[shape[0], shape[1]];
                    T* ptr = (T*)_tensor_data_ptr;
                    for (long i0 = 0, off0 = 0; i0 < shape[0]; i0++, off0 += strides[0]) {
                        for (long i1 = 0, off1 = off0; i1 < shape[1]; i1++, off1 += strides[1]) {
                            result[i0, i1] = ptr[off1];
                        }
                    }
                    return result;
                }
            case 3:
                unsafe {
                    var result = new T[shape[0], shape[1], shape[2]];
                    T* ptr = (T*)_tensor_data_ptr;
                    for (long i0 = 0, off0 = 0; i0 < shape[0]; i0++, off0 += strides[0]) {
                        for (long i1 = 0, off1 = off0; i1 < shape[1]; i1++, off1 += strides[1]) {
                            for (long i2 = 0, off2 = off1; i2 < shape[2]; i2++, off2 += strides[2]) {
                                result[i0, i1, i2] = ptr[off2];
                            }
                        }
                    }
                    return result;
                }
            case 4:
                unsafe {
                    var result = new T[shape[0], shape[1], shape[2], shape[3]];
                    T* ptr = (T*)_tensor_data_ptr;
                    for (long i0 = 0, off0 = 0; i0 < shape[0]; i0++, off0 += strides[0]) {
                        for (long i1 = 0, off1 = off0; i1 < shape[1]; i1++, off1 += strides[1]) {
                            for (long i2 = 0, off2 = off1; i2 < shape[2]; i2++, off2 += strides[2]) {
                                for (long i3 = 0, off3 = off2; i3 < shape[3]; i3++, off3 += strides[3]) {
                                    result[i0, i1, i2, i3] = ptr[off3];
                                }
                            }
                        }
                    }
                    return result;
                }
            case 5:
                unsafe {
                    var result = new T[shape[0], shape[1], shape[2], shape[3], shape[4]];
                    T* ptr = (T*)_tensor_data_ptr;
                    for (long i0 = 0, off0 = 0; i0 < shape[0]; i0++, off0 += strides[0]) {
                        for (long i1 = 0, off1 = off0; i1 < shape[1]; i1++, off1 += strides[1]) {
                            for (long i2 = 0, off2 = off1; i2 < shape[2]; i2++, off2 += strides[2]) {
                                for (long i3 = 0, off3 = off2; i3 < shape[3]; i3++, off3 += strides[3]) {
                                    for (long i4 = 0, off4 = off3; i4 < shape[4]; i4++, off4 += strides[4]) {
                                        result[i0, i1, i2, i3, i4] = ptr[off4];
                                    }
                                }
                            }
                        }
                    }
                    return result;
                }
            case 6:
                unsafe {
                    var result = new T[shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]];
                    T* ptr = (T*)_tensor_data_ptr;
                    for (long i0 = 0, off0 = 0; i0 < shape[0]; i0++, off0 += strides[0]) {
                        for (long i1 = 0, off1 = off0; i1 < shape[1]; i1++, off1 += strides[1]) {
                            for (long i2 = 0, off2 = off1; i2 < shape[2]; i2++, off2 += strides[2]) {
                                for (long i3 = 0, off3 = off2; i3 < shape[3]; i3++, off3 += strides[3]) {
                                    for (long i4 = 0, off4 = off3; i4 < shape[4]; i4++, off4 += strides[4]) {
                                        for (long i5 = 0, off5 = off4; i5 < shape[5]; i5++, off5 += strides[5]) {
                                            result[i0, i1, i2, i3, i4, i5] = ptr[off5];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return result;
                }
            }
        }

        private Array ToNDArray(long[] shape, long[] strides)
        {
            Array array = Array.CreateInstance(typeof(T), shape);
            long[] indexes = new long[_tensor.ndim];
            long[] off = new long[_tensor.ndim];

            while (true) {
                unsafe {
                    T* ptr = (T*)_tensor_data_ptr;
                    array.SetValue(ptr[off[array.Rank - 1]], indexes);
                }

                for (int i = array.Rank - 1; i >= 0; i--) {
                    if (indexes[i] < shape[i] - 1) {
                        indexes[i]++;
                        off[i] += strides[i];
                        for (int j = i; j < array.Rank - 1; j++)
                            off[j + 1] = off[j];
                        break;
                    } else {
                        if (i == 0) {
                            return array;
                        }
                        indexes[i] = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Access elements of the underlying tensor / tensor view.
        /// </summary>
        /// <param name="indices">A linear index into the data.</param>
        /// <returns></returns>
        public T this[params long[] indices] {
            get {
                long index = 0;
                if (indices.Length == 1) {
                    index = indices[0];
                    validate(index);
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        return ptr[TranslateIndex(index, _tensor)];
                    }
                } else {
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        return ptr[TranslateIndex(indices, _tensor)];
                    }
                }
            }
            set {
                long index = 0;
                if (indices.Length == 1) {
                    index = indices[0];
                    validate(index);
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        ptr[TranslateIndex(indices, _tensor)] = value;
                    }
                } else {
                    unsafe {
                        T* ptr = (T*)_tensor_data_ptr;
                        ptr[TranslateIndex(indices, _tensor)] = value;
                    }
                }
            }
        }

        private void validate(long index)
        {
            if (index >= Count) throw new IndexOutOfRangeException();
        }

        public void CopyTo(T[] array, int arrayIndex = 0, long tensorIndex = 0)
        {
            int idx = arrayIndex;
            foreach (int offset in GetSubsequentIndices(tensorIndex)) {
                if (idx >= array.Length) break;
                unsafe { array[idx] = ((T*)_tensor_data_ptr)[offset]; }
                idx += 1;
            }
        }

        public void CopyTo(Span<T> array, int arrayIndex = 0, long tensorIndex = 0)
        {
            int idx = arrayIndex;
            foreach (int offset in GetSubsequentIndices(tensorIndex)) {
                if (idx >= array.Length) break;
                unsafe { array[idx] = ((T*)_tensor_data_ptr)[offset]; }
                idx += 1;
            }
        }

        public void CopyFrom(T[] array, int arrayIndex = 0, long tensorIndex = 0)
        {
            int idx = arrayIndex;
            foreach (int offset in GetSubsequentIndices(tensorIndex)) {
                if (idx >= array.Length) break;
                unsafe { ((T*)_tensor_data_ptr)[offset] = array[idx]; }
                idx += 1;
            }
        }

        public void CopyFrom(ReadOnlySpan<T> array, int arrayIndex = 0, long tensorIndex = 0)
        {
            int idx = arrayIndex;
            foreach (int offset in GetSubsequentIndices(tensorIndex)) {
                if (idx >= array.Length) break;
                unsafe { ((T*)_tensor_data_ptr)[offset] = array[idx]; }
                idx += 1;
            }
        }

        /// <summary>
        /// Translates a linear index within the span represented by the accessor to a linear index
        /// used by the underlying tensor. The two should only be different if the tensor is a view
        /// rather than an allocated tensor.
        /// </summary>
        private static long TranslateIndex(long idx, torch.Tensor tensor)
        {
            if (idx >= tensor.numel() || idx < 0)
                throw new ArgumentOutOfRangeException($"{idx} in a collection of  ${tensor.numel()} elements.");

            if (tensor.is_contiguous() || idx == 0) return idx;

            long result = 0;
            var shape = tensor.shape;
            var strides = tensor.stride();

            for (var i = shape.Length - 1; i >= 0; i--) {
                idx = Math.DivRem(idx, shape[i], out long s);
                result += s * strides[i];
            }

            return result;
        }

        private static long TranslateIndex(long[] idx, torch.Tensor tensor)
        {
            long result = 0;
            var shape = tensor.shape;
            var strides = tensor.stride();

            for (var i = shape.Length - 1; i >= 0; i--) {
                if (idx[i] >= shape[i] || idx[i] < 0)
                    throw new IndexOutOfRangeException($"{idx[i]} >= {shape[i]} in dimension {i}.");
                result += idx[i] * strides[i];
            }

            return result;
        }

        internal static T ReadItemAt(torch.Tensor tensor, long index)
        {
            if (tensor.device_type != DeviceType.CPU) {
                throw new InvalidOperationException("Reading data from non-CPU memory is not supported. Move or copy the tensor to the cpu before reading.");
            }

            tensor.ValidateType(typeof(T));

            var strides = tensor.stride();
            for (var i = 0; i < strides.Length; i++) {
                if (strides[i] < 0)
                    throw new NotImplementedException($"Negative tensor strides are not currently supported. tensor.strides({i}) == {strides[i]}");
            }

            unsafe {
                var res = THSTensor_data(tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                // NOTE: there is no safety here.
                T* ptr = (T*)res;
                return ptr[TranslateIndex(index, tensor)];
            }
        }

        /// <summary>
        /// Compare two tensors element-wise.
        /// </summary>
        /// <param name="left">A tensor</param>
        /// <param name="right">Another tensor</param>
        /// <returns></returns>
        public static bool operator ==(TensorAccessor<T> left, TensorAccessor<T> right)
        {
            if (left.Count != right.Count) return false;

            var lEnum = left.GetEnumerator();
            var rEnum = right.GetEnumerator();

            while (lEnum.MoveNext() && rEnum.MoveNext()) {
                if (!lEnum.Current.Equals(rEnum.Current))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Compare two tensors element-wise.
        /// </summary>
        /// <param name="left">A tensor</param>
        /// <param name="right">Another tensor</param>
        /// <returns></returns>
        public static bool operator !=(TensorAccessor<T> left, TensorAccessor<T> right)
        {
            return !(left == right);
        }


        private IEnumerable<long> GetSubsequentIndices(long startingIndex)
        {
            if (startingIndex < 0 || startingIndex >= Count)
                throw new ArgumentOutOfRangeException(nameof(startingIndex));

            if (Count <= 1) {
                if (Count == 0) {
                    return Enumerable.Empty<long>();
                }

                return (new long[] { 0 }).AsEnumerable<long>();
            }

            if (_tensor.is_contiguous()) {
                return ContiguousIndices(startingIndex);
            }

            var stride = _tensor.stride();
            Debug.Assert(stride.Length > 0);

            if (stride.Length == 1) {
                return SimpleIndices(startingIndex, stride[0]);
            }

            return MultiDimensionIndices(startingIndex);
        }

        private IEnumerable<long> MultiDimensionIndices(long startingIndex)
        {
            long[] shape = _tensor.shape;
            long[] stride = _tensor.stride();
            long[] inds = new long[stride.Length];

            long index = startingIndex;
            long offset = TranslateIndex(startingIndex, _tensor);

            while (true) {

                index += 1;

                yield return offset;

                if (index >= Count) break;

                for (int i = inds.Length - 1; ; i--) {
                    Debug.Assert(i >= 0);
                    offset += stride[i];
                    if (++inds[i] < shape[i])
                        break;

                    // Overflow of current dimension so rewind accordingly.
                    // Can't overflow the final (left-most) dimension.
                    Debug.Assert(i > 0);
                    // Note: for perf, this multiplication could be done once up front and cached in an array.
                    offset -= inds[i] * stride[i];
                    inds[i] = 0;
                }
            }
        }

        private IEnumerable<long> SimpleIndices(long startingIndex, long stride)
        {
            long index = startingIndex;
            long offset = TranslateIndex(startingIndex, _tensor);

            while (index < Count) {
                yield return offset;
                offset += stride;
                index += 1;
            }
        }
        private IEnumerable<long> ContiguousIndices(long startingIndex)
        {
            // If there was an overload for Enumerable.Range that
            // produced long integers, we wouldn't need this implementation.

            long index = startingIndex;
            while (index < Count) {
                yield return index;
                index += 1;
            }
        }


        /// <summary>
        /// Compare two tensors element-wise.
        /// </summary>
        /// <param name="obj">Another tensor</param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            var left = this;
            var right = obj as TensorAccessor<T>;
            if (right == null) return false;

            if (left._tensor_data_ptr == right._tensor_data_ptr) return true;
            if (left.Count != right.Count) return false;
            for (long i = 0; i < left.Count; i++) {
                if (!left[i].Equals(right[i])) return false;
            }
            return true;
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            _tensor_data_ptr = IntPtr.Zero;
            // Clear the tensor that we've been keeping alive.
            _tensor = null;
        }

        private torch.Tensor _tensor;   // Keeping it alive.
        private IntPtr _tensor_data_ptr;

#if true
        public IEnumerator<T> GetEnumerator()
        {
            if (Count <= 1) {
                if (Count == 0)
                    return Enumerable.Empty<T>().GetEnumerator();
                return new T[1] { this[0] }.AsEnumerable<T>().GetEnumerator();
            }

            if (_tensor.is_contiguous()) {
                return new SimpleAtorImpl(this, 1);
            }

            var stride = _tensor.stride();
            Debug.Assert(stride.Length > 0);

            if (stride.Length == 1) {
                return new SimpleAtorImpl(this, stride[0]);
            }

            return new GeneralAtorImpl(this, stride);
        }

        private class SimpleAtorImpl : IEnumerator<T>
        {
            private TensorAccessor<T> _span;
            private readonly long _count;
            private readonly long _stride;

            // State.
            private long _index;
            private long _offset;
            private T _current;

            public SimpleAtorImpl(TensorAccessor<T> span, long stride)
            {
                _span = span;
                _count = span.Count;
                Debug.Assert(_count > 0);
                _stride = stride;
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
                    _offset = 0;
                } else if (++_index >= _count) {
                    Reset();
                    return false;
                } else {
                    _offset += _stride;
                }

                unsafe { _current = ((T*)_span._tensor_data_ptr)[_offset]; }
                return true;
            }

            public void Reset()
            {
                _index = -1;
                _offset = -1;
                _current = default;
            }
        }

        private class GeneralAtorImpl : IEnumerator<T>
        {
            private TensorAccessor<T> _span;
            private readonly long _count;
            private readonly long[] _shape;
            private readonly long[] _stride;
            private readonly long[] _inds;

            // State.
            private long _index;
            private long _offset;

            public GeneralAtorImpl(TensorAccessor<T> span, long[] stride)
            {
                Debug.Assert(stride.Length > 1);
                _span = span;
                _count = span.Count;
                Debug.Assert(_count > 0);
                _shape = span._tensor.shape;
                Debug.Assert(_shape.Length == stride.Length);
                _stride = stride;
                _inds = new long[stride.Length];
                Reset();
            }

            public T Current { get; private set; }

            object IEnumerator.Current => Current;

            public void Dispose()
            {
                // Just clear the span field.
                _span = null;
            }

            public bool MoveNext()
            {
                if (_index < 0) {
                    _index = 0;
                    _offset = 0;
                    Array.Clear(_inds, 0, _inds.Length);
                } else if (++_index >= _count) {
                    Reset();
                    return false;
                } else {
                    for (int i = _inds.Length - 1; ; i--) {
                        Debug.Assert(i >= 0);
                        _offset += _stride[i];
                        if (++_inds[i] < _shape[i])
                            break;

                        // Overflow of current dimension so rewind accordingly.
                        // Can't overflow the final (left-most) dimension.
                        Debug.Assert(i > 0);
                        // Note: for perf, this multiplication could be done once up front and cached in an array.
                        _offset -= _inds[i] * _stride[i];
                        _inds[i] = 0;
                    }
                }

                unsafe { Current = ((T*)_span._tensor_data_ptr)[_offset]; }
                return true;
            }

            public void Reset()
            {
                _index = -1;
                _offset = -1;
                Current = default;
            }
        }
#else
        public IEnumerator<T> GetEnumerator()
        {
            return new TensorAccessorEnumerator(this);
        }
#endif
    }
}
