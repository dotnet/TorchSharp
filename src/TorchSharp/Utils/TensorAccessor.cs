using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;

namespace TorchSharp.Utils
{
    /// <summary>
    /// TensorAccessor is used to present the contents of a tensor or tensor view to the .NET world as an ordered collection
    /// of values that integrates well with things like LINQ and foreach loops in the .NET world.
    /// </summary>
    /// <typeparam name="T">The type of the tensor elements.</typeparam>
    public class TensorAccessor<T> : IDisposable, IEnumerable<T> where T : unmanaged
    {
        public TensorAccessor(torch.Tensor tensor)
        {
            if (tensor.device_type != DeviceType.CPU) {
                throw new InvalidOperationException("Reading data from non-CPU memory is not supported. Move or copy the tensor to the cpu before reading.");
            }

            _shape = tensor.shape;
            _physicalStrides = tensor.stride();

            for (var i = 0; i < _physicalStrides.Length; i++) {
                if (_physicalStrides[i] < 0)
                    throw new NotImplementedException($"Negative tensor strides are not currently supported. tensor.strides({i}) == {_physicalStrides[i]}");
            }

            // Compute the logical strides.

            if (_shape.Length > 0) {
                _logigcalStrides = new long[_shape.Length];
                _logigcalStrides[_shape.Length - 1] = 1;
                for (int i = _shape.Length - 2; i >= 0; i--) {
                    _logigcalStrides[i] = _shape[i + 1] * _logigcalStrides[i + 1];
                    if (_logigcalStrides[i] != _physicalStrides[i]) _needToTranslate = true;
                }
            } else {
                // Special case -- a singleton tensor, i.e. scalar.
                _logigcalStrides = new long[0];
            }

            Count = (int)tensor.numel();

            // Get the data from native code.

            unsafe {
                var res = torch.Tensor.THSTensor_data(tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                // NOTE: there is no safety here.
                _tensor_data_ptr = res;
            }

            _tensor = tensor; // Keep the tensor alive now that everything is alright.
        }

        public long Count { get; private set; }

        public bool IsReadOnly => false;

        public T[] ToArray() => this.ToArray<T>();


        /// <summary>
        /// Translates a linear index within the span represented by the accessor to a linear index
        /// used by the underlying tensor. The two should only be different if the tensor is a view
        /// rather than an allocated tensor.
        /// </summary>
        /// <param name="idx">A linear index into the data view.</param>
        /// <returns></returns>
        private long TranslateIndex(long idx)
        {
            if (!_needToTranslate) return idx;

            // First, turn the linear index into a subscript list, based on the shape, i.e. logical strides.
            var subs = new List<long>();
            for (var i = 0; i < _logigcalStrides.Length; i++) {
                var s = idx / _logigcalStrides[i];
                idx -= s * _logigcalStrides[i];
                subs.Add(s);
            }

            // Then, use the actual (phyiscal) strides and the logical subscripts to determine the output index.

            long result = 0;
            for (var i = 0; i < _physicalStrides.Length; i++) {
                result += subs[i] * _physicalStrides[i];
            }

            return result;
        }

        /// <summary>
        /// Access elements of the underlying tensor / tensor view.
        /// </summary>
        /// <param name="index">A linear index into the data.</param>
        /// <returns></returns>
        public T this[long index] {
            get {
                if (index >= Count) throw new IndexOutOfRangeException();
                unsafe {
                    T* ptr = (T*)_tensor_data_ptr;
                    return ptr[TranslateIndex(index)];
                }
            }
            set {
                if (index >= Count) throw new IndexOutOfRangeException();
                unsafe {
                    T* ptr = (T*)_tensor_data_ptr;
                    ptr[TranslateIndex(index)] = value;
                }
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
            if (left._tensor_data_ptr == right._tensor_data_ptr) return true;
            if (left.Count != right.Count) return false;
            for (long i = 0; i < left.Count; i++) {
                if (!left[i].Equals(right[i])) return false;
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
            if (left._tensor_data_ptr == right._tensor_data_ptr) return false;
            if (left.Count != right.Count) return true;
            for (long i = 0; i < left.Count; i++) {
                if (left[i].Equals(right[i])) return false;
            }
            return true;
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
            return ToArray().GetHashCode();
        }

        public void CopyTo(T[] array, int arrayIndex = 0, long tensorIndex = 0)
        {
            for (int i = 0; i + arrayIndex < array.Length && i + tensorIndex < Count; i++) {
                array[i + arrayIndex] = this[i + tensorIndex];
            }
        }

        public void CopyFrom(T[] array, int arrayIndex = 0, long tensorIndex = 0)
        {
            for (int i = 0; i + arrayIndex < array.Length && i + tensorIndex < Count; i++) {
                this[i + tensorIndex] = array[i + arrayIndex];
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            return new TensorAccessorEnumerator(this);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Dispose()
        {
            // Clear the tensor we've been keeping alive.
            _tensor = null;
        }

        private torch.Tensor _tensor;   // Keeping it alive.
        private IntPtr _tensor_data_ptr;
        private long[] _shape;
        private long[] _physicalStrides;
        private long[] _logigcalStrides;
        private bool _needToTranslate = false;

        private class TensorAccessorEnumerator : IEnumerator<T>
        {
            public TensorAccessorEnumerator(TensorAccessor<T> span)
            {
                _currentIdx = -1;
                _span = span;
            }

            public T Current => _span[_currentIdx];

            object IEnumerator.Current => _span[_currentIdx];

            public void Dispose()
            {
                // Just clear the span field.
                _span = null;
            }

            public bool MoveNext()
            {
                _currentIdx += 1;
                return _currentIdx < _span.Count;
            }

            public void Reset()
            {
                _currentIdx = -1;
            }

            private long _currentIdx;
            private TensorAccessor<T> _span;
        }
    }
}
