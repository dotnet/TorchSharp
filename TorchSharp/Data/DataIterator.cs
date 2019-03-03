using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.Data
{
    /// <summary>
    /// Wrapper around methods used by the DataITerator class.
    /// This is mainly because C# does not allow extern methods on generic classes.
    /// </summary>
    internal static class ExternMethods
    {
        [DllImport("LibTorchSharp")]
        extern internal static IntPtr Data_Current(IntPtr iterator, IntPtr data, IntPtr target);

        [DllImport("LibTorchSharp")]
        extern internal static bool Data_MoveNext(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        extern internal static long Data_Size(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        extern internal static void Data_Reset(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        extern internal static void Data_Dispose(IntPtr iterator);
    }

    /// <summary>
    /// Class implementing enumerable over PyTorch's iterator.
    /// </summary>
    /// <typeparam name="TData"></typeparam>
    /// <typeparam name="TTarget"></typeparam>
    public class DataIterator<TData, TTarget> :
        IDisposable,
        IEnumerable<(ITorchTensor<TData> data, ITorchTensor<TTarget> target)>
    {
        /// <summary>
        ///    Class wrapping PyTorch's iterator object reference.
        /// </summary>
        protected sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            protected override bool ReleaseHandle()
            {
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        protected HType handle;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="handle"></param>
        internal DataIterator(IntPtr handle)
        {
            this.handle = new HType(handle, true);
        }

        ~DataIterator()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                ExternMethods.Data_Dispose(handle.DangerousGetHandle());
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        /// <summary>
        /// Return the total size in Bytes of the input dataset.
        /// </summary>
        /// <returns></returns>
        public long Size()
        {
            return ExternMethods.Data_Size(handle.DangerousGetHandle());
        }

        /// <summary>
        /// Get the enumerator for this iterator.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<(ITorchTensor<TData> data, ITorchTensor<TTarget> target)> GetEnumerator()
        {
            var iter = new DataIteratorEnumerator(this);
            iter.Reset();
            return iter;
        }


        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private class DataIteratorEnumerator : IEnumerator<(ITorchTensor<TData> data, ITorchTensor<TTarget> target)>
        {
            private DataIterator<TData, TTarget> _iterator;

            private readonly PinnedArray<IntPtr> _darray = new PinnedArray<IntPtr>();
            private readonly PinnedArray<IntPtr> _tarray = new PinnedArray<IntPtr>();

            private readonly IntPtr _dRef;
            private readonly IntPtr _tRef;

            private bool _isFirst = true;

            public DataIteratorEnumerator(DataIterator<TData, TTarget> iterator)
            {
                _iterator = iterator;

                _dRef = _darray.CreateArray(new IntPtr[1]);
                _tRef = _tarray.CreateArray(new IntPtr[1]);
            }

            public (ITorchTensor<TData> data, ITorchTensor<TTarget> target) Current
            {
                get
                {
                    ExternMethods.Data_Current(_iterator.handle.DangerousGetHandle(), _dRef, _tRef);   
                    return (_darray.Array[0].ToTorchTensor<TData>(), _tarray.Array[0].ToTorchTensor<TTarget>());
                }
            }

            object IEnumerator.Current => Current;

            public bool MoveNext()
            {
                if (_isFirst)
                {
                    _isFirst = false;
                    return true;
                }

                return ExternMethods.Data_MoveNext(_iterator.handle.DangerousGetHandle());
            }

            public void Reset()
            {
                _isFirst = true;
                ExternMethods.Data_Reset(_iterator.handle.DangerousGetHandle());
            }

            public void Dispose()
            {
                _darray.Dispose();
                _tarray.Dispose();
            }
        }
    }
}
