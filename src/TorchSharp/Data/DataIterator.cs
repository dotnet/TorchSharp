// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp.Data
{
    /// <summary>
    /// Class implementing enumerable over PyTorch's iterator.
    /// </summary>
    public class DataIterator :
        IDisposable,
        IEnumerable<(Tensor data, Tensor target)>
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
        /// Releases the storage.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Implements the .NET Dispose pattern.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                THSData_dispose(handle.DangerousGetHandle());
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
            return THSData_size(handle.DangerousGetHandle());
        }

        /// <summary>
        /// Get the enumerator for this iterator.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<(Tensor data, Tensor target)> GetEnumerator()
        {
            var iter = new DataIteratorEnumerator(this);
            iter.Reset();
            return iter;
        }


        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private class DataIteratorEnumerator : IEnumerator<(Tensor data, Tensor target)>
        {
            private DataIterator _iterator;

            private readonly PinnedArray<IntPtr> _darray = new PinnedArray<IntPtr>();
            private readonly PinnedArray<IntPtr> _tarray = new PinnedArray<IntPtr>();

            private readonly IntPtr _dRef;
            private readonly IntPtr _tRef;

            private bool _isFirst = true;

            public DataIteratorEnumerator(DataIterator iterator)
            {
                _iterator = iterator;

                _dRef = _darray.CreateArray(new IntPtr[1]);
                _tRef = _tarray.CreateArray(new IntPtr[1]);
            }

            public (Tensor data, Tensor target) Current
            {
                get
                {
                    THSData_current(_iterator.handle.DangerousGetHandle(), _dRef, _tRef);
                    return (new Tensor(_darray.Array[0]), new Tensor(_tarray.Array[0]));
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

                return THSData_moveNext(_iterator.handle.DangerousGetHandle());
            }

            public void Reset()
            {
                _isFirst = true;
                THSData_reset(_iterator.handle.DangerousGetHandle());
            }

            public void Dispose()
            {
                _darray.Dispose();
                _tarray.Dispose();
            }
        }
    }
}
