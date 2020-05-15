// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
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
        extern internal static IntPtr THSData_current(IntPtr iterator, IntPtr data, IntPtr target);

        [DllImport("LibTorchSharp")]
        extern internal static bool THSData_moveNext(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        extern internal static long THSData_size(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        extern internal static void THSData_reset(IntPtr iterator);

        [DllImport("LibTorchSharp")]
        extern internal static void THSData_dispose(IntPtr iterator);
    }

    /// <summary>
    /// Class implementing enumerable over PyTorch's iterator.
    /// </summary>
    public class DataIterator :
        IDisposable,
        IEnumerable<(TorchTensor data, TorchTensor target)>
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
                ExternMethods.THSData_dispose(handle.DangerousGetHandle());
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
            return ExternMethods.THSData_size(handle.DangerousGetHandle());
        }

        /// <summary>
        /// Get the enumerator for this iterator.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<(TorchTensor data, TorchTensor target)> GetEnumerator()
        {
            var iter = new DataIteratorEnumerator(this);
            iter.Reset();
            return iter;
        }


        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private class DataIteratorEnumerator : IEnumerator<(TorchTensor data, TorchTensor target)>
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

            public (TorchTensor data, TorchTensor target) Current
            {
                get
                {
                    ExternMethods.THSData_current(_iterator.handle.DangerousGetHandle(), _dRef, _tRef);   
                    return (new TorchTensor(_darray.Array[0]), new TorchTensor(_tarray.Array[0]));
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

                return ExternMethods.THSData_moveNext(_iterator.handle.DangerousGetHandle());
            }

            public void Reset()
            {
                _isFirst = true;
                ExternMethods.THSData_reset(_iterator.handle.DangerousGetHandle());
            }

            public void Dispose()
            {
                _darray.Dispose();
                _tarray.Dispose();
            }
        }
    }
}
