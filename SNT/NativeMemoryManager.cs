using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;
using System;

namespace Torch.SNT
{
    /// <summary>
    ///   Class used to wrap the native memory of a Torch Tensor.  
    /// </summary>
    internal sealed class NativeMemory<T> : MemoryManager<T>
    {
        private int refCount = 0;
        private IntPtr memory;
        private int length;

        public NativeMemory (IntPtr memory, int length)
        {
            if (length < 0)
            {
                throw new ArgumentOutOfRangeException (nameof (length), "Length cannot be negative.");
            }

            this.memory = memory;
            this.length = length;
        }

        /// <summary>
        /// Destructor for the native memory.
        /// The Tensor memory lifecycle is managed by Torch. Disposing the memory
        /// from here will through a "double free or corruption" error.
        /// <summary>
        ~NativeMemory ()
        {
            Dispose (false);
        }

        /// <summary>
        /// Whether the memory is disposed or not
        /// </summary>
        public bool IsDisposed { get; private set; } = false;

        /// <summary>
        /// Returns a span wrapping the underlying memory.
        /// Remember to Unpin the memory once the span is disposed.
        /// </summary>
        public override Span<T> GetSpan ()
        {
            unsafe
            {
                return new Span<T> (memory.ToPointer (), length);
            }
        }

        /// <summary>
        /// Returns a handle to the memory that has been pinned and hence its address can be taken.
        /// </summary>
        /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
        public override MemoryHandle Pin (int elementIndex = 0)
        {
            unsafe
            {
                Retain();
                if ((uint)elementIndex > length) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                void* pointer = Unsafe.Add<T> ((void*)memory, elementIndex);
                return new MemoryHandle (pointer, default, this);
            }
        }

        /// <summary>
        /// Lets the garbage collector know that the object is free to be moved now.
        /// </summary>
        public override void Unpin ()
        {
            Release();
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>        
        public void Dispose ()
        {
            Dispose (false);
            GC.SuppressFinalize (this);
        }

        protected override void Dispose (bool disposing)
        {
            if (IsDisposed)
            {
                return;
            }

            if (!Equals (refCount, 0))
            {
                throw new InvalidOperationException ("Disposing over unmatched Release/Retain");
            }

            IsDisposed = true;

            if (disposing)
            {
                // Typically this would call into a native method appropriate for the platform
                Marshal.FreeHGlobal (memory);
                memory = IntPtr.Zero;
                length = 0;
            }
        }

        private bool Release ()
        {
            int newRefCount = Interlocked.Decrement (ref refCount);

            if (newRefCount < 0)
            {
                throw new InvalidOperationException ("Unmatched Release/Retain");
            }

            return newRefCount != 0;
        }

        private void Retain ()
        {
            if (IsDisposed)
            {
                throw new ObjectDisposedException (nameof (NativeMemory<T>));
            }

            Interlocked.Increment (ref refCount);
        }
    }
}
