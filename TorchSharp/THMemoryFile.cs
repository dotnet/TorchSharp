using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;

namespace Torch.IO
{
    /// <summary>
    /// Bindings for the native THMemoryFile APIs
    /// </summary>
    public class MemoryFile : File {

        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for memory files.
        /// </summary>
        public class CharStorage : IDisposable {
            internal sealed class HType : SafeHandle {
                public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
                {
                    SetHandle (preexistingHandle);
                }
                
                public override bool IsInvalid => handle == (IntPtr) 0;
                // This is just for marshalling
                internal HType () : base (IntPtr.Zero, true)
                {
                }
                
                [DllImport ("caffe2")]
                extern static void THCharStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THCharStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THCharStorage_new ();
            
            /// <summary>
            ///   Initializes an empty ByteStorage instance.
            /// </summary>
            public CharStorage ()
            {
                handle = THCharStorage_new ();
            }
            
            internal CharStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THCharStorage_newWithSize(IntPtr size);
            
            /// <summary>
            ///   Initializes a ByteStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public CharStorage (long size)
            {
                handle = THCharStorage_newWithSize((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~CharStorage ()
            {
                Dispose (false);
            }
            
            /// <summary>
            ///   Releases the storage.
            /// </summary>        
            public void Dispose ()
            {
                Dispose (true);
                GC.SuppressFinalize (this);
            }
            
            /// <summary>
            ///   Implements the .NET Dispose pattern.
            /// </summary>
            protected void Dispose (bool disposing)
            {
                if (disposing){
                    handle.Dispose ();
                    handle = null;
                }
            }
            
            [DllImport ("caffe2")]
            extern static byte THCharStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);

            [DllImport ("caffe2")]
            extern static void THCharStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  byte value);
            
            /// <summary>
            ///   Access an element of the storage at the given index.
            /// </summary>
            public byte this [long index] {
                get => THCharStorage_get (handle, (IntPtr) (index));
                set {
                    THCharStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static byte THCharStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THCharStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THCharStorage_fill (HType handle, byte value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (byte value)
            {
                THCharStorage_fill (handle, value);
            }
        }

        [DllImport ("caffe2")]
        extern static HType THMemoryFile_newWithStorage(CharStorage.HType handle, string mode);

        /// <summary>
        ///    Creates an empty memory file from an existing storage buffer.
        /// </summary>
        /// <param name="storage">A storage object.</param>
        /// <param name="mode">Standard POSIX file modes: "r", "w", "rw", etc.</param>
        public MemoryFile(CharStorage storage, string mode)
        {
            handle = THMemoryFile_newWithStorage(storage.handle, mode);
        }

        [DllImport ("caffe2")]
        extern static HType THMemoryFile_new(string mode);

        /// <summary>
        ///    Creates an empty memory file.
        /// </summary>
        /// <param name="mode">Standard POSIX file modes: "r", "w", "rw", etc.</param>
        public MemoryFile (string mode)
        {
            handle = THMemoryFile_new (mode);
        }

        /// <summary>
        ///    Creates a memory file from and existing memory file handle.
        /// </summary>
        /// <param name="handle">A memory file handle.</param>
        internal MemoryFile (HType handle)
        {
            this.handle = handle;
        }

        [DllImport ("caffe2")]
        extern static CharStorage.HType THMemoryFile_storage(HType self);

        /// <summary>
        ///    Gets the underlying storage handle.
        /// </summary>
        public CharStorage Storage { get { return new CharStorage(THMemoryFile_storage(this.handle)); } }

        [DllImport ("caffe2")]
        extern static void THMemoryFile_longSize(HType self, int size);

        /// <summary>
        ///    Sets the size of long values.
        /// </summary>
        /// <remarks>Should be one of 0, 4, or 8.</remarks>
        public int LongSize { set { THMemoryFile_longSize(this.handle, value); } }
    }
}