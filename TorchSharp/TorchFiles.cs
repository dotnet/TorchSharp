using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;

namespace Torch.IO
{
    /// <summary>
    ///   Abstract base class for all Torch files.
    /// </summary>
    public abstract partial class File : IDisposable
    {
        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            public override bool IsInvalid => handle == (IntPtr)0;
            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            [DllImport ("caffe2")]  
            extern static void THFile_free (IntPtr handle);

            protected override bool ReleaseHandle()
            {
                THFile_free(handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;

        /// <summary>
        ///   Finalizer
        /// </summary>
        ~File()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the underlying storage.
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
            if (IsOpen)
            {
                Close();
            }

            if (disposing)
            {
                handle.Dispose();
                handle = null;
            }
        }

        [DllImport("caffe2")]
        extern static int THFile_isOpened(HType self);
        [DllImport("caffe2")]
        extern static int THFile_isReadable(HType self);
        [DllImport("caffe2")]
        extern static int THFile_isWritable(HType self);
        [DllImport("caffe2")]
        extern static void THFile_seek(HType self, long position);
        [DllImport("caffe2")]
        extern static void THFile_seekEnd(HType self);

        [DllImport("caffe2")]
        extern static long THFile_position(HType self);

        /// <summary>
        ///   The open status of the file.
        /// </summary>
        public bool IsOpen { get { return 0 != THFile_isOpened(this.handle); } }

        /// <summary>
        ///   The readability status of the file.
        /// </summary>
        public bool CanRead { get { return 0 != THFile_isReadable(this.handle); } }

        /// <summary>
        ///   The writability status of the file.
        /// </summary>
        public bool CanWrite { get { return 0 != THFile_isWritable(this.handle); } }

        /// <summary>
        ///   Seek to the given position, counted from the beginning of the file.
        /// </summary>
        /// <param name="position"></param>
        public void Seek(long position) { THFile_seek(this.handle, position); }

        /// <summary>
        ///   Seek to the last position of the file. 
        /// </summary>
        public void SeekEnd() { THFile_seekEnd(this.handle); }

        /// <summary>
        ///   The current file position.
        /// </summary>
        public long Position { get { return THFile_position(this.handle); } }


        [DllImport("caffe2")]
        extern static void THFile_close(HType self);

        /// <summary>
        ///   Close the file.
        /// </summary>
        public void Close() { THFile_close(this.handle); }

    }
}