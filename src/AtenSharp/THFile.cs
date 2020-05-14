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
            private static extern void THFile_free (IntPtr handle);

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
        private static extern int THFile_isOpened(HType self);

        /// <summary>
        ///   The open status of the file.
        /// </summary>
        public bool IsOpen { get { return 0 != THFile_isOpened(this.handle); } }

[       DllImport("caffe2")]
        private static extern int THFile_isBinary(HType self);

        [DllImport("caffe2")]
        private static extern void THFile_binary(HType self);

        /// <summary>
        ///   Set/get the binary file mode.
        /// </summary>
        public bool IsBinary { 
            get { return 0 != THFile_isBinary(this.handle); } 
            set { if (value) { THFile_binary(this.handle); } else { THFile_ascii(this.handle); }} 
        }

        [DllImport("caffe2")]
        private static extern void THFile_ascii(HType self);

        /// <summary>
        ///   Set/get the text file mode.
        /// </summary>
        public bool IsAscii { 
            get { return 0 == THFile_isBinary(this.handle); } 
            set { if (value) { THFile_ascii(this.handle); } else { THFile_binary(this.handle); }} 
        }
        
        [DllImport("caffe2")]
        private static extern int THFile_isReadable(HType self);
        
        /// <summary>
        ///   The readability status of the file.
        /// </summary>
        public bool CanRead { get { return 0 != THFile_isReadable(this.handle); } }

        [DllImport("caffe2")]
        private static extern int THFile_isWritable(HType self);

        /// <summary>
        ///   The writability status of the file.
        /// </summary>
        public bool CanWrite { get { return 0 != THFile_isWritable(this.handle); } }

        [DllImport("caffe2")]
        private static extern void THFile_seek(HType self, long position);

        /// <summary>
        ///   Seek to the given position, counted from the beginning of the file.
        /// </summary>
        /// <param name="position"></param>
        public void Seek(long position) { THFile_seek(this.handle, position); }

        [DllImport("caffe2")]
        private static extern void THFile_seekEnd(HType self);

        /// <summary>
        ///   Seek to the last position of the file. 
        /// </summary>
        public void SeekEnd() { THFile_seekEnd(this.handle); }

        [DllImport("caffe2")]
        private static extern long THFile_position(HType self);

        /// <summary>
        ///   The current file position.
        /// </summary>
        public long Position { get { return THFile_position(this.handle); } }

        [DllImport("caffe2")]
        private static extern void THFile_synchronize(HType self);       

        /// <summary>
        ///   Flush everything to disk.
        /// </summary>
        public void Flush() { THFile_synchronize(this.handle); }

        [DllImport("caffe2")]
        private static extern void THFile_close(HType self);

        /// <summary>
        ///   Close the file.
        /// </summary>
        public void Close() { THFile_close(this.handle); }

        [DllImport("caffe2")]
        private static extern int THFile_hasError(HType self);

        /// <summary>
        ///   Check whether the file object has outstanding errors.
        /// </summary>
        public bool HasError { get { return THFile_hasError(this.handle) != 0; }}

        [DllImport("caffe2")]
        private static extern void THFile_clearError(HType self);

        /// <summary>
        ///   Check whether the file object has outstanding errors.
        /// </summary>
        public void ClearError() { THFile_clearError(this.handle); }

        [DllImport("caffe2")]
        private static extern int THFile_isQuiet(HType self);

        [DllImport("caffe2")]
        private static extern void THFile_quiet(HType self);

        [DllImport("caffe2")]
        private static extern void THFile_pedantic(HType self);
        
        /// <summary>
        ///   If true, the file is silent about errors.
        /// </summary>
        public bool IsQuiet { 
            get { return THFile_isQuiet(this.handle) != 0; } 
            set { if (value) THFile_quiet(this.handle); else THFile_pedantic(this.handle); } 
        }

        [DllImport("caffe2")]
        private static extern int THFile_isAutoSpacing(HType self);

        [DllImport("caffe2")]
        private static extern void THFile_autoSpacing(HType self);

        [DllImport("caffe2")]
        private static extern void THFile_noAutoSpacing(HType self);

        /// <summary>
        ///   If true, the file will insert spaces and newlines in text files.
        /// </summary>
        public bool IsAutoSpacing
        {
            get { return THFile_isAutoSpacing(this.handle) != 0; }
            set { if (value) THFile_autoSpacing(this.handle); else THFile_noAutoSpacing(this.handle); }
        }
    }
}