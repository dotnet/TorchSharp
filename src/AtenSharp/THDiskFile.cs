using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;

namespace Torch.IO {
    /// <summary>
    ///   Bindings for THDiskfile native API.
    /// </summary>
    public class DiskFile : File {

        [DllImport("caffe2")]
        private static extern File.HType THDiskFile_new(string name, string mode, int isQuiet);
        [DllImport("caffe2")]
        private static extern File.HType THPipeFile_new(string name, string mode, int isQuiet);

        /// <summary>
        ///   Create a disk file. 
        /// </summary>
        /// <param name="name">A file name.</param>
        /// <param name="mode">A standard POSIX file mode: "r", "w", "rw", etc.</param>
        /// <param name="isPipe">Is this a pipe or not?</param>
        /// <param name="isQuiet"></param>
        public DiskFile(string name, string mode, bool isPipe = false, bool isQuiet = false)
        {
            var binary = mode.IndexOf('b') != -1;
            if (binary)
            {
                mode = mode.Replace("b","");
            }
            this.handle = isPipe ?
                THPipeFile_new(name, mode, isQuiet ? 1 : 0) :
                THDiskFile_new(name, mode, isQuiet ? 1 : 0);
            this.IsBinary = binary;
        }

        [DllImport("caffe2")]
        private static extern string THDiskFile_name(File.HType self);

        /// <summary>
        ///   The file name.
        /// </summary>
        public string Name { get { return THDiskFile_name(this.handle); } }

        [DllImport("caffe2")]
        private static extern int THDiskFile_isLittleEndianCPU();

        /// <summary>
        ///   Is the native byte order little-endian?
        /// </summary>
        public static bool IsLittleEndianCPU {  get { return 0 != THDiskFile_isLittleEndianCPU(); } }

        [DllImport("caffe2")]
        private static extern int THDiskFile_isBigEndianCPU();

        /// <summary>
        ///   Is the native byte order big-endian?
        /// </summary>
        public static bool IsBigEndianCPU { get { return 0 != THDiskFile_isBigEndianCPU(); } }

        [DllImport("caffe2")]
        private static extern void THDiskFile_nativeEndianEncoding(File.HType self);

        /// <summary>
        ///   Use the native byte order for encoding multi-byte data when writing to the file.
        /// </summary>
        public void UseNativeEndianEncoding()
        {
            THDiskFile_nativeEndianEncoding(this.handle);
        }

        [DllImport("caffe2")]
        private static extern void THDiskFile_littleEndianEncoding(File.HType self);

        /// <summary>
        ///   Use the little-endian byte order for encoding multi-byte data when writing to the file.
        /// </summary>
        public void UseLittleEndianEncoding()
        {
            THDiskFile_littleEndianEncoding(this.handle);
        }

        [DllImport("caffe2")]
        private static extern void THDiskFile_bigEndianEncoding(File.HType self);

        /// <summary>
        ///   Use the big-endian byte order for encoding multi-byte data when writing to the file.
        /// </summary>
        public void UseBigEndianEncoding()
        {
            THDiskFile_bigEndianEncoding(this.handle);
        }

        [DllImport("caffe2")]
        private static extern void THDiskFile_longSize(File.HType self, int size);

        /// <summary>
        ///    Sets the size of long values.
        /// </summary>
        /// <remarks>Should be one of 0, 4, or 8.</remarks>
        public int LongSize { set { THDiskFile_longSize(this.handle, value); } }

        [DllImport("caffe2")]
        private static extern void THDiskFile_noBuffer(File.HType self);

        /// <summary>
        ///    Don't buffer disk reads and writes.
        /// </summary>
        public void NoBuffer() { THDiskFile_noBuffer(this.handle); }
    }
}
