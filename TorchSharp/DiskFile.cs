using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;

namespace Torch.IO {
    public class DiskFile  : File {

        [DllImport("caffe2")] extern static File.HType THDiskFile_new(string name, string mode, int isQuiet);
        [DllImport("caffe2")] extern static File.HType THPipeFile_new(string name, string mode, int isQuiet);

        /// <summary>
        ///   Create a disk file. 
        /// </summary>
        /// <param name="name">A file name.</param>
        /// <param name="mode">A standard POSIX file mode: "r", "w", "rw", etc.</param>
        /// <param name="isPipe">Is this a pipe or not?</param>
        /// <param name="isQuiet"></param>
        public DiskFile(string name, string mode, bool isPipe = false, bool isQuiet = false)
        {
            this.handle = isPipe ?
                THPipeFile_new(name, mode, isQuiet ? 1 : 0) :
                THDiskFile_new(name, mode, isQuiet ? 1 : 0);
        }

        [DllImport("caffe2")] extern static string THDiskFile_name(File.HType self);

        /// <summary>
        ///   The file name.
        /// </summary>
        public string Name {  get { return THDiskFile_name(this.handle); } }

        [DllImport("caffe2")] extern static int THDiskFile_isLittleEndianCPU();
        [DllImport("caffe2")] extern static int THDiskFile_isBigEndianCPU();
        [DllImport("caffe2")] extern static void THDiskFile_nativeEndianEncoding(File.HType self);
        [DllImport("caffe2")] extern static void THDiskFile_littleEndianEncoding(File.HType self);
        [DllImport("caffe2")] extern static void THDiskFile_bigEndianEncoding(File.HType self);
        [DllImport("caffe2")] extern static void THDiskFile_longSize(File.HType self, int size);

        /// <summary>
        ///    Sets the size of long values.
        /// </summary>
        /// <remarks>Should be one of 0, 4, or 8.</remarks>
        public int LongSize { set { THDiskFile_longSize(this.handle, value); } }

        [DllImport("caffe2")] extern static void THDiskFile_noBuffer(File.HType self);
    }
}
