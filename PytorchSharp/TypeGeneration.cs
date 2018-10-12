using System;
using System.Runtime.InteropServices;

namespace PytorchSharp {

    public class ByteStorage : IDisposable {
        IntPtr handle;
        
        [DllImport ("caffe2")]
        extern static IntPtr THByteStorage_new ();
        
        public ByteStorage ()
        {
            handle = THByteStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THByteStorage_new_with_size (IntPtr size);
        
        public ByteStorage (long size)
        {
            handle = THByteStorage_new_with_size ((IntPtr) size);
        }
        
        ~ByteStorage ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                
                THByteStorage_free (handle);
                handle = IntPtr.Zero;
            }
        }
        
        public byte this [long index] {
            get {
                return  default;
            }
            set {
            }
        }
    }
