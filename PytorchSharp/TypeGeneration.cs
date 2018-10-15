using System;
using System.Runtime.InteropServices;

namespace PytorchSharp {

    public class ByteStorage : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The storage has been disposed");
        }

        [DllImport ("caffe2")]
        extern static SafeHandle THByteStorage_new ();
        
        public ByteStorage ()
        {
            handle = THByteStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THByteStorage_new_with_size (IntPtr size);
        
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
        
        [DllImport ("caffe2")]
        extern static void TH_ByteStorage_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ByteStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static byte TH_ByteStorage_get (SafeHandle handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_ByteStorage_set (SafeHandle handle, /*ptrdiff_t*/IntPtr pos,  byte value);
        
        public byte this [long index] {
            get {
                CheckHandle ();
            
                return  TH_ByteStorage_get (handle, (IntPtr) (index));
            }
            set {
                CheckHandle ();

                TH_ByteStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static byte TH_ByteStorage_resize (SafeHandle handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            CheckHandle ();
            TH_ByteStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_ByteStorage_fill (SafeHandle handle, byte value);
        
        public void Fill (byte value)
        {
            CheckHandle ();
            TH_ByteStorage_fill (handle, value);
        }
    }
    
    public class ByteTensor : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The tensor has been disposed");
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THByteTensor_new ();
        
        public ByteTensor ()
        {
            handle = THByteTensor_new ();
        }
        
        ~ByteTensor ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ByteTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
    }

    public class ShortStorage : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The storage has been disposed");
        }

        [DllImport ("caffe2")]
        extern static SafeHandle THShortStorage_new ();
        
        public ShortStorage ()
        {
            handle = THShortStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THShortStorage_new_with_size (IntPtr size);
        
        public ShortStorage (long size)
        {
            handle = THShortStorage_new_with_size ((IntPtr) size);
        }
        
        ~ShortStorage ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortStorage_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ShortStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static short TH_ShortStorage_get (SafeHandle handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_ShortStorage_set (SafeHandle handle, /*ptrdiff_t*/IntPtr pos,  short value);
        
        public short this [long index] {
            get {
                CheckHandle ();
            
                return  TH_ShortStorage_get (handle, (IntPtr) (index));
            }
            set {
                CheckHandle ();

                TH_ShortStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static short TH_ShortStorage_resize (SafeHandle handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            CheckHandle ();
            TH_ShortStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_ShortStorage_fill (SafeHandle handle, short value);
        
        public void Fill (short value)
        {
            CheckHandle ();
            TH_ShortStorage_fill (handle, value);
        }
    }
    
    public class ShortTensor : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The tensor has been disposed");
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THShortTensor_new ();
        
        public ShortTensor ()
        {
            handle = THShortTensor_new ();
        }
        
        ~ShortTensor ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ShortTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
    }

    public class IntStorage : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The storage has been disposed");
        }

        [DllImport ("caffe2")]
        extern static SafeHandle THIntStorage_new ();
        
        public IntStorage ()
        {
            handle = THIntStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THIntStorage_new_with_size (IntPtr size);
        
        public IntStorage (long size)
        {
            handle = THIntStorage_new_with_size ((IntPtr) size);
        }
        
        ~IntStorage ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntStorage_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_IntStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static int TH_IntStorage_get (SafeHandle handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_IntStorage_set (SafeHandle handle, /*ptrdiff_t*/IntPtr pos,  int value);
        
        public int this [long index] {
            get {
                CheckHandle ();
            
                return  TH_IntStorage_get (handle, (IntPtr) (index));
            }
            set {
                CheckHandle ();

                TH_IntStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static int TH_IntStorage_resize (SafeHandle handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            CheckHandle ();
            TH_IntStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_IntStorage_fill (SafeHandle handle, int value);
        
        public void Fill (int value)
        {
            CheckHandle ();
            TH_IntStorage_fill (handle, value);
        }
    }
    
    public class IntTensor : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The tensor has been disposed");
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THIntTensor_new ();
        
        public IntTensor ()
        {
            handle = THIntTensor_new ();
        }
        
        ~IntTensor ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_IntTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
    }

    public class LongStorage : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The storage has been disposed");
        }

        [DllImport ("caffe2")]
        extern static SafeHandle THLongStorage_new ();
        
        public LongStorage ()
        {
            handle = THLongStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THLongStorage_new_with_size (IntPtr size);
        
        public LongStorage (long size)
        {
            handle = THLongStorage_new_with_size ((IntPtr) size);
        }
        
        ~LongStorage ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongStorage_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_LongStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static long TH_LongStorage_get (SafeHandle handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_LongStorage_set (SafeHandle handle, /*ptrdiff_t*/IntPtr pos,  long value);
        
        public long this [long index] {
            get {
                CheckHandle ();
            
                return  TH_LongStorage_get (handle, (IntPtr) (index));
            }
            set {
                CheckHandle ();

                TH_LongStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static long TH_LongStorage_resize (SafeHandle handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            CheckHandle ();
            TH_LongStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_LongStorage_fill (SafeHandle handle, long value);
        
        public void Fill (long value)
        {
            CheckHandle ();
            TH_LongStorage_fill (handle, value);
        }
    }
    
    public class LongTensor : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The tensor has been disposed");
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THLongTensor_new ();
        
        public LongTensor ()
        {
            handle = THLongTensor_new ();
        }
        
        ~LongTensor ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_LongTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
    }

    public class DoubleStorage : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The storage has been disposed");
        }

        [DllImport ("caffe2")]
        extern static SafeHandle THDoubleStorage_new ();
        
        public DoubleStorage ()
        {
            handle = THDoubleStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THDoubleStorage_new_with_size (IntPtr size);
        
        public DoubleStorage (long size)
        {
            handle = THDoubleStorage_new_with_size ((IntPtr) size);
        }
        
        ~DoubleStorage ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleStorage_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_DoubleStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static double TH_DoubleStorage_get (SafeHandle handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleStorage_set (SafeHandle handle, /*ptrdiff_t*/IntPtr pos,  double value);
        
        public double this [long index] {
            get {
                CheckHandle ();
            
                return  TH_DoubleStorage_get (handle, (IntPtr) (index));
            }
            set {
                CheckHandle ();

                TH_DoubleStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static double TH_DoubleStorage_resize (SafeHandle handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            CheckHandle ();
            TH_DoubleStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_DoubleStorage_fill (SafeHandle handle, double value);
        
        public void Fill (double value)
        {
            CheckHandle ();
            TH_DoubleStorage_fill (handle, value);
        }
    }
    
    public class DoubleTensor : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The tensor has been disposed");
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THDoubleTensor_new ();
        
        public DoubleTensor ()
        {
            handle = THDoubleTensor_new ();
        }
        
        ~DoubleTensor ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_DoubleTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
    }

    public class FloatStorage : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The storage has been disposed");
        }

        [DllImport ("caffe2")]
        extern static SafeHandle THFloatStorage_new ();
        
        public FloatStorage ()
        {
            handle = THFloatStorage_new ();
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THFloatStorage_new_with_size (IntPtr size);
        
        public FloatStorage (long size)
        {
            handle = THFloatStorage_new_with_size ((IntPtr) size);
        }
        
        ~FloatStorage ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatStorage_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_FloatStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static float TH_FloatStorage_get (SafeHandle handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_FloatStorage_set (SafeHandle handle, /*ptrdiff_t*/IntPtr pos,  float value);
        
        public float this [long index] {
            get {
                CheckHandle ();
            
                return  TH_FloatStorage_get (handle, (IntPtr) (index));
            }
            set {
                CheckHandle ();

                TH_FloatStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static float TH_FloatStorage_resize (SafeHandle handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            CheckHandle ();
            TH_FloatStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_FloatStorage_fill (SafeHandle handle, float value);
        
        public void Fill (float value)
        {
            CheckHandle ();
            TH_FloatStorage_fill (handle, value);
        }
    }
    
    public class FloatTensor : IDisposable {
        SafeHandle handle;
        
        void CheckHandle ()
        {
            if (handle.IsInvalid)
                throw new ObjectDisposedException ("The tensor has been disposed");
        }
        
        [DllImport ("caffe2")]
        extern static SafeHandle THFloatTensor_new ();
        
        public FloatTensor ()
        {
            handle = THFloatTensor_new ();
        }
        
        ~FloatTensor ()
        {
            Dispose (false);
        }
        
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_free (SafeHandle handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_FloatTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
    }
}