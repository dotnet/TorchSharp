using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using HType=Microsoft.Win32.SafeHandles.SafeHandleZeroOrMinusOneIsInvalid;

namespace PytorchSharp {

    public class ByteStorage : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THByteStorage_new ();
        
        public ByteStorage ()
        {
            handle = THByteStorage_new ();
        }
        
        internal ByteStorage (HType fromHandle)
        {
            this.handle = fromHandle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THByteStorage_new_with_size (IntPtr size);
        
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
        extern static void TH_ByteStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ByteStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static byte TH_ByteStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_ByteStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  byte value);
        
        public byte this [long index] {
            get => TH_ByteStorage_get (handle, (IntPtr) (index));
            set {
                TH_ByteStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static byte TH_ByteStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            TH_ByteStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_ByteStorage_fill (HType handle, byte value);
        
        public void Fill (byte value)
        {
            TH_ByteStorage_fill (handle, value);
        }
    }
    
    public class ByteTensor : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public ByteTensor ()
        {
            handle = THByteTensor_new ();
        }
        
        public ByteTensor (HType handle)
        {
            this.handle = handle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newWithSize1d (long size0);
        /// <summary>
        ///    Creates a 1D tensor of the specified size.
        /// </summary>    
        /// <param name="size0">Size for the first dimension.</param>

        public ByteTensor (long size0)
        {
            handle = THByteTensor_newWithSize1d (size0);
        }
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newWithSize2d (long size0, long size1);
        
        /// <summary>
        ///    Creates a 2D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        public ByteTensor (long size0, long size1)
        {
            handle = THByteTensor_newWithSize2d (size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newWithSize3d (long size0, long size1, long size2);
        /// <summary>
        ///    Creates a 3D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        public ByteTensor (long size0, long size1, long size2)
        {
            handle = THByteTensor_newWithSize3d (size0, size1, size2);
        }
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newWithSize4d (long size0, long size1, long size2, long size3);
        /// <summary>
        ///    Creates a 4D tensor of the specified size.
        /// </summary>
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        /// <param name="size3">Size for the fourth dimension.</param>
        public ByteTensor (long size0, long size1, long size2, long size3)
        {
            handle = THByteTensor_newWithSize4d (size0, size1, size2, size3);
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
        extern static void TH_ByteTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ByteTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            TH_ByteTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_fill (HType handle, byte value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (byte value)
        {
            TH_ByteTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static HType TH_ByteTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public ByteStorage Storage => new ByteStorage (TH_ByteTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int TH_ByteTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => TH_ByteTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long TH_ByteTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return TH_ByteTensor_size (handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long [] Shape {
            get {
                    var dims = new long [Dimensions];
                    for (int i = 0; i < dims.Length; i++)
                            dims [i] = (long)GetTensorDimension (i);

                    return dims;
            }
        }


        [DllImport ("caffe2")]
        extern static long TH_ByteTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return TH_ByteTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr TH_ByteTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe byte *Data => (byte*) TH_ByteTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType TH_ByteTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public ByteTensor Clone () => new ByteTensor (TH_ByteTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType TH_ByteTensor_newSelect (HType handle, int dim, long slideIndex);
        
        public ByteTensor Select (int dim, long slideIndex) => new ByteTensor (TH_ByteTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType TH_ByteTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public ByteTensor Narrow (int dim, long firstIndex, long size) => new ByteTensor (TH_ByteTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType TH_ByteTensor_newTranspose (HType handle, int dim1, int dim2);
        public ByteTensor Transpose (int dim1, int dim2) => new ByteTensor (TH_ByteTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType TH_ByteTensor_newUnfold (HType handle, int dim1, long size, long step);
        public ByteTensor Unfold (int dim, long size, long step) => new ByteTensor (TH_ByteTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            TH_ByteTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            TH_ByteTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            TH_ByteTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            TH_ByteTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            TH_ByteTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (ByteTensor src)
        {
            TH_ByteTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_set (HType handle, HType src);
        
        public void Set (ByteTensor src)
        {
            TH_ByteTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_set1d (HType handle, long x0, byte value);
        [DllImport ("caffe2")]
        extern static byte TH_ByteTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public byte this [long x0] {
            get => TH_ByteTensor_get1d (handle, x0);
            set => TH_ByteTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_set2d (HType handle, long x0, long x1, byte value);
        [DllImport ("caffe2")]
        extern static byte TH_ByteTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public byte this [long x0, long x1] {
            get => TH_ByteTensor_get2d (handle, x0, x1);
            set => TH_ByteTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_set3d (HType handle, long x0, long x1, long x2, byte value);
        [DllImport ("caffe2")]
        extern static byte TH_ByteTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public byte this [long x0, long x1, long x2] {
            get => TH_ByteTensor_get3d (handle, x0, x1, x2);
            set => TH_ByteTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void TH_ByteTensor_set4d (HType handle, long x0, long x1, long x2, long x3, byte value);
        [DllImport ("caffe2")]
        extern static byte TH_ByteTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public byte this [long x0, long x1, long x2, long x3] {
            get => TH_ByteTensor_get4d (handle, x0, x1, x2, x3);
            set => TH_ByteTensor_set4d (handle, x0, x1, x2, x3, value);
        }
                
        
        #if false
        [DllImport ("caffe2")]
        extern static string TH_ByteTensor_
        #endif
    }

    public class ShortStorage : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THShortStorage_new ();
        
        public ShortStorage ()
        {
            handle = THShortStorage_new ();
        }
        
        internal ShortStorage (HType fromHandle)
        {
            this.handle = fromHandle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THShortStorage_new_with_size (IntPtr size);
        
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
        extern static void TH_ShortStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ShortStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static short TH_ShortStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_ShortStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  short value);
        
        public short this [long index] {
            get => TH_ShortStorage_get (handle, (IntPtr) (index));
            set {
                TH_ShortStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static short TH_ShortStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            TH_ShortStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_ShortStorage_fill (HType handle, short value);
        
        public void Fill (short value)
        {
            TH_ShortStorage_fill (handle, value);
        }
    }
    
    public class ShortTensor : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public ShortTensor ()
        {
            handle = THShortTensor_new ();
        }
        
        public ShortTensor (HType handle)
        {
            this.handle = handle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newWithSize1d (long size0);
        /// <summary>
        ///    Creates a 1D tensor of the specified size.
        /// </summary>    
        /// <param name="size0">Size for the first dimension.</param>

        public ShortTensor (long size0)
        {
            handle = THShortTensor_newWithSize1d (size0);
        }
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newWithSize2d (long size0, long size1);
        
        /// <summary>
        ///    Creates a 2D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        public ShortTensor (long size0, long size1)
        {
            handle = THShortTensor_newWithSize2d (size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newWithSize3d (long size0, long size1, long size2);
        /// <summary>
        ///    Creates a 3D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        public ShortTensor (long size0, long size1, long size2)
        {
            handle = THShortTensor_newWithSize3d (size0, size1, size2);
        }
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newWithSize4d (long size0, long size1, long size2, long size3);
        /// <summary>
        ///    Creates a 4D tensor of the specified size.
        /// </summary>
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        /// <param name="size3">Size for the fourth dimension.</param>
        public ShortTensor (long size0, long size1, long size2, long size3)
        {
            handle = THShortTensor_newWithSize4d (size0, size1, size2, size3);
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
        extern static void TH_ShortTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_ShortTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            TH_ShortTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_fill (HType handle, short value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (short value)
        {
            TH_ShortTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static HType TH_ShortTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public ShortStorage Storage => new ShortStorage (TH_ShortTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int TH_ShortTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => TH_ShortTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long TH_ShortTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return TH_ShortTensor_size (handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long [] Shape {
            get {
                    var dims = new long [Dimensions];
                    for (int i = 0; i < dims.Length; i++)
                            dims [i] = (long)GetTensorDimension (i);

                    return dims;
            }
        }


        [DllImport ("caffe2")]
        extern static long TH_ShortTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return TH_ShortTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr TH_ShortTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe short *Data => (short*) TH_ShortTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType TH_ShortTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public ShortTensor Clone () => new ShortTensor (TH_ShortTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType TH_ShortTensor_newSelect (HType handle, int dim, long slideIndex);
        
        public ShortTensor Select (int dim, long slideIndex) => new ShortTensor (TH_ShortTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType TH_ShortTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public ShortTensor Narrow (int dim, long firstIndex, long size) => new ShortTensor (TH_ShortTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType TH_ShortTensor_newTranspose (HType handle, int dim1, int dim2);
        public ShortTensor Transpose (int dim1, int dim2) => new ShortTensor (TH_ShortTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType TH_ShortTensor_newUnfold (HType handle, int dim1, long size, long step);
        public ShortTensor Unfold (int dim, long size, long step) => new ShortTensor (TH_ShortTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            TH_ShortTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            TH_ShortTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            TH_ShortTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            TH_ShortTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            TH_ShortTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (ShortTensor src)
        {
            TH_ShortTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_set (HType handle, HType src);
        
        public void Set (ShortTensor src)
        {
            TH_ShortTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_set1d (HType handle, long x0, short value);
        [DllImport ("caffe2")]
        extern static short TH_ShortTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public short this [long x0] {
            get => TH_ShortTensor_get1d (handle, x0);
            set => TH_ShortTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_set2d (HType handle, long x0, long x1, short value);
        [DllImport ("caffe2")]
        extern static short TH_ShortTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public short this [long x0, long x1] {
            get => TH_ShortTensor_get2d (handle, x0, x1);
            set => TH_ShortTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_set3d (HType handle, long x0, long x1, long x2, short value);
        [DllImport ("caffe2")]
        extern static short TH_ShortTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public short this [long x0, long x1, long x2] {
            get => TH_ShortTensor_get3d (handle, x0, x1, x2);
            set => TH_ShortTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void TH_ShortTensor_set4d (HType handle, long x0, long x1, long x2, long x3, short value);
        [DllImport ("caffe2")]
        extern static short TH_ShortTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public short this [long x0, long x1, long x2, long x3] {
            get => TH_ShortTensor_get4d (handle, x0, x1, x2, x3);
            set => TH_ShortTensor_set4d (handle, x0, x1, x2, x3, value);
        }
                
        
        #if false
        [DllImport ("caffe2")]
        extern static string TH_ShortTensor_
        #endif
    }

    public class IntStorage : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THIntStorage_new ();
        
        public IntStorage ()
        {
            handle = THIntStorage_new ();
        }
        
        internal IntStorage (HType fromHandle)
        {
            this.handle = fromHandle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THIntStorage_new_with_size (IntPtr size);
        
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
        extern static void TH_IntStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_IntStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static int TH_IntStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_IntStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  int value);
        
        public int this [long index] {
            get => TH_IntStorage_get (handle, (IntPtr) (index));
            set {
                TH_IntStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static int TH_IntStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            TH_IntStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_IntStorage_fill (HType handle, int value);
        
        public void Fill (int value)
        {
            TH_IntStorage_fill (handle, value);
        }
    }
    
    public class IntTensor : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public IntTensor ()
        {
            handle = THIntTensor_new ();
        }
        
        public IntTensor (HType handle)
        {
            this.handle = handle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newWithSize1d (long size0);
        /// <summary>
        ///    Creates a 1D tensor of the specified size.
        /// </summary>    
        /// <param name="size0">Size for the first dimension.</param>

        public IntTensor (long size0)
        {
            handle = THIntTensor_newWithSize1d (size0);
        }
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newWithSize2d (long size0, long size1);
        
        /// <summary>
        ///    Creates a 2D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        public IntTensor (long size0, long size1)
        {
            handle = THIntTensor_newWithSize2d (size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newWithSize3d (long size0, long size1, long size2);
        /// <summary>
        ///    Creates a 3D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        public IntTensor (long size0, long size1, long size2)
        {
            handle = THIntTensor_newWithSize3d (size0, size1, size2);
        }
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newWithSize4d (long size0, long size1, long size2, long size3);
        /// <summary>
        ///    Creates a 4D tensor of the specified size.
        /// </summary>
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        /// <param name="size3">Size for the fourth dimension.</param>
        public IntTensor (long size0, long size1, long size2, long size3)
        {
            handle = THIntTensor_newWithSize4d (size0, size1, size2, size3);
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
        extern static void TH_IntTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_IntTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            TH_IntTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_fill (HType handle, int value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (int value)
        {
            TH_IntTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static HType TH_IntTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public IntStorage Storage => new IntStorage (TH_IntTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int TH_IntTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => TH_IntTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long TH_IntTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return TH_IntTensor_size (handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long [] Shape {
            get {
                    var dims = new long [Dimensions];
                    for (int i = 0; i < dims.Length; i++)
                            dims [i] = (long)GetTensorDimension (i);

                    return dims;
            }
        }


        [DllImport ("caffe2")]
        extern static long TH_IntTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return TH_IntTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr TH_IntTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe int *Data => (int*) TH_IntTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType TH_IntTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public IntTensor Clone () => new IntTensor (TH_IntTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType TH_IntTensor_newSelect (HType handle, int dim, long slideIndex);
        
        public IntTensor Select (int dim, long slideIndex) => new IntTensor (TH_IntTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType TH_IntTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public IntTensor Narrow (int dim, long firstIndex, long size) => new IntTensor (TH_IntTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType TH_IntTensor_newTranspose (HType handle, int dim1, int dim2);
        public IntTensor Transpose (int dim1, int dim2) => new IntTensor (TH_IntTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType TH_IntTensor_newUnfold (HType handle, int dim1, long size, long step);
        public IntTensor Unfold (int dim, long size, long step) => new IntTensor (TH_IntTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            TH_IntTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            TH_IntTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            TH_IntTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void TH_IntTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            TH_IntTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            TH_IntTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (IntTensor src)
        {
            TH_IntTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_set (HType handle, HType src);
        
        public void Set (IntTensor src)
        {
            TH_IntTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_set1d (HType handle, long x0, int value);
        [DllImport ("caffe2")]
        extern static int TH_IntTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public int this [long x0] {
            get => TH_IntTensor_get1d (handle, x0);
            set => TH_IntTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_set2d (HType handle, long x0, long x1, int value);
        [DllImport ("caffe2")]
        extern static int TH_IntTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public int this [long x0, long x1] {
            get => TH_IntTensor_get2d (handle, x0, x1);
            set => TH_IntTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void TH_IntTensor_set3d (HType handle, long x0, long x1, long x2, int value);
        [DllImport ("caffe2")]
        extern static int TH_IntTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public int this [long x0, long x1, long x2] {
            get => TH_IntTensor_get3d (handle, x0, x1, x2);
            set => TH_IntTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void TH_IntTensor_set4d (HType handle, long x0, long x1, long x2, long x3, int value);
        [DllImport ("caffe2")]
        extern static int TH_IntTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public int this [long x0, long x1, long x2, long x3] {
            get => TH_IntTensor_get4d (handle, x0, x1, x2, x3);
            set => TH_IntTensor_set4d (handle, x0, x1, x2, x3, value);
        }
                
        
        #if false
        [DllImport ("caffe2")]
        extern static string TH_IntTensor_
        #endif
    }

    public class LongStorage : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THLongStorage_new ();
        
        public LongStorage ()
        {
            handle = THLongStorage_new ();
        }
        
        internal LongStorage (HType fromHandle)
        {
            this.handle = fromHandle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THLongStorage_new_with_size (IntPtr size);
        
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
        extern static void TH_LongStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_LongStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static long TH_LongStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_LongStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  long value);
        
        public long this [long index] {
            get => TH_LongStorage_get (handle, (IntPtr) (index));
            set {
                TH_LongStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static long TH_LongStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            TH_LongStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_LongStorage_fill (HType handle, long value);
        
        public void Fill (long value)
        {
            TH_LongStorage_fill (handle, value);
        }
    }
    
    public class LongTensor : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public LongTensor ()
        {
            handle = THLongTensor_new ();
        }
        
        public LongTensor (HType handle)
        {
            this.handle = handle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newWithSize1d (long size0);
        /// <summary>
        ///    Creates a 1D tensor of the specified size.
        /// </summary>    
        /// <param name="size0">Size for the first dimension.</param>

        public LongTensor (long size0)
        {
            handle = THLongTensor_newWithSize1d (size0);
        }
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newWithSize2d (long size0, long size1);
        
        /// <summary>
        ///    Creates a 2D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        public LongTensor (long size0, long size1)
        {
            handle = THLongTensor_newWithSize2d (size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newWithSize3d (long size0, long size1, long size2);
        /// <summary>
        ///    Creates a 3D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        public LongTensor (long size0, long size1, long size2)
        {
            handle = THLongTensor_newWithSize3d (size0, size1, size2);
        }
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newWithSize4d (long size0, long size1, long size2, long size3);
        /// <summary>
        ///    Creates a 4D tensor of the specified size.
        /// </summary>
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        /// <param name="size3">Size for the fourth dimension.</param>
        public LongTensor (long size0, long size1, long size2, long size3)
        {
            handle = THLongTensor_newWithSize4d (size0, size1, size2, size3);
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
        extern static void TH_LongTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_LongTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            TH_LongTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_fill (HType handle, long value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (long value)
        {
            TH_LongTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static HType TH_LongTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public LongStorage Storage => new LongStorage (TH_LongTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int TH_LongTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => TH_LongTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long TH_LongTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return TH_LongTensor_size (handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long [] Shape {
            get {
                    var dims = new long [Dimensions];
                    for (int i = 0; i < dims.Length; i++)
                            dims [i] = (long)GetTensorDimension (i);

                    return dims;
            }
        }


        [DllImport ("caffe2")]
        extern static long TH_LongTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return TH_LongTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr TH_LongTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe long *Data => (long*) TH_LongTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType TH_LongTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public LongTensor Clone () => new LongTensor (TH_LongTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType TH_LongTensor_newSelect (HType handle, int dim, long slideIndex);
        
        public LongTensor Select (int dim, long slideIndex) => new LongTensor (TH_LongTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType TH_LongTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public LongTensor Narrow (int dim, long firstIndex, long size) => new LongTensor (TH_LongTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType TH_LongTensor_newTranspose (HType handle, int dim1, int dim2);
        public LongTensor Transpose (int dim1, int dim2) => new LongTensor (TH_LongTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType TH_LongTensor_newUnfold (HType handle, int dim1, long size, long step);
        public LongTensor Unfold (int dim, long size, long step) => new LongTensor (TH_LongTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            TH_LongTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            TH_LongTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            TH_LongTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void TH_LongTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            TH_LongTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            TH_LongTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (LongTensor src)
        {
            TH_LongTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_set (HType handle, HType src);
        
        public void Set (LongTensor src)
        {
            TH_LongTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_set1d (HType handle, long x0, long value);
        [DllImport ("caffe2")]
        extern static long TH_LongTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public long this [long x0] {
            get => TH_LongTensor_get1d (handle, x0);
            set => TH_LongTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_set2d (HType handle, long x0, long x1, long value);
        [DllImport ("caffe2")]
        extern static long TH_LongTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public long this [long x0, long x1] {
            get => TH_LongTensor_get2d (handle, x0, x1);
            set => TH_LongTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void TH_LongTensor_set3d (HType handle, long x0, long x1, long x2, long value);
        [DllImport ("caffe2")]
        extern static long TH_LongTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public long this [long x0, long x1, long x2] {
            get => TH_LongTensor_get3d (handle, x0, x1, x2);
            set => TH_LongTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void TH_LongTensor_set4d (HType handle, long x0, long x1, long x2, long x3, long value);
        [DllImport ("caffe2")]
        extern static long TH_LongTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public long this [long x0, long x1, long x2, long x3] {
            get => TH_LongTensor_get4d (handle, x0, x1, x2, x3);
            set => TH_LongTensor_set4d (handle, x0, x1, x2, x3, value);
        }
                
        
        #if false
        [DllImport ("caffe2")]
        extern static string TH_LongTensor_
        #endif
    }

    public class DoubleStorage : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THDoubleStorage_new ();
        
        public DoubleStorage ()
        {
            handle = THDoubleStorage_new ();
        }
        
        internal DoubleStorage (HType fromHandle)
        {
            this.handle = fromHandle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THDoubleStorage_new_with_size (IntPtr size);
        
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
        extern static void TH_DoubleStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_DoubleStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static double TH_DoubleStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  double value);
        
        public double this [long index] {
            get => TH_DoubleStorage_get (handle, (IntPtr) (index));
            set {
                TH_DoubleStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static double TH_DoubleStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            TH_DoubleStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_DoubleStorage_fill (HType handle, double value);
        
        public void Fill (double value)
        {
            TH_DoubleStorage_fill (handle, value);
        }
    }
    
    public class DoubleTensor : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public DoubleTensor ()
        {
            handle = THDoubleTensor_new ();
        }
        
        public DoubleTensor (HType handle)
        {
            this.handle = handle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newWithSize1d (long size0);
        /// <summary>
        ///    Creates a 1D tensor of the specified size.
        /// </summary>    
        /// <param name="size0">Size for the first dimension.</param>

        public DoubleTensor (long size0)
        {
            handle = THDoubleTensor_newWithSize1d (size0);
        }
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newWithSize2d (long size0, long size1);
        
        /// <summary>
        ///    Creates a 2D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        public DoubleTensor (long size0, long size1)
        {
            handle = THDoubleTensor_newWithSize2d (size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newWithSize3d (long size0, long size1, long size2);
        /// <summary>
        ///    Creates a 3D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        public DoubleTensor (long size0, long size1, long size2)
        {
            handle = THDoubleTensor_newWithSize3d (size0, size1, size2);
        }
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newWithSize4d (long size0, long size1, long size2, long size3);
        /// <summary>
        ///    Creates a 4D tensor of the specified size.
        /// </summary>
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        /// <param name="size3">Size for the fourth dimension.</param>
        public DoubleTensor (long size0, long size1, long size2, long size3)
        {
            handle = THDoubleTensor_newWithSize4d (size0, size1, size2, size3);
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
        extern static void TH_DoubleTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_DoubleTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            TH_DoubleTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_fill (HType handle, double value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (double value)
        {
            TH_DoubleTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static HType TH_DoubleTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public DoubleStorage Storage => new DoubleStorage (TH_DoubleTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int TH_DoubleTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => TH_DoubleTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long TH_DoubleTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return TH_DoubleTensor_size (handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long [] Shape {
            get {
                    var dims = new long [Dimensions];
                    for (int i = 0; i < dims.Length; i++)
                            dims [i] = (long)GetTensorDimension (i);

                    return dims;
            }
        }


        [DllImport ("caffe2")]
        extern static long TH_DoubleTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return TH_DoubleTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr TH_DoubleTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe double *Data => (double*) TH_DoubleTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType TH_DoubleTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public DoubleTensor Clone () => new DoubleTensor (TH_DoubleTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType TH_DoubleTensor_newSelect (HType handle, int dim, long slideIndex);
        
        public DoubleTensor Select (int dim, long slideIndex) => new DoubleTensor (TH_DoubleTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType TH_DoubleTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public DoubleTensor Narrow (int dim, long firstIndex, long size) => new DoubleTensor (TH_DoubleTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType TH_DoubleTensor_newTranspose (HType handle, int dim1, int dim2);
        public DoubleTensor Transpose (int dim1, int dim2) => new DoubleTensor (TH_DoubleTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType TH_DoubleTensor_newUnfold (HType handle, int dim1, long size, long step);
        public DoubleTensor Unfold (int dim, long size, long step) => new DoubleTensor (TH_DoubleTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            TH_DoubleTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            TH_DoubleTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            TH_DoubleTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            TH_DoubleTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            TH_DoubleTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (DoubleTensor src)
        {
            TH_DoubleTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_set (HType handle, HType src);
        
        public void Set (DoubleTensor src)
        {
            TH_DoubleTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_set1d (HType handle, long x0, double value);
        [DllImport ("caffe2")]
        extern static double TH_DoubleTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public double this [long x0] {
            get => TH_DoubleTensor_get1d (handle, x0);
            set => TH_DoubleTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_set2d (HType handle, long x0, long x1, double value);
        [DllImport ("caffe2")]
        extern static double TH_DoubleTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public double this [long x0, long x1] {
            get => TH_DoubleTensor_get2d (handle, x0, x1);
            set => TH_DoubleTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_set3d (HType handle, long x0, long x1, long x2, double value);
        [DllImport ("caffe2")]
        extern static double TH_DoubleTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public double this [long x0, long x1, long x2] {
            get => TH_DoubleTensor_get3d (handle, x0, x1, x2);
            set => TH_DoubleTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void TH_DoubleTensor_set4d (HType handle, long x0, long x1, long x2, long x3, double value);
        [DllImport ("caffe2")]
        extern static double TH_DoubleTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public double this [long x0, long x1, long x2, long x3] {
            get => TH_DoubleTensor_get4d (handle, x0, x1, x2, x3);
            set => TH_DoubleTensor_set4d (handle, x0, x1, x2, x3, value);
        }
                
        
        #if false
        [DllImport ("caffe2")]
        extern static string TH_DoubleTensor_
        #endif
    }

    public class FloatStorage : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THFloatStorage_new ();
        
        public FloatStorage ()
        {
            handle = THFloatStorage_new ();
        }
        
        internal FloatStorage (HType fromHandle)
        {
            this.handle = fromHandle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THFloatStorage_new_with_size (IntPtr size);
        
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
        extern static void TH_FloatStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_FloatStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static float TH_FloatStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        
        [DllImport ("caffe2")]
        extern static void TH_FloatStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  float value);
        
        public float this [long index] {
            get => TH_FloatStorage_get (handle, (IntPtr) (index));
            set {
                TH_FloatStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static float TH_FloatStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            TH_FloatStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void TH_FloatStorage_fill (HType handle, float value);
        
        public void Fill (float value)
        {
            TH_FloatStorage_fill (handle, value);
        }
    }
    
    public class FloatTensor : IDisposable {
        HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public FloatTensor ()
        {
            handle = THFloatTensor_new ();
        }
        
        public FloatTensor (HType handle)
        {
            this.handle = handle;
        }
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newWithSize1d (long size0);
        /// <summary>
        ///    Creates a 1D tensor of the specified size.
        /// </summary>    
        /// <param name="size0">Size for the first dimension.</param>

        public FloatTensor (long size0)
        {
            handle = THFloatTensor_newWithSize1d (size0);
        }
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newWithSize2d (long size0, long size1);
        
        /// <summary>
        ///    Creates a 2D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        public FloatTensor (long size0, long size1)
        {
            handle = THFloatTensor_newWithSize2d (size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newWithSize3d (long size0, long size1, long size2);
        /// <summary>
        ///    Creates a 3D tensor of the specified size.
        /// </summary>        
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        public FloatTensor (long size0, long size1, long size2)
        {
            handle = THFloatTensor_newWithSize3d (size0, size1, size2);
        }
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newWithSize4d (long size0, long size1, long size2, long size3);
        /// <summary>
        ///    Creates a 4D tensor of the specified size.
        /// </summary>
        /// <param name="size0">Size for the first dimension.</param>
        /// <param name="size1">Size for the second dimension.</param>
        /// <param name="size2">Size for the third dimension.</param>
        /// <param name="size3">Size for the fourth dimension.</param>
        public FloatTensor (long size0, long size1, long size2, long size3)
        {
            handle = THFloatTensor_newWithSize4d (size0, size1, size2, size3);
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
        extern static void TH_FloatTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                TH_FloatTensor_free (handle);
                handle.Dispose ();
            }
        }
        
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            TH_FloatTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_fill (HType handle, float value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (float value)
        {
            TH_FloatTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static HType TH_FloatTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public FloatStorage Storage => new FloatStorage (TH_FloatTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int TH_FloatTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => TH_FloatTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long TH_FloatTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return TH_FloatTensor_size (handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long [] Shape {
            get {
                    var dims = new long [Dimensions];
                    for (int i = 0; i < dims.Length; i++)
                            dims [i] = (long)GetTensorDimension (i);

                    return dims;
            }
        }


        [DllImport ("caffe2")]
        extern static long TH_FloatTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return TH_FloatTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr TH_FloatTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe float *Data => (float*) TH_FloatTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType TH_FloatTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public FloatTensor Clone () => new FloatTensor (TH_FloatTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType TH_FloatTensor_newSelect (HType handle, int dim, long slideIndex);
        
        public FloatTensor Select (int dim, long slideIndex) => new FloatTensor (TH_FloatTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType TH_FloatTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public FloatTensor Narrow (int dim, long firstIndex, long size) => new FloatTensor (TH_FloatTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType TH_FloatTensor_newTranspose (HType handle, int dim1, int dim2);
        public FloatTensor Transpose (int dim1, int dim2) => new FloatTensor (TH_FloatTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType TH_FloatTensor_newUnfold (HType handle, int dim1, long size, long step);
        public FloatTensor Unfold (int dim, long size, long step) => new FloatTensor (TH_FloatTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            TH_FloatTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            TH_FloatTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            TH_FloatTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            TH_FloatTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            TH_FloatTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (FloatTensor src)
        {
            TH_FloatTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_set (HType handle, HType src);
        
        public void Set (FloatTensor src)
        {
            TH_FloatTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_set1d (HType handle, long x0, float value);
        [DllImport ("caffe2")]
        extern static float TH_FloatTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public float this [long x0] {
            get => TH_FloatTensor_get1d (handle, x0);
            set => TH_FloatTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_set2d (HType handle, long x0, long x1, float value);
        [DllImport ("caffe2")]
        extern static float TH_FloatTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public float this [long x0, long x1] {
            get => TH_FloatTensor_get2d (handle, x0, x1);
            set => TH_FloatTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_set3d (HType handle, long x0, long x1, long x2, float value);
        [DllImport ("caffe2")]
        extern static float TH_FloatTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public float this [long x0, long x1, long x2] {
            get => TH_FloatTensor_get3d (handle, x0, x1, x2);
            set => TH_FloatTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void TH_FloatTensor_set4d (HType handle, long x0, long x1, long x2, long x3, float value);
        [DllImport ("caffe2")]
        extern static float TH_FloatTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        public float this [long x0, long x1, long x2, long x3] {
            get => TH_FloatTensor_get4d (handle, x0, x1, x2, x3);
            set => TH_FloatTensor_set4d (handle, x0, x1, x2, x3, value);
        }
    }
}