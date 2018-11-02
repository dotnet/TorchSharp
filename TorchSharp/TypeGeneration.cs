using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.Text;

namespace TorchSharp {
    public partial class ByteTensor : IDisposable {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        public class ByteStorage : IDisposable {
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
                extern static void THByteStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THByteStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THByteStorage_new ();
            
            /// <summary>
            ///   Initializes an empty ByteStorage instance.
            /// </summary>
            public ByteStorage ()
            {
                handle = THByteStorage_new ();
            }
            
            internal ByteStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THByteStorage_new_withSize (IntPtr size);
            
            /// <summary>
            ///   Initializes a ByteStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public ByteStorage (long size)
            {
                handle = THByteStorage_new_withSize ((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~ByteStorage ()
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
            extern static byte THByteStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
            [DllImport ("caffe2")]
            extern static void THByteStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  byte value);
            
            /// <summary>
            /// </summary>
            public byte this [long index] {
                get => THByteStorage_get (handle, (IntPtr) (index));
                set {
                    THByteStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static byte THByteStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THByteStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THByteStorage_fill (HType handle, byte value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (byte value)
            {
                THByteStorage_fill (handle, value);
            }
        }
    }
    
    /// <summary>
    ///   Tensor of type Byte.
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use the default constructor to create an empty tensor, or invoke one of the
    ///     constructors with one (1D), two (2D), three (3D), or four parameters (4D) to x
    ///     create a tensor for the desired number of dimensions.
    ///   </para>
    /// </remarks>
    public partial class ByteTensor : IDisposable {
        internal sealed class HType : SafeHandle {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }
                
            public override bool IsInvalid => handle == (IntPtr) 0;

            [DllImport ("caffe2")]
            extern static void THByteTensor_free (IntPtr handle);
                
            protected override bool ReleaseHandle ()
            {
                THByteTensor_free (handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public ByteTensor ()
        {
            handle = THByteTensor_new ();
        }

        internal ByteTensor (HType handle)
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
        
        /// <summary>
        ///  Finalizer for ~ByteTensor
        /// </summary>
        ~ByteTensor ()
        {
            Dispose (false);
        }
        
        /// <summary>
        ///   Releases the tensor and its associated data.
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
            if (disposing) {
                handle.Dispose ();
                handle = null;
            }
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            THByteTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_fill (HType handle, byte value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (byte value)
        {
            THByteTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_maskedFill (HType handle1, ByteTensor.HType handle2, byte value);
        
        /// <summary>
        ///  Fills the tensor with the specified value at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where the value should be filled.</param>
        /// <param name="value">The value to write at the indicated locations.</param>
        public void MaskedFill (ByteTensor mask, byte value)
        {
            THByteTensor_maskedFill (handle, mask.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_maskedCopy (HType handle1, ByteTensor.HType handle2, HType src);
        
        /// <summary>
        ///  Copies elements from the source tensor to the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the destination the value should be filled.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedCopy (ByteTensor mask, ByteTensor src)
        {
            THByteTensor_maskedCopy (handle, mask.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_maskedSelect (HType handle1, HType src, ByteTensor.HType handle2);
        
        /// <summary>
        ///  Copies elements from the source tensor at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the source the value should be fetched.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There will be as many elements in the tensor as there are 1s in the mask.
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedSelect (ByteTensor mask, ByteTensor src)
        {
            THByteTensor_maskedSelect (handle, src.handle, mask.handle);
        }

        [DllImport ("caffe2")]
        extern static ByteStorage.HType THByteTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public ByteStorage Storage => new ByteStorage (THByteTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int THByteTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => THByteTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long THByteTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return THByteTensor_size (handle, dim);
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
        extern static long THByteTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return THByteTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THByteTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe byte *Data => (byte*) THByteTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public ByteTensor Clone () => new ByteTensor (THByteTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newSelect (HType handle, int dim, long slideIndex);
        
        /// <summary>
        ///   Returns a new Tensor which is a tensor slice at the given index in the dimension dim. 
        /// </summary>
        /// <remarks>
        ///   The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.
        /// </remarks>
        /// <param name="dim">Dimension to select</param>
        /// <param name="slideIndex">Beginning of the tensor slice</param>
        public ByteTensor Select (int dim, long slideIndex) => new ByteTensor (THByteTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THByteTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        /// <summary>
        /// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from firstIndexto firstIndex+size-1.
        /// </summary>
        /// <param name="dim">The dimension to narrow</param>
        /// <param name="firstIndex">Initial index to narrow</param>
        /// <param name="size">Number of elements</param>
        public ByteTensor Narrow (int dim, long firstIndex, long size) => new ByteTensor (THByteTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newTranspose (HType handle, int dim1, int dim2);
        
        /// <summary>
        /// Returns a tensor where dimensions dim1 and dim2 have been swapped. 
        /// </summary>
        /// <param name="dim1">First dimension</param>
        /// <param name="dim2">Second dimension</param>
        public ByteTensor Transpose (int dim1, int dim2) => new ByteTensor (THByteTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newUnfold (HType handle, int dim1, long size, long step);
        
        /// <summary>
        ///   Returns a tensor which contains all slices of size size in the dimension dim. Step between two slices is given by step.
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        public ByteTensor Unfold (int dim, long size, long step) => new ByteTensor (THByteTensor_newUnfold (handle, dim, size, step));
        
        [DllImport("caffe2")]
        extern static HType THByteTensor_newWithStorage1d(ByteStorage.HType handle, IntPtr offset, long size, long stride);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size">Size of the first dimension.</param>     
        /// <param name="stride">Stride of the first dimension.</param>          
        public ByteTensor NewWithStorage1d(IntPtr offset, long size, long stride)
        {
            return new ByteTensor(THByteTensor_newWithStorage1d(Storage.handle, offset, size, stride));
        }

        [DllImport("caffe2")]
        extern static HType THByteTensor_newWithStorage2d(ByteStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        public ByteTensor NewWithStorage2d(IntPtr offset, long size0, long stride0, long size1, long stride1)
        {
            return new ByteTensor(THByteTensor_newWithStorage2d(Storage.handle, offset, size0, stride0, size1, stride1));
        }

        [DllImport("caffe2")]
        extern static HType THByteTensor_newWithStorage3d(ByteStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        public ByteTensor NewWithStorage3d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
        {
            return new ByteTensor(THByteTensor_newWithStorage3d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2));
        }

        [DllImport("caffe2")]
        extern static HType THByteTensor_newWithStorage4d(ByteStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        /// <param name="size3">Size of the forth dimension.</param>     
        /// <param name="stride3">Stride of the forth dimension.</param>
        public ByteTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new ByteTensor(THByteTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze ()
        {
            THByteTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze1d (ByteTensor src, int dimension)
        {
            THByteTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Unsqueeze1d (ByteTensor src, int dimension)
        {
            THByteTensor_unsqueeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_resize1d (HType handle, long size);
        
        /// <summary>
        ///   Resizes the tensor to be one dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size">The desired new size for the first dimension of the tensor.</param>
        public void Resize1d (long size)
        {
            THByteTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize2d (HType handle, long size0, long size1);
        /// <summary>
        ///   Resizes the tensor to be two dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        public void Resize2d (long size0, long size1)
        {
            THByteTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        /// <summary>
        ///   Resizes the tensor to be three dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        public void Resize3d (long size0, long size1, long size2)
        {
            THByteTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        /// <summary>
        ///   Resizes the tensor to be four dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THByteTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        /// <summary>
        ///   Resizes the tensor to be five dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        /// <param name="size4">The desired new size for the fifth dimension of the tensor.</param>
        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THByteTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resizeAs (HType handle, HType src);
       
        /// <summary>
        ///   Resizes the tensor to match the dimensions of the specified src tensor, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="src">The source tensor whose shape will be mirrored by this tensor.</param>
        public void ResizeAs (ByteTensor src)
        {
            THByteTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_set (HType handle, HType src);
        
        /// <summary>
        ///   The tensor will use the same storage as the provided source, so any changes to that tensor are visible on this one.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Set (ByteTensor src)
        {
            THByteTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_set1d (HType handle, long x0, byte value);
        [DllImport ("caffe2")]
        extern static byte THByteTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>       
        /// <param name="x0">Index to access.</param> 
        public byte this [long x0] {
            get => THByteTensor_get1d (handle, x0);
            set => THByteTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_set2d (HType handle, long x0, long x1, byte value);
        [DllImport ("caffe2")]
        extern static byte THByteTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>    
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        public byte this [long x0, long x1] {
            get => THByteTensor_get2d (handle, x0, x1);
            set => THByteTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_set3d (HType handle, long x0, long x1, long x2, byte value);
        [DllImport ("caffe2")]
        extern static byte THByteTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        public byte this [long x0, long x1, long x2] {
            get => THByteTensor_get3d (handle, x0, x1, x2);
            set => THByteTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_set4d (HType handle, long x0, long x1, long x2, long x3, byte value);
        [DllImport ("caffe2")]
        extern static byte THByteTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        /// <param name="x3">Index in the fourth dimension to access.</param>     
        public byte this [long x0, long x1, long x2, long x3] {
            get => THByteTensor_get4d (handle, x0, x1, x2, x3);
            set => THByteTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static byte THByteTensor_random (HType handle, IntPtr thgenerator);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Geometric (RandomGenerator source, double p)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_geometric (handle, source.handle, p);
        }
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using a newly initialized Random number geneator.
        /// </summary>
        /// <param name="n">The upper limit for the values to be generated</param>        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                CappedRandom (r, n);
        }

        
        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public override string ToString ()
        {
            var n = Dimensions;
            if (n == 0)
                    return "[]";

            StringBuilder sb = new StringBuilder ("[");
            for (int i = 0; i < n; i++) {
                    sb.Append (GetTensorDimension (i));
                    if (i + 1 < n)
                            sb.Append ("x");
            }
            sb.Append ("]");
            return sb.ToString ();
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_add (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Add operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Add operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Add (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_add (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Add operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Add(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Add (byte value)
        {
            var result = new ByteTensor ();
            Add (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_sub (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Sub operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Sub operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Sub (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_sub (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Sub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Sub(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Sub (byte value)
        {
            var result = new ByteTensor ();
            Sub (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_mul (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Mul operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Mul operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Mul (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_mul (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Mul operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Mul(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Mul (byte value)
        {
            var result = new ByteTensor ();
            Mul (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_div (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Div operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Div operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Div (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_div (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Div operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Div(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Div (byte value)
        {
            var result = new ByteTensor ();
            Div (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_lshift (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the LShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the LShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void LShift (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_lshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the LShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.LShift(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor LShift (byte value)
        {
            var result = new ByteTensor ();
            LShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_rshift (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the RShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the RShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void RShift (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_rshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the RShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.RShift(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor RShift (byte value)
        {
            var result = new ByteTensor ();
            RShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_fmod (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Fmod operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Fmod (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_fmod (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Fmod(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Fmod (byte value)
        {
            var result = new ByteTensor ();
            Fmod (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_remainder (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Remainder operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Remainder (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_remainder (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Remainder(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Remainder (byte value)
        {
            var result = new ByteTensor ();
            Remainder (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_clamp (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Clamp operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Clamp (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_clamp (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Clamp(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor Clamp (byte value)
        {
            var result = new ByteTensor ();
            Clamp (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_bitand (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitAnd operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitAnd (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_bitand (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitAnd(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor BitAnd (byte value)
        {
            var result = new ByteTensor ();
            BitAnd (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_bitor (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitOr operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitOr (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_bitor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitOr(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor BitOr (byte value)
        {
            var result = new ByteTensor ();
            BitOr (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_bitxor (HType result, HType source, byte value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitXor operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitXor (ByteTensor source, byte value, ByteTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THByteTensor_bitxor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitXor(PytorchSharp.ByteTensor, Byte, PytorchSharp.Byte)"/>.
        /// </remarks>
        public ByteTensor BitXor (byte value)
        {
            var result = new ByteTensor ();
            BitXor (this, value, result);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cadd (HType result, HType t, byte value, HType src);
        /// <summary>
        ///   Performs the CAdd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CAdd (byte value, ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cadd (result.handle, this.handle, value, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_csub (HType result, HType t, byte value, HType src);
        /// <summary>
        ///   Performs the CSub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CSub (byte value, ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_csub (result.handle, this.handle, value, src.handle);
            return result;
        }




        [DllImport ("caffe2")]
        extern static long THByteTensor_dot (HType self, HType other);
        
        /// <summary>
        ///   Returns the tensor product between this tensor and the provided one
        /// </summary>
        /// <returns>
        ///   The dot product
        /// </returns>
        public long Dot (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
           
            return THByteTensor_dot (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_match (HType result, HType m1, HType m2, byte gain);
        
        /// <summary>
        ///   
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor Match (ByteTensor m2, byte gain)
        {
            if (m2 == null)
                throw new ArgumentNullException (nameof (m2));
            var result = new ByteTensor ();
            THByteTensor_match (result.handle, this.handle, m2.handle, gain);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cmul (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMul of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CMul (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cmul (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cpow (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CPow of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CPow (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cpow (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cdiv (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CDiv of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CDiv (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cdiv (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_clshift (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CLShift of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CLShift (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_clshift (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cfmod (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CFMod of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CFMod (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cfmod (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cremainder (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CRemainder of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CRemainder (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cremainder (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cbitand (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitAnd of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CBitAnd (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cbitand (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cbitor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitOr of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CBitOr (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cbitor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cbitxor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitXor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CBitXor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cbitxor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addcmul (HType result, HType t, byte value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCMul of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddCMul (byte value, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_addcmul (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addcdiv (HType result, HType t, byte value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCDiv of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddCDiv (byte value, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_addcdiv (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addmv (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMV of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddMV (byte beta, byte alpha, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_addmv (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addmm (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddMM (byte beta, byte alpha, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_addmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addr (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddR (byte beta, byte alpha, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_addr (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addbmm (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddBMM (byte beta, byte alpha, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_addbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_baddbmm (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor BAddBMM (byte beta, byte alpha, ByteTensor src1, ByteTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ByteTensor ();
            THByteTensor_baddbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

 
        [DllImport ("caffe2")]
        extern static byte THByteTensor_minall (HType result);

        /// <summary>
        ///   Returns the minimum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The minimum value of the tensor.
        /// </returns>
        public byte MinAll ()
        {
            return THByteTensor_minall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static byte THByteTensor_maxall (HType result);

        /// <summary>
        ///   Returns the maximum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The maximum value of the tensor.
        /// </returns>
        public byte MaxAll ()
        {
            return THByteTensor_maxall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static byte THByteTensor_medianall (HType result);

        /// <summary>
        ///   Returns the median of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The median of the tensor.
        /// </returns>
        public byte MedianAll ()
        {
            return THByteTensor_medianall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THByteTensor_sumall (HType result);

        /// <summary>
        ///   Returns the sum of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The sum of the tensor.
        /// </returns>
        public long SumAll ()
        {
            return THByteTensor_sumall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THByteTensor_prodall (HType result);

        /// <summary>
        ///   Returns the product of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The product of the tensor.
        /// </returns>
        public long ProdAll ()
        {
            return THByteTensor_prodall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THByteTensor_meanall (HType result);

        /// <summary>
        ///   Returns the mean of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The mean of the tensor.
        /// </returns>
        public long MeanAll ()
        {
            return THByteTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_indexSelect (HType tensor, HType src, int dim, LongTensor.HType index);
        
        /// <summary>
        ///   Returns a new Tensor which indexes the original Tensor along dimension dim
        ///   using the entries in index.  The returned Tensor has the same number of dimensions as the 
        ///   original Tensor. The returned Tensor does not use the same storage as the original Tensor.
        /// </summary>
        /// <param name="dim">Dimension to select</param>
        /// <param name="index">Entries to extract</param>
        public ByteTensor IndexSelect (int dim, LongTensor index)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
             
            var res = new ByteTensor ();
            THByteTensor_indexSelect (res.handle, handle, dim, index.handle);
            return res;
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_indexCopy (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Copies the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexCopy (int dim, LongTensor index, ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            
            THByteTensor_indexCopy (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void Copy (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copy (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_copyByte (HType tensor, ByteTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a byte tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyByte (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyByte (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_copyShort (HType tensor, ShortTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a short tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyShort (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyShort (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_copyInt (HType tensor, IntTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a int tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyInt (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyInt (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_copyLong (HType tensor, LongTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a long tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyLong (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyLong (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_copyFloat (HType tensor, FloatTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a float tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyFloat (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyFloat (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_copyDouble (HType tensor, DoubleTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a double tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyDouble (this.handle, src.handle);
        }
        
    }
    public partial class ShortTensor : IDisposable {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        public class ShortStorage : IDisposable {
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
                extern static void THShortStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THShortStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THShortStorage_new ();
            
            /// <summary>
            ///   Initializes an empty ShortStorage instance.
            /// </summary>
            public ShortStorage ()
            {
                handle = THShortStorage_new ();
            }
            
            internal ShortStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THShortStorage_new_withSize (IntPtr size);
            
            /// <summary>
            ///   Initializes a ShortStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public ShortStorage (long size)
            {
                handle = THShortStorage_new_withSize ((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~ShortStorage ()
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
            extern static short THShortStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
            [DllImport ("caffe2")]
            extern static void THShortStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  short value);
            
            /// <summary>
            /// </summary>
            public short this [long index] {
                get => THShortStorage_get (handle, (IntPtr) (index));
                set {
                    THShortStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static short THShortStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THShortStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THShortStorage_fill (HType handle, short value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (short value)
            {
                THShortStorage_fill (handle, value);
            }
        }
    }
    
    /// <summary>
    ///   Tensor of type Short.
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use the default constructor to create an empty tensor, or invoke one of the
    ///     constructors with one (1D), two (2D), three (3D), or four parameters (4D) to x
    ///     create a tensor for the desired number of dimensions.
    ///   </para>
    /// </remarks>
    public partial class ShortTensor : IDisposable {
        internal sealed class HType : SafeHandle {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }
                
            public override bool IsInvalid => handle == (IntPtr) 0;

            [DllImport ("caffe2")]
            extern static void THShortTensor_free (IntPtr handle);
                
            protected override bool ReleaseHandle ()
            {
                THShortTensor_free (handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public ShortTensor ()
        {
            handle = THShortTensor_new ();
        }

        internal ShortTensor (HType handle)
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
        
        /// <summary>
        ///  Finalizer for ~ShortTensor
        /// </summary>
        ~ShortTensor ()
        {
            Dispose (false);
        }
        
        /// <summary>
        ///   Releases the tensor and its associated data.
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
            if (disposing) {
                handle.Dispose ();
                handle = null;
            }
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            THShortTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_fill (HType handle, short value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (short value)
        {
            THShortTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_maskedFill (HType handle1, ByteTensor.HType handle2, short value);
        
        /// <summary>
        ///  Fills the tensor with the specified value at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where the value should be filled.</param>
        /// <param name="value">The value to write at the indicated locations.</param>
        public void MaskedFill (ByteTensor mask, short value)
        {
            THShortTensor_maskedFill (handle, mask.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_maskedCopy (HType handle1, ByteTensor.HType handle2, HType src);
        
        /// <summary>
        ///  Copies elements from the source tensor to the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the destination the value should be filled.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedCopy (ByteTensor mask, ShortTensor src)
        {
            THShortTensor_maskedCopy (handle, mask.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_maskedSelect (HType handle1, HType src, ByteTensor.HType handle2);
        
        /// <summary>
        ///  Copies elements from the source tensor at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the source the value should be fetched.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There will be as many elements in the tensor as there are 1s in the mask.
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedSelect (ByteTensor mask, ShortTensor src)
        {
            THShortTensor_maskedSelect (handle, src.handle, mask.handle);
        }

        [DllImport ("caffe2")]
        extern static ShortStorage.HType THShortTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public ShortStorage Storage => new ShortStorage (THShortTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int THShortTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => THShortTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long THShortTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return THShortTensor_size (handle, dim);
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
        extern static long THShortTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return THShortTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THShortTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe short *Data => (short*) THShortTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public ShortTensor Clone () => new ShortTensor (THShortTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newSelect (HType handle, int dim, long slideIndex);
        
        /// <summary>
        ///   Returns a new Tensor which is a tensor slice at the given index in the dimension dim. 
        /// </summary>
        /// <remarks>
        ///   The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.
        /// </remarks>
        /// <param name="dim">Dimension to select</param>
        /// <param name="slideIndex">Beginning of the tensor slice</param>
        public ShortTensor Select (int dim, long slideIndex) => new ShortTensor (THShortTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THShortTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        /// <summary>
        /// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from firstIndexto firstIndex+size-1.
        /// </summary>
        /// <param name="dim">The dimension to narrow</param>
        /// <param name="firstIndex">Initial index to narrow</param>
        /// <param name="size">Number of elements</param>
        public ShortTensor Narrow (int dim, long firstIndex, long size) => new ShortTensor (THShortTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newTranspose (HType handle, int dim1, int dim2);
        
        /// <summary>
        /// Returns a tensor where dimensions dim1 and dim2 have been swapped. 
        /// </summary>
        /// <param name="dim1">First dimension</param>
        /// <param name="dim2">Second dimension</param>
        public ShortTensor Transpose (int dim1, int dim2) => new ShortTensor (THShortTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newUnfold (HType handle, int dim1, long size, long step);
        
        /// <summary>
        ///   Returns a tensor which contains all slices of size size in the dimension dim. Step between two slices is given by step.
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        public ShortTensor Unfold (int dim, long size, long step) => new ShortTensor (THShortTensor_newUnfold (handle, dim, size, step));
        
        [DllImport("caffe2")]
        extern static HType THShortTensor_newWithStorage1d(ShortStorage.HType handle, IntPtr offset, long size, long stride);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size">Size of the first dimension.</param>     
        /// <param name="stride">Stride of the first dimension.</param>          
        public ShortTensor NewWithStorage1d(IntPtr offset, long size, long stride)
        {
            return new ShortTensor(THShortTensor_newWithStorage1d(Storage.handle, offset, size, stride));
        }

        [DllImport("caffe2")]
        extern static HType THShortTensor_newWithStorage2d(ShortStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        public ShortTensor NewWithStorage2d(IntPtr offset, long size0, long stride0, long size1, long stride1)
        {
            return new ShortTensor(THShortTensor_newWithStorage2d(Storage.handle, offset, size0, stride0, size1, stride1));
        }

        [DllImport("caffe2")]
        extern static HType THShortTensor_newWithStorage3d(ShortStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        public ShortTensor NewWithStorage3d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
        {
            return new ShortTensor(THShortTensor_newWithStorage3d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2));
        }

        [DllImport("caffe2")]
        extern static HType THShortTensor_newWithStorage4d(ShortStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        /// <param name="size3">Size of the forth dimension.</param>     
        /// <param name="stride3">Stride of the forth dimension.</param>
        public ShortTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new ShortTensor(THShortTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze ()
        {
            THShortTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze1d (ShortTensor src, int dimension)
        {
            THShortTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Unsqueeze1d (ShortTensor src, int dimension)
        {
            THShortTensor_unsqueeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_resize1d (HType handle, long size);
        
        /// <summary>
        ///   Resizes the tensor to be one dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size">The desired new size for the first dimension of the tensor.</param>
        public void Resize1d (long size)
        {
            THShortTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize2d (HType handle, long size0, long size1);
        /// <summary>
        ///   Resizes the tensor to be two dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        public void Resize2d (long size0, long size1)
        {
            THShortTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        /// <summary>
        ///   Resizes the tensor to be three dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        public void Resize3d (long size0, long size1, long size2)
        {
            THShortTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        /// <summary>
        ///   Resizes the tensor to be four dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THShortTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        /// <summary>
        ///   Resizes the tensor to be five dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        /// <param name="size4">The desired new size for the fifth dimension of the tensor.</param>
        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THShortTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resizeAs (HType handle, HType src);
       
        /// <summary>
        ///   Resizes the tensor to match the dimensions of the specified src tensor, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="src">The source tensor whose shape will be mirrored by this tensor.</param>
        public void ResizeAs (ShortTensor src)
        {
            THShortTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_set (HType handle, HType src);
        
        /// <summary>
        ///   The tensor will use the same storage as the provided source, so any changes to that tensor are visible on this one.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Set (ShortTensor src)
        {
            THShortTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_set1d (HType handle, long x0, short value);
        [DllImport ("caffe2")]
        extern static short THShortTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>       
        /// <param name="x0">Index to access.</param> 
        public short this [long x0] {
            get => THShortTensor_get1d (handle, x0);
            set => THShortTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_set2d (HType handle, long x0, long x1, short value);
        [DllImport ("caffe2")]
        extern static short THShortTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>    
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        public short this [long x0, long x1] {
            get => THShortTensor_get2d (handle, x0, x1);
            set => THShortTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_set3d (HType handle, long x0, long x1, long x2, short value);
        [DllImport ("caffe2")]
        extern static short THShortTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        public short this [long x0, long x1, long x2] {
            get => THShortTensor_get3d (handle, x0, x1, x2);
            set => THShortTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_set4d (HType handle, long x0, long x1, long x2, long x3, short value);
        [DllImport ("caffe2")]
        extern static short THShortTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        /// <param name="x3">Index in the fourth dimension to access.</param>     
        public short this [long x0, long x1, long x2, long x3] {
            get => THShortTensor_get4d (handle, x0, x1, x2, x3);
            set => THShortTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static short THShortTensor_random (HType handle, IntPtr thgenerator);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Geometric (RandomGenerator source, double p)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_geometric (handle, source.handle, p);
        }
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using a newly initialized Random number geneator.
        /// </summary>
        /// <param name="n">The upper limit for the values to be generated</param>        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                CappedRandom (r, n);
        }

        
        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public override string ToString ()
        {
            var n = Dimensions;
            if (n == 0)
                    return "[]";

            StringBuilder sb = new StringBuilder ("[");
            for (int i = 0; i < n; i++) {
                    sb.Append (GetTensorDimension (i));
                    if (i + 1 < n)
                            sb.Append ("x");
            }
            sb.Append ("]");
            return sb.ToString ();
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_add (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Add operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Add operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Add (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_add (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Add operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Add(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Add (short value)
        {
            var result = new ShortTensor ();
            Add (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_sub (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Sub operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Sub operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Sub (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_sub (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Sub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Sub(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Sub (short value)
        {
            var result = new ShortTensor ();
            Sub (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_mul (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Mul operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Mul operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Mul (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_mul (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Mul operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Mul(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Mul (short value)
        {
            var result = new ShortTensor ();
            Mul (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_div (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Div operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Div operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Div (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_div (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Div operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Div(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Div (short value)
        {
            var result = new ShortTensor ();
            Div (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_lshift (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the LShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the LShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void LShift (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_lshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the LShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.LShift(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor LShift (short value)
        {
            var result = new ShortTensor ();
            LShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_rshift (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the RShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the RShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void RShift (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_rshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the RShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.RShift(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor RShift (short value)
        {
            var result = new ShortTensor ();
            RShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_fmod (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Fmod operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Fmod (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_fmod (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Fmod(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Fmod (short value)
        {
            var result = new ShortTensor ();
            Fmod (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_remainder (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Remainder operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Remainder (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_remainder (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Remainder(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Remainder (short value)
        {
            var result = new ShortTensor ();
            Remainder (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_clamp (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Clamp operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Clamp (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_clamp (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Clamp(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor Clamp (short value)
        {
            var result = new ShortTensor ();
            Clamp (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_bitand (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitAnd operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitAnd (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_bitand (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitAnd(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor BitAnd (short value)
        {
            var result = new ShortTensor ();
            BitAnd (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_bitor (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitOr operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitOr (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_bitor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitOr(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor BitOr (short value)
        {
            var result = new ShortTensor ();
            BitOr (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_bitxor (HType result, HType source, short value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitXor operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitXor (ShortTensor source, short value, ShortTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THShortTensor_bitxor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitXor(PytorchSharp.ShortTensor, Short, PytorchSharp.Short)"/>.
        /// </remarks>
        public ShortTensor BitXor (short value)
        {
            var result = new ShortTensor ();
            BitXor (this, value, result);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cadd (HType result, HType t, short value, HType src);
        /// <summary>
        ///   Performs the CAdd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CAdd (short value, ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cadd (result.handle, this.handle, value, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_csub (HType result, HType t, short value, HType src);
        /// <summary>
        ///   Performs the CSub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CSub (short value, ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_csub (result.handle, this.handle, value, src.handle);
            return result;
        }




        [DllImport ("caffe2")]
        extern static long THShortTensor_dot (HType self, HType other);
        
        /// <summary>
        ///   Returns the tensor product between this tensor and the provided one
        /// </summary>
        /// <returns>
        ///   The dot product
        /// </returns>
        public long Dot (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
           
            return THShortTensor_dot (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_match (HType result, HType m1, HType m2, short gain);
        
        /// <summary>
        ///   
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor Match (ShortTensor m2, short gain)
        {
            if (m2 == null)
                throw new ArgumentNullException (nameof (m2));
            var result = new ShortTensor ();
            THShortTensor_match (result.handle, this.handle, m2.handle, gain);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cmul (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMul of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CMul (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cmul (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cpow (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CPow of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CPow (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cpow (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cdiv (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CDiv of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CDiv (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cdiv (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_clshift (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CLShift of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CLShift (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_clshift (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cfmod (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CFMod of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CFMod (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cfmod (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cremainder (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CRemainder of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CRemainder (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cremainder (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cbitand (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitAnd of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CBitAnd (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cbitand (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cbitor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitOr of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CBitOr (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cbitor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cbitxor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitXor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CBitXor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cbitxor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addcmul (HType result, HType t, short value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCMul of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddCMul (short value, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_addcmul (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addcdiv (HType result, HType t, short value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCDiv of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddCDiv (short value, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_addcdiv (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addmv (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMV of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddMV (short beta, short alpha, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_addmv (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addmm (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddMM (short beta, short alpha, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_addmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addr (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddR (short beta, short alpha, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_addr (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addbmm (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddBMM (short beta, short alpha, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_addbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_baddbmm (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor BAddBMM (short beta, short alpha, ShortTensor src1, ShortTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new ShortTensor ();
            THShortTensor_baddbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

 
        [DllImport ("caffe2")]
        extern static short THShortTensor_minall (HType result);

        /// <summary>
        ///   Returns the minimum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The minimum value of the tensor.
        /// </returns>
        public short MinAll ()
        {
            return THShortTensor_minall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static short THShortTensor_maxall (HType result);

        /// <summary>
        ///   Returns the maximum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The maximum value of the tensor.
        /// </returns>
        public short MaxAll ()
        {
            return THShortTensor_maxall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static short THShortTensor_medianall (HType result);

        /// <summary>
        ///   Returns the median of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The median of the tensor.
        /// </returns>
        public short MedianAll ()
        {
            return THShortTensor_medianall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THShortTensor_sumall (HType result);

        /// <summary>
        ///   Returns the sum of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The sum of the tensor.
        /// </returns>
        public long SumAll ()
        {
            return THShortTensor_sumall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THShortTensor_prodall (HType result);

        /// <summary>
        ///   Returns the product of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The product of the tensor.
        /// </returns>
        public long ProdAll ()
        {
            return THShortTensor_prodall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THShortTensor_meanall (HType result);

        /// <summary>
        ///   Returns the mean of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The mean of the tensor.
        /// </returns>
        public long MeanAll ()
        {
            return THShortTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_indexSelect (HType tensor, HType src, int dim, LongTensor.HType index);
        
        /// <summary>
        ///   Returns a new Tensor which indexes the original Tensor along dimension dim
        ///   using the entries in index.  The returned Tensor has the same number of dimensions as the 
        ///   original Tensor. The returned Tensor does not use the same storage as the original Tensor.
        /// </summary>
        /// <param name="dim">Dimension to select</param>
        /// <param name="index">Entries to extract</param>
        public ShortTensor IndexSelect (int dim, LongTensor index)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
             
            var res = new ShortTensor ();
            THShortTensor_indexSelect (res.handle, handle, dim, index.handle);
            return res;
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_indexCopy (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Copies the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexCopy (int dim, LongTensor index, ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            
            THShortTensor_indexCopy (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void Copy (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copy (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_copyByte (HType tensor, ByteTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a byte tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyByte (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyByte (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_copyShort (HType tensor, ShortTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a short tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyShort (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyShort (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_copyInt (HType tensor, IntTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a int tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyInt (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyInt (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_copyLong (HType tensor, LongTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a long tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyLong (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyLong (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_copyFloat (HType tensor, FloatTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a float tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyFloat (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyFloat (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_copyDouble (HType tensor, DoubleTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a double tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyDouble (this.handle, src.handle);
        }
        
    }
    public partial class IntTensor : IDisposable {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        public class IntStorage : IDisposable {
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
                extern static void THIntStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THIntStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THIntStorage_new ();
            
            /// <summary>
            ///   Initializes an empty IntStorage instance.
            /// </summary>
            public IntStorage ()
            {
                handle = THIntStorage_new ();
            }
            
            internal IntStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THIntStorage_new_withSize (IntPtr size);
            
            /// <summary>
            ///   Initializes a IntStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public IntStorage (long size)
            {
                handle = THIntStorage_new_withSize ((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~IntStorage ()
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
            extern static int THIntStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
            [DllImport ("caffe2")]
            extern static void THIntStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  int value);
            
            /// <summary>
            /// </summary>
            public int this [long index] {
                get => THIntStorage_get (handle, (IntPtr) (index));
                set {
                    THIntStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static int THIntStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THIntStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THIntStorage_fill (HType handle, int value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (int value)
            {
                THIntStorage_fill (handle, value);
            }
        }
    }
    
    /// <summary>
    ///   Tensor of type Int.
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use the default constructor to create an empty tensor, or invoke one of the
    ///     constructors with one (1D), two (2D), three (3D), or four parameters (4D) to x
    ///     create a tensor for the desired number of dimensions.
    ///   </para>
    /// </remarks>
    public partial class IntTensor : IDisposable {
        internal sealed class HType : SafeHandle {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }
                
            public override bool IsInvalid => handle == (IntPtr) 0;

            [DllImport ("caffe2")]
            extern static void THIntTensor_free (IntPtr handle);
                
            protected override bool ReleaseHandle ()
            {
                THIntTensor_free (handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public IntTensor ()
        {
            handle = THIntTensor_new ();
        }

        internal IntTensor (HType handle)
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
        
        /// <summary>
        ///  Finalizer for ~IntTensor
        /// </summary>
        ~IntTensor ()
        {
            Dispose (false);
        }
        
        /// <summary>
        ///   Releases the tensor and its associated data.
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
            if (disposing) {
                handle.Dispose ();
                handle = null;
            }
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            THIntTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_fill (HType handle, int value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (int value)
        {
            THIntTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_maskedFill (HType handle1, ByteTensor.HType handle2, int value);
        
        /// <summary>
        ///  Fills the tensor with the specified value at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where the value should be filled.</param>
        /// <param name="value">The value to write at the indicated locations.</param>
        public void MaskedFill (ByteTensor mask, int value)
        {
            THIntTensor_maskedFill (handle, mask.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_maskedCopy (HType handle1, ByteTensor.HType handle2, HType src);
        
        /// <summary>
        ///  Copies elements from the source tensor to the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the destination the value should be filled.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedCopy (ByteTensor mask, IntTensor src)
        {
            THIntTensor_maskedCopy (handle, mask.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_maskedSelect (HType handle1, HType src, ByteTensor.HType handle2);
        
        /// <summary>
        ///  Copies elements from the source tensor at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the source the value should be fetched.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There will be as many elements in the tensor as there are 1s in the mask.
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedSelect (ByteTensor mask, IntTensor src)
        {
            THIntTensor_maskedSelect (handle, src.handle, mask.handle);
        }

        [DllImport ("caffe2")]
        extern static IntStorage.HType THIntTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public IntStorage Storage => new IntStorage (THIntTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int THIntTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => THIntTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long THIntTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return THIntTensor_size (handle, dim);
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
        extern static long THIntTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return THIntTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THIntTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe int *Data => (int*) THIntTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public IntTensor Clone () => new IntTensor (THIntTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newSelect (HType handle, int dim, long slideIndex);
        
        /// <summary>
        ///   Returns a new Tensor which is a tensor slice at the given index in the dimension dim. 
        /// </summary>
        /// <remarks>
        ///   The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.
        /// </remarks>
        /// <param name="dim">Dimension to select</param>
        /// <param name="slideIndex">Beginning of the tensor slice</param>
        public IntTensor Select (int dim, long slideIndex) => new IntTensor (THIntTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THIntTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        /// <summary>
        /// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from firstIndexto firstIndex+size-1.
        /// </summary>
        /// <param name="dim">The dimension to narrow</param>
        /// <param name="firstIndex">Initial index to narrow</param>
        /// <param name="size">Number of elements</param>
        public IntTensor Narrow (int dim, long firstIndex, long size) => new IntTensor (THIntTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newTranspose (HType handle, int dim1, int dim2);
        
        /// <summary>
        /// Returns a tensor where dimensions dim1 and dim2 have been swapped. 
        /// </summary>
        /// <param name="dim1">First dimension</param>
        /// <param name="dim2">Second dimension</param>
        public IntTensor Transpose (int dim1, int dim2) => new IntTensor (THIntTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newUnfold (HType handle, int dim1, long size, long step);
        
        /// <summary>
        ///   Returns a tensor which contains all slices of size size in the dimension dim. Step between two slices is given by step.
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        public IntTensor Unfold (int dim, long size, long step) => new IntTensor (THIntTensor_newUnfold (handle, dim, size, step));
        
        [DllImport("caffe2")]
        extern static HType THIntTensor_newWithStorage1d(IntStorage.HType handle, IntPtr offset, long size, long stride);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size">Size of the first dimension.</param>     
        /// <param name="stride">Stride of the first dimension.</param>          
        public IntTensor NewWithStorage1d(IntPtr offset, long size, long stride)
        {
            return new IntTensor(THIntTensor_newWithStorage1d(Storage.handle, offset, size, stride));
        }

        [DllImport("caffe2")]
        extern static HType THIntTensor_newWithStorage2d(IntStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        public IntTensor NewWithStorage2d(IntPtr offset, long size0, long stride0, long size1, long stride1)
        {
            return new IntTensor(THIntTensor_newWithStorage2d(Storage.handle, offset, size0, stride0, size1, stride1));
        }

        [DllImport("caffe2")]
        extern static HType THIntTensor_newWithStorage3d(IntStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        public IntTensor NewWithStorage3d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
        {
            return new IntTensor(THIntTensor_newWithStorage3d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2));
        }

        [DllImport("caffe2")]
        extern static HType THIntTensor_newWithStorage4d(IntStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        /// <param name="size3">Size of the forth dimension.</param>     
        /// <param name="stride3">Stride of the forth dimension.</param>
        public IntTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new IntTensor(THIntTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze ()
        {
            THIntTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze1d (IntTensor src, int dimension)
        {
            THIntTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Unsqueeze1d (IntTensor src, int dimension)
        {
            THIntTensor_unsqueeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_resize1d (HType handle, long size);
        
        /// <summary>
        ///   Resizes the tensor to be one dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size">The desired new size for the first dimension of the tensor.</param>
        public void Resize1d (long size)
        {
            THIntTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize2d (HType handle, long size0, long size1);
        /// <summary>
        ///   Resizes the tensor to be two dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        public void Resize2d (long size0, long size1)
        {
            THIntTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        /// <summary>
        ///   Resizes the tensor to be three dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        public void Resize3d (long size0, long size1, long size2)
        {
            THIntTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        /// <summary>
        ///   Resizes the tensor to be four dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THIntTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        /// <summary>
        ///   Resizes the tensor to be five dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        /// <param name="size4">The desired new size for the fifth dimension of the tensor.</param>
        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THIntTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resizeAs (HType handle, HType src);
       
        /// <summary>
        ///   Resizes the tensor to match the dimensions of the specified src tensor, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="src">The source tensor whose shape will be mirrored by this tensor.</param>
        public void ResizeAs (IntTensor src)
        {
            THIntTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_set (HType handle, HType src);
        
        /// <summary>
        ///   The tensor will use the same storage as the provided source, so any changes to that tensor are visible on this one.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Set (IntTensor src)
        {
            THIntTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_set1d (HType handle, long x0, int value);
        [DllImport ("caffe2")]
        extern static int THIntTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>       
        /// <param name="x0">Index to access.</param> 
        public int this [long x0] {
            get => THIntTensor_get1d (handle, x0);
            set => THIntTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_set2d (HType handle, long x0, long x1, int value);
        [DllImport ("caffe2")]
        extern static int THIntTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>    
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        public int this [long x0, long x1] {
            get => THIntTensor_get2d (handle, x0, x1);
            set => THIntTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_set3d (HType handle, long x0, long x1, long x2, int value);
        [DllImport ("caffe2")]
        extern static int THIntTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        public int this [long x0, long x1, long x2] {
            get => THIntTensor_get3d (handle, x0, x1, x2);
            set => THIntTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_set4d (HType handle, long x0, long x1, long x2, long x3, int value);
        [DllImport ("caffe2")]
        extern static int THIntTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        /// <param name="x3">Index in the fourth dimension to access.</param>     
        public int this [long x0, long x1, long x2, long x3] {
            get => THIntTensor_get4d (handle, x0, x1, x2, x3);
            set => THIntTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static int THIntTensor_random (HType handle, IntPtr thgenerator);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Geometric (RandomGenerator source, double p)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_geometric (handle, source.handle, p);
        }
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using a newly initialized Random number geneator.
        /// </summary>
        /// <param name="n">The upper limit for the values to be generated</param>        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                CappedRandom (r, n);
        }

        
        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public override string ToString ()
        {
            var n = Dimensions;
            if (n == 0)
                    return "[]";

            StringBuilder sb = new StringBuilder ("[");
            for (int i = 0; i < n; i++) {
                    sb.Append (GetTensorDimension (i));
                    if (i + 1 < n)
                            sb.Append ("x");
            }
            sb.Append ("]");
            return sb.ToString ();
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_add (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Add operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Add operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Add (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_add (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Add operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Add(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Add (int value)
        {
            var result = new IntTensor ();
            Add (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_sub (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Sub operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Sub operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Sub (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_sub (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Sub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Sub(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Sub (int value)
        {
            var result = new IntTensor ();
            Sub (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_mul (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Mul operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Mul operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Mul (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_mul (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Mul operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Mul(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Mul (int value)
        {
            var result = new IntTensor ();
            Mul (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_div (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Div operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Div operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Div (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_div (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Div operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Div(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Div (int value)
        {
            var result = new IntTensor ();
            Div (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_lshift (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the LShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the LShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void LShift (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_lshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the LShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.LShift(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor LShift (int value)
        {
            var result = new IntTensor ();
            LShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_rshift (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the RShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the RShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void RShift (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_rshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the RShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.RShift(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor RShift (int value)
        {
            var result = new IntTensor ();
            RShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_fmod (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Fmod operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Fmod (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_fmod (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Fmod(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Fmod (int value)
        {
            var result = new IntTensor ();
            Fmod (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_remainder (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Remainder operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Remainder (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_remainder (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Remainder(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Remainder (int value)
        {
            var result = new IntTensor ();
            Remainder (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_clamp (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Clamp operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Clamp (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_clamp (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Clamp(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor Clamp (int value)
        {
            var result = new IntTensor ();
            Clamp (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_bitand (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitAnd operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitAnd (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_bitand (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitAnd(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor BitAnd (int value)
        {
            var result = new IntTensor ();
            BitAnd (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_bitor (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitOr operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitOr (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_bitor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitOr(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor BitOr (int value)
        {
            var result = new IntTensor ();
            BitOr (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_bitxor (HType result, HType source, int value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitXor operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitXor (IntTensor source, int value, IntTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THIntTensor_bitxor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitXor(PytorchSharp.IntTensor, Int, PytorchSharp.Int)"/>.
        /// </remarks>
        public IntTensor BitXor (int value)
        {
            var result = new IntTensor ();
            BitXor (this, value, result);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cadd (HType result, HType t, int value, HType src);
        /// <summary>
        ///   Performs the CAdd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CAdd (int value, IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cadd (result.handle, this.handle, value, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_csub (HType result, HType t, int value, HType src);
        /// <summary>
        ///   Performs the CSub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CSub (int value, IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_csub (result.handle, this.handle, value, src.handle);
            return result;
        }




        [DllImport ("caffe2")]
        extern static long THIntTensor_dot (HType self, HType other);
        
        /// <summary>
        ///   Returns the tensor product between this tensor and the provided one
        /// </summary>
        /// <returns>
        ///   The dot product
        /// </returns>
        public long Dot (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
           
            return THIntTensor_dot (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_match (HType result, HType m1, HType m2, int gain);
        
        /// <summary>
        ///   
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor Match (IntTensor m2, int gain)
        {
            if (m2 == null)
                throw new ArgumentNullException (nameof (m2));
            var result = new IntTensor ();
            THIntTensor_match (result.handle, this.handle, m2.handle, gain);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cmul (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMul of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CMul (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cmul (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cpow (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CPow of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CPow (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cpow (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cdiv (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CDiv of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CDiv (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cdiv (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_clshift (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CLShift of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CLShift (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_clshift (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cfmod (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CFMod of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CFMod (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cfmod (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cremainder (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CRemainder of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CRemainder (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cremainder (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cbitand (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitAnd of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CBitAnd (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cbitand (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cbitor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitOr of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CBitOr (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cbitor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cbitxor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitXor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CBitXor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cbitxor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addcmul (HType result, HType t, int value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCMul of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddCMul (int value, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_addcmul (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addcdiv (HType result, HType t, int value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCDiv of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddCDiv (int value, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_addcdiv (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addmv (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMV of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddMV (int beta, int alpha, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_addmv (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addmm (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddMM (int beta, int alpha, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_addmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addr (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddR (int beta, int alpha, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_addr (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addbmm (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddBMM (int beta, int alpha, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_addbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_baddbmm (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor BAddBMM (int beta, int alpha, IntTensor src1, IntTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new IntTensor ();
            THIntTensor_baddbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

 
        [DllImport ("caffe2")]
        extern static int THIntTensor_minall (HType result);

        /// <summary>
        ///   Returns the minimum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The minimum value of the tensor.
        /// </returns>
        public int MinAll ()
        {
            return THIntTensor_minall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static int THIntTensor_maxall (HType result);

        /// <summary>
        ///   Returns the maximum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The maximum value of the tensor.
        /// </returns>
        public int MaxAll ()
        {
            return THIntTensor_maxall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static int THIntTensor_medianall (HType result);

        /// <summary>
        ///   Returns the median of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The median of the tensor.
        /// </returns>
        public int MedianAll ()
        {
            return THIntTensor_medianall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THIntTensor_sumall (HType result);

        /// <summary>
        ///   Returns the sum of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The sum of the tensor.
        /// </returns>
        public long SumAll ()
        {
            return THIntTensor_sumall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THIntTensor_prodall (HType result);

        /// <summary>
        ///   Returns the product of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The product of the tensor.
        /// </returns>
        public long ProdAll ()
        {
            return THIntTensor_prodall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THIntTensor_meanall (HType result);

        /// <summary>
        ///   Returns the mean of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The mean of the tensor.
        /// </returns>
        public long MeanAll ()
        {
            return THIntTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_indexSelect (HType tensor, HType src, int dim, LongTensor.HType index);
        
        /// <summary>
        ///   Returns a new Tensor which indexes the original Tensor along dimension dim
        ///   using the entries in index.  The returned Tensor has the same number of dimensions as the 
        ///   original Tensor. The returned Tensor does not use the same storage as the original Tensor.
        /// </summary>
        /// <param name="dim">Dimension to select</param>
        /// <param name="index">Entries to extract</param>
        public IntTensor IndexSelect (int dim, LongTensor index)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
             
            var res = new IntTensor ();
            THIntTensor_indexSelect (res.handle, handle, dim, index.handle);
            return res;
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_indexCopy (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Copies the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexCopy (int dim, LongTensor index, IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            
            THIntTensor_indexCopy (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void Copy (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copy (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_copyByte (HType tensor, ByteTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a byte tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyByte (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyByte (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_copyShort (HType tensor, ShortTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a short tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyShort (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyShort (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_copyInt (HType tensor, IntTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a int tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyInt (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyInt (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_copyLong (HType tensor, LongTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a long tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyLong (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyLong (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_copyFloat (HType tensor, FloatTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a float tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyFloat (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyFloat (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_copyDouble (HType tensor, DoubleTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a double tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyDouble (this.handle, src.handle);
        }
        
    }
    public partial class LongTensor : IDisposable {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        public class LongStorage : IDisposable {
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
                extern static void THLongStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THLongStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THLongStorage_new ();
            
            /// <summary>
            ///   Initializes an empty LongStorage instance.
            /// </summary>
            public LongStorage ()
            {
                handle = THLongStorage_new ();
            }
            
            internal LongStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THLongStorage_new_withSize (IntPtr size);
            
            /// <summary>
            ///   Initializes a LongStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public LongStorage (long size)
            {
                handle = THLongStorage_new_withSize ((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~LongStorage ()
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
            extern static long THLongStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
            [DllImport ("caffe2")]
            extern static void THLongStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  long value);
            
            /// <summary>
            /// </summary>
            public long this [long index] {
                get => THLongStorage_get (handle, (IntPtr) (index));
                set {
                    THLongStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static long THLongStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THLongStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THLongStorage_fill (HType handle, long value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (long value)
            {
                THLongStorage_fill (handle, value);
            }
        }
    }
    
    /// <summary>
    ///   Tensor of type Long.
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use the default constructor to create an empty tensor, or invoke one of the
    ///     constructors with one (1D), two (2D), three (3D), or four parameters (4D) to x
    ///     create a tensor for the desired number of dimensions.
    ///   </para>
    /// </remarks>
    public partial class LongTensor : IDisposable {
        internal sealed class HType : SafeHandle {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }
                
            public override bool IsInvalid => handle == (IntPtr) 0;

            [DllImport ("caffe2")]
            extern static void THLongTensor_free (IntPtr handle);
                
            protected override bool ReleaseHandle ()
            {
                THLongTensor_free (handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public LongTensor ()
        {
            handle = THLongTensor_new ();
        }

        internal LongTensor (HType handle)
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
        
        /// <summary>
        ///  Finalizer for ~LongTensor
        /// </summary>
        ~LongTensor ()
        {
            Dispose (false);
        }
        
        /// <summary>
        ///   Releases the tensor and its associated data.
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
            if (disposing) {
                handle.Dispose ();
                handle = null;
            }
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            THLongTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_fill (HType handle, long value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (long value)
        {
            THLongTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_maskedFill (HType handle1, ByteTensor.HType handle2, long value);
        
        /// <summary>
        ///  Fills the tensor with the specified value at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where the value should be filled.</param>
        /// <param name="value">The value to write at the indicated locations.</param>
        public void MaskedFill (ByteTensor mask, long value)
        {
            THLongTensor_maskedFill (handle, mask.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_maskedCopy (HType handle1, ByteTensor.HType handle2, HType src);
        
        /// <summary>
        ///  Copies elements from the source tensor to the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the destination the value should be filled.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedCopy (ByteTensor mask, LongTensor src)
        {
            THLongTensor_maskedCopy (handle, mask.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_maskedSelect (HType handle1, HType src, ByteTensor.HType handle2);
        
        /// <summary>
        ///  Copies elements from the source tensor at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the source the value should be fetched.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There will be as many elements in the tensor as there are 1s in the mask.
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedSelect (ByteTensor mask, LongTensor src)
        {
            THLongTensor_maskedSelect (handle, src.handle, mask.handle);
        }

        [DllImport ("caffe2")]
        extern static LongStorage.HType THLongTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public LongStorage Storage => new LongStorage (THLongTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int THLongTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => THLongTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long THLongTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return THLongTensor_size (handle, dim);
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
        extern static long THLongTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return THLongTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THLongTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe long *Data => (long*) THLongTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public LongTensor Clone () => new LongTensor (THLongTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newSelect (HType handle, int dim, long slideIndex);
        
        /// <summary>
        ///   Returns a new Tensor which is a tensor slice at the given index in the dimension dim. 
        /// </summary>
        /// <remarks>
        ///   The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.
        /// </remarks>
        /// <param name="dim">Dimension to select</param>
        /// <param name="slideIndex">Beginning of the tensor slice</param>
        public LongTensor Select (int dim, long slideIndex) => new LongTensor (THLongTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THLongTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        /// <summary>
        /// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from firstIndexto firstIndex+size-1.
        /// </summary>
        /// <param name="dim">The dimension to narrow</param>
        /// <param name="firstIndex">Initial index to narrow</param>
        /// <param name="size">Number of elements</param>
        public LongTensor Narrow (int dim, long firstIndex, long size) => new LongTensor (THLongTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newTranspose (HType handle, int dim1, int dim2);
        
        /// <summary>
        /// Returns a tensor where dimensions dim1 and dim2 have been swapped. 
        /// </summary>
        /// <param name="dim1">First dimension</param>
        /// <param name="dim2">Second dimension</param>
        public LongTensor Transpose (int dim1, int dim2) => new LongTensor (THLongTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newUnfold (HType handle, int dim1, long size, long step);
        
        /// <summary>
        ///   Returns a tensor which contains all slices of size size in the dimension dim. Step between two slices is given by step.
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        public LongTensor Unfold (int dim, long size, long step) => new LongTensor (THLongTensor_newUnfold (handle, dim, size, step));
        
        [DllImport("caffe2")]
        extern static HType THLongTensor_newWithStorage1d(LongStorage.HType handle, IntPtr offset, long size, long stride);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size">Size of the first dimension.</param>     
        /// <param name="stride">Stride of the first dimension.</param>          
        public LongTensor NewWithStorage1d(IntPtr offset, long size, long stride)
        {
            return new LongTensor(THLongTensor_newWithStorage1d(Storage.handle, offset, size, stride));
        }

        [DllImport("caffe2")]
        extern static HType THLongTensor_newWithStorage2d(LongStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        public LongTensor NewWithStorage2d(IntPtr offset, long size0, long stride0, long size1, long stride1)
        {
            return new LongTensor(THLongTensor_newWithStorage2d(Storage.handle, offset, size0, stride0, size1, stride1));
        }

        [DllImport("caffe2")]
        extern static HType THLongTensor_newWithStorage3d(LongStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        public LongTensor NewWithStorage3d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
        {
            return new LongTensor(THLongTensor_newWithStorage3d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2));
        }

        [DllImport("caffe2")]
        extern static HType THLongTensor_newWithStorage4d(LongStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        /// <param name="size3">Size of the forth dimension.</param>     
        /// <param name="stride3">Stride of the forth dimension.</param>
        public LongTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new LongTensor(THLongTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze ()
        {
            THLongTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze1d (LongTensor src, int dimension)
        {
            THLongTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Unsqueeze1d (LongTensor src, int dimension)
        {
            THLongTensor_unsqueeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_resize1d (HType handle, long size);
        
        /// <summary>
        ///   Resizes the tensor to be one dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size">The desired new size for the first dimension of the tensor.</param>
        public void Resize1d (long size)
        {
            THLongTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize2d (HType handle, long size0, long size1);
        /// <summary>
        ///   Resizes the tensor to be two dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        public void Resize2d (long size0, long size1)
        {
            THLongTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        /// <summary>
        ///   Resizes the tensor to be three dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        public void Resize3d (long size0, long size1, long size2)
        {
            THLongTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        /// <summary>
        ///   Resizes the tensor to be four dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THLongTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        /// <summary>
        ///   Resizes the tensor to be five dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        /// <param name="size4">The desired new size for the fifth dimension of the tensor.</param>
        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THLongTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resizeAs (HType handle, HType src);
       
        /// <summary>
        ///   Resizes the tensor to match the dimensions of the specified src tensor, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="src">The source tensor whose shape will be mirrored by this tensor.</param>
        public void ResizeAs (LongTensor src)
        {
            THLongTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_set (HType handle, HType src);
        
        /// <summary>
        ///   The tensor will use the same storage as the provided source, so any changes to that tensor are visible on this one.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Set (LongTensor src)
        {
            THLongTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_set1d (HType handle, long x0, long value);
        [DllImport ("caffe2")]
        extern static long THLongTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>       
        /// <param name="x0">Index to access.</param> 
        public long this [long x0] {
            get => THLongTensor_get1d (handle, x0);
            set => THLongTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_set2d (HType handle, long x0, long x1, long value);
        [DllImport ("caffe2")]
        extern static long THLongTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>    
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        public long this [long x0, long x1] {
            get => THLongTensor_get2d (handle, x0, x1);
            set => THLongTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_set3d (HType handle, long x0, long x1, long x2, long value);
        [DllImport ("caffe2")]
        extern static long THLongTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        public long this [long x0, long x1, long x2] {
            get => THLongTensor_get3d (handle, x0, x1, x2);
            set => THLongTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_set4d (HType handle, long x0, long x1, long x2, long x3, long value);
        [DllImport ("caffe2")]
        extern static long THLongTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        /// <param name="x3">Index in the fourth dimension to access.</param>     
        public long this [long x0, long x1, long x2, long x3] {
            get => THLongTensor_get4d (handle, x0, x1, x2, x3);
            set => THLongTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static long THLongTensor_random (HType handle, IntPtr thgenerator);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Geometric (RandomGenerator source, double p)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_geometric (handle, source.handle, p);
        }
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using a newly initialized Random number geneator.
        /// </summary>
        /// <param name="n">The upper limit for the values to be generated</param>        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                CappedRandom (r, n);
        }

        
        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public override string ToString ()
        {
            var n = Dimensions;
            if (n == 0)
                    return "[]";

            StringBuilder sb = new StringBuilder ("[");
            for (int i = 0; i < n; i++) {
                    sb.Append (GetTensorDimension (i));
                    if (i + 1 < n)
                            sb.Append ("x");
            }
            sb.Append ("]");
            return sb.ToString ();
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_add (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Add operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Add operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Add (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_add (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Add operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Add(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Add (long value)
        {
            var result = new LongTensor ();
            Add (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_sub (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Sub operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Sub operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Sub (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_sub (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Sub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Sub(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Sub (long value)
        {
            var result = new LongTensor ();
            Sub (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_mul (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Mul operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Mul operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Mul (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_mul (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Mul operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Mul(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Mul (long value)
        {
            var result = new LongTensor ();
            Mul (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_div (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Div operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Div operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Div (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_div (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Div operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Div(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Div (long value)
        {
            var result = new LongTensor ();
            Div (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_lshift (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the LShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the LShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void LShift (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_lshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the LShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.LShift(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor LShift (long value)
        {
            var result = new LongTensor ();
            LShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_rshift (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the RShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the RShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void RShift (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_rshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the RShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.RShift(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor RShift (long value)
        {
            var result = new LongTensor ();
            RShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_fmod (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Fmod operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Fmod (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_fmod (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Fmod(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Fmod (long value)
        {
            var result = new LongTensor ();
            Fmod (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_remainder (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Remainder operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Remainder (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_remainder (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Remainder(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Remainder (long value)
        {
            var result = new LongTensor ();
            Remainder (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_clamp (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Clamp operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Clamp (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_clamp (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Clamp(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor Clamp (long value)
        {
            var result = new LongTensor ();
            Clamp (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_bitand (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitAnd operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitAnd (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_bitand (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitAnd(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor BitAnd (long value)
        {
            var result = new LongTensor ();
            BitAnd (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_bitor (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitOr operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitOr (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_bitor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitOr operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitOr(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor BitOr (long value)
        {
            var result = new LongTensor ();
            BitOr (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_bitxor (HType result, HType source, long value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitXor operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitXor (LongTensor source, long value, LongTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THLongTensor_bitxor (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitXor operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitXor(PytorchSharp.LongTensor, Long, PytorchSharp.Long)"/>.
        /// </remarks>
        public LongTensor BitXor (long value)
        {
            var result = new LongTensor ();
            BitXor (this, value, result);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cadd (HType result, HType t, long value, HType src);
        /// <summary>
        ///   Performs the CAdd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CAdd (long value, LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cadd (result.handle, this.handle, value, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_csub (HType result, HType t, long value, HType src);
        /// <summary>
        ///   Performs the CSub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CSub (long value, LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_csub (result.handle, this.handle, value, src.handle);
            return result;
        }




        [DllImport ("caffe2")]
        extern static long THLongTensor_dot (HType self, HType other);
        
        /// <summary>
        ///   Returns the tensor product between this tensor and the provided one
        /// </summary>
        /// <returns>
        ///   The dot product
        /// </returns>
        public long Dot (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
           
            return THLongTensor_dot (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_match (HType result, HType m1, HType m2, long gain);
        
        /// <summary>
        ///   
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor Match (LongTensor m2, long gain)
        {
            if (m2 == null)
                throw new ArgumentNullException (nameof (m2));
            var result = new LongTensor ();
            THLongTensor_match (result.handle, this.handle, m2.handle, gain);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cmul (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMul of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CMul (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cmul (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cpow (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CPow of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CPow (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cpow (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cdiv (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CDiv of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CDiv (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cdiv (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_clshift (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CLShift of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CLShift (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_clshift (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cfmod (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CFMod of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CFMod (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cfmod (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cremainder (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CRemainder of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CRemainder (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cremainder (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cbitand (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitAnd of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CBitAnd (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cbitand (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cbitor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitOr of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CBitOr (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cbitor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cbitxor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitXor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CBitXor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cbitxor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addcmul (HType result, HType t, long value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCMul of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddCMul (long value, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_addcmul (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addcdiv (HType result, HType t, long value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCDiv of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddCDiv (long value, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_addcdiv (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addmv (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMV of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddMV (long beta, long alpha, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_addmv (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addmm (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddMM (long beta, long alpha, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_addmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addr (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddR (long beta, long alpha, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_addr (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addbmm (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddBMM (long beta, long alpha, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_addbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_baddbmm (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor BAddBMM (long beta, long alpha, LongTensor src1, LongTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new LongTensor ();
            THLongTensor_baddbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

 
        [DllImport ("caffe2")]
        extern static long THLongTensor_minall (HType result);

        /// <summary>
        ///   Returns the minimum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The minimum value of the tensor.
        /// </returns>
        public long MinAll ()
        {
            return THLongTensor_minall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THLongTensor_maxall (HType result);

        /// <summary>
        ///   Returns the maximum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The maximum value of the tensor.
        /// </returns>
        public long MaxAll ()
        {
            return THLongTensor_maxall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THLongTensor_medianall (HType result);

        /// <summary>
        ///   Returns the median of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The median of the tensor.
        /// </returns>
        public long MedianAll ()
        {
            return THLongTensor_medianall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THLongTensor_sumall (HType result);

        /// <summary>
        ///   Returns the sum of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The sum of the tensor.
        /// </returns>
        public long SumAll ()
        {
            return THLongTensor_sumall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THLongTensor_prodall (HType result);

        /// <summary>
        ///   Returns the product of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The product of the tensor.
        /// </returns>
        public long ProdAll ()
        {
            return THLongTensor_prodall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static long THLongTensor_meanall (HType result);

        /// <summary>
        ///   Returns the mean of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The mean of the tensor.
        /// </returns>
        public long MeanAll ()
        {
            return THLongTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_indexSelect (HType tensor, HType src, int dim, LongTensor.HType index);
        
        /// <summary>
        ///   Returns a new Tensor which indexes the original Tensor along dimension dim
        ///   using the entries in index.  The returned Tensor has the same number of dimensions as the 
        ///   original Tensor. The returned Tensor does not use the same storage as the original Tensor.
        /// </summary>
        /// <param name="dim">Dimension to select</param>
        /// <param name="index">Entries to extract</param>
        public LongTensor IndexSelect (int dim, LongTensor index)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
             
            var res = new LongTensor ();
            THLongTensor_indexSelect (res.handle, handle, dim, index.handle);
            return res;
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_indexCopy (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Copies the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexCopy (int dim, LongTensor index, LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            
            THLongTensor_indexCopy (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void Copy (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copy (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_copyByte (HType tensor, ByteTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a byte tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyByte (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyByte (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_copyShort (HType tensor, ShortTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a short tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyShort (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyShort (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_copyInt (HType tensor, IntTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a int tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyInt (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyInt (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_copyLong (HType tensor, LongTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a long tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyLong (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyLong (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_copyFloat (HType tensor, FloatTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a float tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyFloat (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyFloat (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_copyDouble (HType tensor, DoubleTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a double tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyDouble (this.handle, src.handle);
        }
        
    }
    public partial class DoubleTensor : IDisposable {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        public class DoubleStorage : IDisposable {
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
                extern static void THDoubleStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THDoubleStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THDoubleStorage_new ();
            
            /// <summary>
            ///   Initializes an empty DoubleStorage instance.
            /// </summary>
            public DoubleStorage ()
            {
                handle = THDoubleStorage_new ();
            }
            
            internal DoubleStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THDoubleStorage_new_withSize (IntPtr size);
            
            /// <summary>
            ///   Initializes a DoubleStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public DoubleStorage (long size)
            {
                handle = THDoubleStorage_new_withSize ((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~DoubleStorage ()
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
            extern static double THDoubleStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
            [DllImport ("caffe2")]
            extern static void THDoubleStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  double value);
            
            /// <summary>
            /// </summary>
            public double this [long index] {
                get => THDoubleStorage_get (handle, (IntPtr) (index));
                set {
                    THDoubleStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static double THDoubleStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THDoubleStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THDoubleStorage_fill (HType handle, double value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (double value)
            {
                THDoubleStorage_fill (handle, value);
            }
        }
    }
    
    /// <summary>
    ///   Tensor of type Double.
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use the default constructor to create an empty tensor, or invoke one of the
    ///     constructors with one (1D), two (2D), three (3D), or four parameters (4D) to x
    ///     create a tensor for the desired number of dimensions.
    ///   </para>
    /// </remarks>
    public partial class DoubleTensor : IDisposable {
        internal sealed class HType : SafeHandle {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }
                
            public override bool IsInvalid => handle == (IntPtr) 0;

            [DllImport ("caffe2")]
            extern static void THDoubleTensor_free (IntPtr handle);
                
            protected override bool ReleaseHandle ()
            {
                THDoubleTensor_free (handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public DoubleTensor ()
        {
            handle = THDoubleTensor_new ();
        }

        internal DoubleTensor (HType handle)
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
        
        /// <summary>
        ///  Finalizer for ~DoubleTensor
        /// </summary>
        ~DoubleTensor ()
        {
            Dispose (false);
        }
        
        /// <summary>
        ///   Releases the tensor and its associated data.
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
            if (disposing) {
                handle.Dispose ();
                handle = null;
            }
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            THDoubleTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_fill (HType handle, double value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (double value)
        {
            THDoubleTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_maskedFill (HType handle1, ByteTensor.HType handle2, double value);
        
        /// <summary>
        ///  Fills the tensor with the specified value at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where the value should be filled.</param>
        /// <param name="value">The value to write at the indicated locations.</param>
        public void MaskedFill (ByteTensor mask, double value)
        {
            THDoubleTensor_maskedFill (handle, mask.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_maskedCopy (HType handle1, ByteTensor.HType handle2, HType src);
        
        /// <summary>
        ///  Copies elements from the source tensor to the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the destination the value should be filled.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedCopy (ByteTensor mask, DoubleTensor src)
        {
            THDoubleTensor_maskedCopy (handle, mask.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_maskedSelect (HType handle1, HType src, ByteTensor.HType handle2);
        
        /// <summary>
        ///  Copies elements from the source tensor at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the source the value should be fetched.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There will be as many elements in the tensor as there are 1s in the mask.
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedSelect (ByteTensor mask, DoubleTensor src)
        {
            THDoubleTensor_maskedSelect (handle, src.handle, mask.handle);
        }

        [DllImport ("caffe2")]
        extern static DoubleStorage.HType THDoubleTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public DoubleStorage Storage => new DoubleStorage (THDoubleTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int THDoubleTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => THDoubleTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long THDoubleTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return THDoubleTensor_size (handle, dim);
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
        extern static long THDoubleTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return THDoubleTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THDoubleTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe double *Data => (double*) THDoubleTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public DoubleTensor Clone () => new DoubleTensor (THDoubleTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newSelect (HType handle, int dim, long slideIndex);
        
        /// <summary>
        ///   Returns a new Tensor which is a tensor slice at the given index in the dimension dim. 
        /// </summary>
        /// <remarks>
        ///   The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.
        /// </remarks>
        /// <param name="dim">Dimension to select</param>
        /// <param name="slideIndex">Beginning of the tensor slice</param>
        public DoubleTensor Select (int dim, long slideIndex) => new DoubleTensor (THDoubleTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        /// <summary>
        /// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from firstIndexto firstIndex+size-1.
        /// </summary>
        /// <param name="dim">The dimension to narrow</param>
        /// <param name="firstIndex">Initial index to narrow</param>
        /// <param name="size">Number of elements</param>
        public DoubleTensor Narrow (int dim, long firstIndex, long size) => new DoubleTensor (THDoubleTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newTranspose (HType handle, int dim1, int dim2);
        
        /// <summary>
        /// Returns a tensor where dimensions dim1 and dim2 have been swapped. 
        /// </summary>
        /// <param name="dim1">First dimension</param>
        /// <param name="dim2">Second dimension</param>
        public DoubleTensor Transpose (int dim1, int dim2) => new DoubleTensor (THDoubleTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newUnfold (HType handle, int dim1, long size, long step);
        
        /// <summary>
        ///   Returns a tensor which contains all slices of size size in the dimension dim. Step between two slices is given by step.
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        public DoubleTensor Unfold (int dim, long size, long step) => new DoubleTensor (THDoubleTensor_newUnfold (handle, dim, size, step));
        
        [DllImport("caffe2")]
        extern static HType THDoubleTensor_newWithStorage1d(DoubleStorage.HType handle, IntPtr offset, long size, long stride);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size">Size of the first dimension.</param>     
        /// <param name="stride">Stride of the first dimension.</param>          
        public DoubleTensor NewWithStorage1d(IntPtr offset, long size, long stride)
        {
            return new DoubleTensor(THDoubleTensor_newWithStorage1d(Storage.handle, offset, size, stride));
        }

        [DllImport("caffe2")]
        extern static HType THDoubleTensor_newWithStorage2d(DoubleStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        public DoubleTensor NewWithStorage2d(IntPtr offset, long size0, long stride0, long size1, long stride1)
        {
            return new DoubleTensor(THDoubleTensor_newWithStorage2d(Storage.handle, offset, size0, stride0, size1, stride1));
        }

        [DllImport("caffe2")]
        extern static HType THDoubleTensor_newWithStorage3d(DoubleStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        public DoubleTensor NewWithStorage3d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
        {
            return new DoubleTensor(THDoubleTensor_newWithStorage3d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2));
        }

        [DllImport("caffe2")]
        extern static HType THDoubleTensor_newWithStorage4d(DoubleStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        /// <param name="size3">Size of the forth dimension.</param>     
        /// <param name="stride3">Stride of the forth dimension.</param>
        public DoubleTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new DoubleTensor(THDoubleTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze ()
        {
            THDoubleTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze1d (DoubleTensor src, int dimension)
        {
            THDoubleTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Unsqueeze1d (DoubleTensor src, int dimension)
        {
            THDoubleTensor_unsqueeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize1d (HType handle, long size);
        
        /// <summary>
        ///   Resizes the tensor to be one dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size">The desired new size for the first dimension of the tensor.</param>
        public void Resize1d (long size)
        {
            THDoubleTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize2d (HType handle, long size0, long size1);
        /// <summary>
        ///   Resizes the tensor to be two dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        public void Resize2d (long size0, long size1)
        {
            THDoubleTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        /// <summary>
        ///   Resizes the tensor to be three dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        public void Resize3d (long size0, long size1, long size2)
        {
            THDoubleTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        /// <summary>
        ///   Resizes the tensor to be four dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THDoubleTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        /// <summary>
        ///   Resizes the tensor to be five dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        /// <param name="size4">The desired new size for the fifth dimension of the tensor.</param>
        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THDoubleTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resizeAs (HType handle, HType src);
       
        /// <summary>
        ///   Resizes the tensor to match the dimensions of the specified src tensor, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="src">The source tensor whose shape will be mirrored by this tensor.</param>
        public void ResizeAs (DoubleTensor src)
        {
            THDoubleTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_set (HType handle, HType src);
        
        /// <summary>
        ///   The tensor will use the same storage as the provided source, so any changes to that tensor are visible on this one.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Set (DoubleTensor src)
        {
            THDoubleTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_set1d (HType handle, long x0, double value);
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>       
        /// <param name="x0">Index to access.</param> 
        public double this [long x0] {
            get => THDoubleTensor_get1d (handle, x0);
            set => THDoubleTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_set2d (HType handle, long x0, long x1, double value);
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>    
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        public double this [long x0, long x1] {
            get => THDoubleTensor_get2d (handle, x0, x1);
            set => THDoubleTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_set3d (HType handle, long x0, long x1, long x2, double value);
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        public double this [long x0, long x1, long x2] {
            get => THDoubleTensor_get3d (handle, x0, x1, x2);
            set => THDoubleTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_set4d (HType handle, long x0, long x1, long x2, long x3, double value);
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        /// <param name="x3">Index in the fourth dimension to access.</param>     
        public double this [long x0, long x1, long x2, long x3] {
            get => THDoubleTensor_get4d (handle, x0, x1, x2, x3);
            set => THDoubleTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_random (HType handle, IntPtr thgenerator);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Geometric (RandomGenerator source, double p)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_geometric (handle, source.handle, p);
        }
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using a newly initialized Random number geneator.
        /// </summary>
        /// <param name="n">The upper limit for the values to be generated</param>        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                CappedRandom (r, n);
        }

#if false
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_bernoulli_DoubleTensor (HType self, IntPtr thgenerator, HType p);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void BernoulliTensor (RandomGenerator source, DoubleTensor p)
        {
            THDoubleTensor_bernoulli_DoubleTensor(this.handle, source.handle, p.handle);
        }
#endif

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_uniform (HType self, IntPtr thgenerator, double min, double max);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Uniform (RandomGenerator source, double min, double max)
        {
            THDoubleTensor_uniform(this.handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_exponential (HType self, IntPtr thgenerator, double lambda);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Exponential (RandomGenerator source, double lambda)
        {
            THDoubleTensor_exponential(this.handle, source.handle, lambda);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cauchy (HType self, IntPtr thgenerator, double median, double sigma);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Cauchy (RandomGenerator source, double median, double sigma)
        {
            THDoubleTensor_cauchy(this.handle, source.handle, median, sigma);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_logNormal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void LogNormal (RandomGenerator source, double mean, double stdv)
        {
            THDoubleTensor_logNormal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Normal (RandomGenerator source, double mean, double stdv)
        {
            THDoubleTensor_normal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal_means (HType self, IntPtr thgenerator, HType means, double stdv);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalMeans (RandomGenerator source, DoubleTensor means, double stdv)
        {
            THDoubleTensor_normal_means(this.handle, source.handle, means.handle, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal_stddevs (HType self, IntPtr thgenerator, double mean, HType stdvs);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalStdvs (RandomGenerator source, double mean, DoubleTensor stdvs)
        {
            THDoubleTensor_normal_stddevs(this.handle, source.handle, mean, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal_means_stddevs (HType self, IntPtr thgenerator, HType means, HType stdvs);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalMeansStdvs (RandomGenerator source, DoubleTensor means, DoubleTensor stdvs)
        {
            THDoubleTensor_normal_means_stddevs(this.handle, source.handle, means.handle, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_multinomial (HType self, IntPtr thgenerator, HType prob_dist, int n_sample, int with_replacement);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalMeansStdvs (RandomGenerator source, DoubleTensor prob_dist, int n_sample, int with_replacement)
        {
            THDoubleTensor_multinomial(this.handle, source.handle, prob_dist.handle, n_sample, with_replacement);
        }
        
        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public override string ToString ()
        {
            var n = Dimensions;
            if (n == 0)
                    return "[]";

            StringBuilder sb = new StringBuilder ("[");
            for (int i = 0; i < n; i++) {
                    sb.Append (GetTensorDimension (i));
                    if (i + 1 < n)
                            sb.Append ("x");
            }
            sb.Append ("]");
            return sb.ToString ();
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_add (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Add operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Add operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Add (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_add (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Add operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Add(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Add (double value)
        {
            var result = new DoubleTensor ();
            Add (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sub (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Sub operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Sub operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Sub (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_sub (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Sub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Sub(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Sub (double value)
        {
            var result = new DoubleTensor ();
            Sub (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_mul (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Mul operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Mul operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Mul (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_mul (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Mul operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Mul(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Mul (double value)
        {
            var result = new DoubleTensor ();
            Mul (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_div (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Div operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Div operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Div (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_div (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Div operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Div(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Div (double value)
        {
            var result = new DoubleTensor ();
            Div (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_lshift (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the LShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the LShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void LShift (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_lshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the LShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.LShift(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor LShift (double value)
        {
            var result = new DoubleTensor ();
            LShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_rshift (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the RShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the RShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void RShift (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_rshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the RShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.RShift(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor RShift (double value)
        {
            var result = new DoubleTensor ();
            RShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_fmod (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Fmod operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Fmod (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_fmod (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Fmod(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Fmod (double value)
        {
            var result = new DoubleTensor ();
            Fmod (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_remainder (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Remainder operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Remainder (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_remainder (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Remainder(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Remainder (double value)
        {
            var result = new DoubleTensor ();
            Remainder (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_clamp (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Clamp operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Clamp (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_clamp (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Clamp(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor Clamp (double value)
        {
            var result = new DoubleTensor ();
            Clamp (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_bitand (HType result, HType source, double value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitAnd operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitAnd (DoubleTensor source, double value, DoubleTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THDoubleTensor_bitand (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitAnd(PytorchSharp.DoubleTensor, Double, PytorchSharp.Double)"/>.
        /// </remarks>
        public DoubleTensor BitAnd (double value)
        {
            var result = new DoubleTensor ();
            BitAnd (this, value, result);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cadd (HType result, HType t, double value, HType src);
        /// <summary>
        ///   Performs the CAdd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CAdd (double value, DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cadd (result.handle, this.handle, value, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_csub (HType result, HType t, double value, HType src);
        /// <summary>
        ///   Performs the CSub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CSub (double value, DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_csub (result.handle, this.handle, value, src.handle);
            return result;
        }


                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sigmoid (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sigmoid of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Sigmoid (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_sigmoid (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_log (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Log (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_log (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_lgamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Lgamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Lgamma (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_lgamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_digamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Digamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Digamma (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_digamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_trigamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Trigamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Trigamma (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_trigamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_polygamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Polygamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Polygamma (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_polygamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_log10 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log10 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Log10 (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_log10 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_log1p (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log1p of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Log1p (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_log1p (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_log2 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log2 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Log2 (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_log2 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_exp (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Exp of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Exp (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_exp (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_expm1 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Expm1 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Expm1 (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_expm1 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cos (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Cos of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Cos (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cos (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_acos (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Acos of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Acos (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_acos (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cosh (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Cosh of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Cosh (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cosh (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sin (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sin of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Sin (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_sin (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_asin (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Asin of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Asin (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_asin (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sinh (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sinh of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Sinh (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_sinh (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_tan (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Tan of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Tan (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_tan (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_atan (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Atan of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Atan (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_atan (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_atan2 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Atan2 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Atan2 (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_atan2 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_tanh (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Tanh of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Tanh (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_tanh (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_erf (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Erf of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Erf (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_erf (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_erfc (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Erfc of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Erfc (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_erfc (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_erfinv (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Erfinv of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Erfinv (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_erfinv (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sqrt (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sqrt of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Sqrt (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_sqrt (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_rsqrt (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Rsqrt of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Rsqrt (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_rsqrt (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_ceil (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Ceil of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Ceil (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_ceil (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_floor (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Floor of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Floor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_floor (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_round (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Round of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Round (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_round (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_abs (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Abs of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Abs (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_abs (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_trunc (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Trunc of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Trunc (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_trunc (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_frac (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Frac of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Frac (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_frac (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cinv (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the cinv of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor cinv (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cinv (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_neg (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the neg of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor neg (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_neg (result.handle, this.handle);
            return result;
        }


                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_pow (HType result, HType x, double y);

        /// <summary>
        ///   Returns a new tensor with <see paramref="this"/> raised to the power of <see paramref="y"/>.
        /// </summary>
        /// <param 
        /// <param name="y">The exponent.</param>
        public DoubleTensor Pow (double y)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_pow (result.handle, this.handle, y);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_tpow (HType result, double x, HType y);

        /// <summary>
        ///   Returns a new tensor with <see paramref="x"/> raised to the power of <see paramref="this"/>.
        /// </summary>
        /// <param 
        /// <param name="x">The base.</param>
        public DoubleTensor TPow (double x)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_tpow (result.handle, x, this.handle);
            return result;
        }


        [DllImport ("caffe2")]
        extern static double THDoubleTensor_dot (HType self, HType other);
        
        /// <summary>
        ///   Returns the tensor product between this tensor and the provided one
        /// </summary>
        /// <returns>
        ///   The dot product
        /// </returns>
        public double Dot (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
           
            return THDoubleTensor_dot (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_match (HType result, HType m1, HType m2, double gain);
        
        /// <summary>
        ///   
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Match (DoubleTensor m2, double gain)
        {
            if (m2 == null)
                throw new ArgumentNullException (nameof (m2));
            var result = new DoubleTensor ();
            THDoubleTensor_match (result.handle, this.handle, m2.handle, gain);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cmul (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMul of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CMul (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cmul (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cpow (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CPow of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CPow (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cpow (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cdiv (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CDiv of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CDiv (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cdiv (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_clshift (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CLShift of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CLShift (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_clshift (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cfmod (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CFMod of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CFMod (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cfmod (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cremainder (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CRemainder of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CRemainder (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cremainder (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cbitand (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitAnd of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CBitAnd (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cbitand (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cbitor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitOr of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CBitOr (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cbitor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cbitxor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitXor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CBitXor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cbitxor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addcmul (HType result, HType t, double value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCMul of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddCMul (double value, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_addcmul (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addcdiv (HType result, HType t, double value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCDiv of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddCDiv (double value, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_addcdiv (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addmv (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMV of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddMV (double beta, double alpha, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_addmv (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addmm (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddMM (double beta, double alpha, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_addmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addr (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddR (double beta, double alpha, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_addr (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addbmm (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddBMM (double beta, double alpha, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_addbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_baddbmm (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor BAddBMM (double beta, double alpha, DoubleTensor src1, DoubleTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new DoubleTensor ();
            THDoubleTensor_baddbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

 
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_minall (HType result);

        /// <summary>
        ///   Returns the minimum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The minimum value of the tensor.
        /// </returns>
        public double MinAll ()
        {
            return THDoubleTensor_minall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_maxall (HType result);

        /// <summary>
        ///   Returns the maximum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The maximum value of the tensor.
        /// </returns>
        public double MaxAll ()
        {
            return THDoubleTensor_maxall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_medianall (HType result);

        /// <summary>
        ///   Returns the median of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The median of the tensor.
        /// </returns>
        public double MedianAll ()
        {
            return THDoubleTensor_medianall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_sumall (HType result);

        /// <summary>
        ///   Returns the sum of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The sum of the tensor.
        /// </returns>
        public double SumAll ()
        {
            return THDoubleTensor_sumall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_prodall (HType result);

        /// <summary>
        ///   Returns the product of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The product of the tensor.
        /// </returns>
        public double ProdAll ()
        {
            return THDoubleTensor_prodall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_meanall (HType result);

        /// <summary>
        ///   Returns the mean of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The mean of the tensor.
        /// </returns>
        public double MeanAll ()
        {
            return THDoubleTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_indexSelect (HType tensor, HType src, int dim, LongTensor.HType index);
        
        /// <summary>
        ///   Returns a new Tensor which indexes the original Tensor along dimension dim
        ///   using the entries in index.  The returned Tensor has the same number of dimensions as the 
        ///   original Tensor. The returned Tensor does not use the same storage as the original Tensor.
        /// </summary>
        /// <param name="dim">Dimension to select</param>
        /// <param name="index">Entries to extract</param>
        public DoubleTensor IndexSelect (int dim, LongTensor index)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
             
            var res = new DoubleTensor ();
            THDoubleTensor_indexSelect (res.handle, handle, dim, index.handle);
            return res;
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_indexCopy (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Copies the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexCopy (int dim, LongTensor index, DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            
            THDoubleTensor_indexCopy (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void Copy (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copy (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copyByte (HType tensor, ByteTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a byte tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyByte (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyByte (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copyShort (HType tensor, ShortTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a short tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyShort (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyShort (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copyInt (HType tensor, IntTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a int tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyInt (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyInt (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copyLong (HType tensor, LongTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a long tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyLong (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyLong (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copyFloat (HType tensor, FloatTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a float tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyFloat (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyFloat (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copyDouble (HType tensor, DoubleTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a double tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyDouble (this.handle, src.handle);
        }
        
    }
    public partial class FloatTensor : IDisposable {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        public class FloatStorage : IDisposable {
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
                extern static void THFloatStorage_free (IntPtr handle);
            
                
                protected override bool ReleaseHandle ()
                {
                    THFloatStorage_free (handle);
                    handle = IntPtr.Zero;
                    return true;
                }
            }

            internal HType handle;
            
            [DllImport ("caffe2")]
            extern static HType THFloatStorage_new ();
            
            /// <summary>
            ///   Initializes an empty FloatStorage instance.
            /// </summary>
            public FloatStorage ()
            {
                handle = THFloatStorage_new ();
            }
            
            internal FloatStorage (HType fromHandle)
            {
                this.handle = fromHandle;
            }
            
            [DllImport ("caffe2")]
            extern static HType THFloatStorage_new_withSize (IntPtr size);
            
            /// <summary>
            ///   Initializes a FloatStorage instance with the specified size.
            /// </summary>        
            /// <param name="size">The desired number of elements in the storage</param>
            public FloatStorage (long size)
            {
                handle = THFloatStorage_new_withSize ((IntPtr) size);
            }
            
            /// <summary>
            /// Finalizer
            /// </summary>
            ~FloatStorage ()
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
            extern static float THFloatStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
            [DllImport ("caffe2")]
            extern static void THFloatStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  float value);
            
            /// <summary>
            /// </summary>
            public float this [long index] {
                get => THFloatStorage_get (handle, (IntPtr) (index));
                set {
                    THFloatStorage_set (handle, (IntPtr) (index), value);
                }
            }
            
            [DllImport ("caffe2")]
            extern static float THFloatStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
            
            /// <summary>
            ///   Changes the size of this storage to the new requested size.
            /// </summary>
            /// <param name="size">The desired new size.</param>
            public void Resize (ulong size)
            {
                THFloatStorage_resize (handle, (UIntPtr) size);
            }

            [DllImport ("caffe2")]
            extern static void THFloatStorage_fill (HType handle, float value);
            
            /// <summary>
            ///   Fills every element of the storage with the specified value.
            /// </summary>
            /// <param name="value">Value used for each element</param>
            public void Fill (float value)
            {
                THFloatStorage_fill (handle, value);
            }
        }
    }
    
    /// <summary>
    ///   Tensor of type Float.
    /// </summary>
    /// <remarks>
    ///   <para>
    ///     Use the default constructor to create an empty tensor, or invoke one of the
    ///     constructors with one (1D), two (2D), three (3D), or four parameters (4D) to x
    ///     create a tensor for the desired number of dimensions.
    ///   </para>
    /// </remarks>
    public partial class FloatTensor : IDisposable {
        internal sealed class HType : SafeHandle {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }
                
            public override bool IsInvalid => handle == (IntPtr) 0;

            [DllImport ("caffe2")]
            extern static void THFloatTensor_free (IntPtr handle);
                
            protected override bool ReleaseHandle ()
            {
                THFloatTensor_free (handle);
                handle = IntPtr.Zero;
                return true;
            }
        }
        internal HType handle;
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_new ();
        
        /// <summary>
        ///    Creates an empty tensor.
        /// </summary>
        public FloatTensor ()
        {
            handle = THFloatTensor_new ();
        }

        internal FloatTensor (HType handle)
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
        
        /// <summary>
        ///  Finalizer for ~FloatTensor
        /// </summary>
        ~FloatTensor ()
        {
            Dispose (false);
        }
        
        /// <summary>
        ///   Releases the tensor and its associated data.
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
            if (disposing) {
                handle.Dispose ();
                handle = null;
            }
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_zero (HType handle);
     
        /// <summary>
        ///  Fills the tensor with zeros
        /// </summary>
        public void ZeroFill ()
        {
            THFloatTensor_zero (handle);
        }   
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_fill (HType handle, float value);
        
        /// <summary>
        ///  Fills the tensor with the specified value
        /// </summary>
        public void Fill (float value)
        {
            THFloatTensor_fill (handle, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_maskedFill (HType handle1, ByteTensor.HType handle2, float value);
        
        /// <summary>
        ///  Fills the tensor with the specified value at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where the value should be filled.</param>
        /// <param name="value">The value to write at the indicated locations.</param>
        public void MaskedFill (ByteTensor mask, float value)
        {
            THFloatTensor_maskedFill (handle, mask.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_maskedCopy (HType handle1, ByteTensor.HType handle2, HType src);
        
        /// <summary>
        ///  Copies elements from the source tensor to the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the destination the value should be filled.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedCopy (ByteTensor mask, FloatTensor src)
        {
            THFloatTensor_maskedCopy (handle, mask.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_maskedSelect (HType handle1, HType src, ByteTensor.HType handle2);
        
        /// <summary>
        ///  Copies elements from the source tensor at the locations indicated by the mask.
        /// </summary>
        /// <param name="mask">A byte tensor with values 0 or 1 indicating the locations where in the source the value should be fetched.</param>
        /// <param name="src">The source tensor.</param>
        /// <remarks>
        ///  There will be as many elements in the tensor as there are 1s in the mask.
        ///  There must be at least as many elements in the source tensor as there are 1s in the mask.
        /// </remarks>
        public void MaskedSelect (ByteTensor mask, FloatTensor src)
        {
            THFloatTensor_maskedSelect (handle, src.handle, mask.handle);
        }

        [DllImport ("caffe2")]
        extern static FloatStorage.HType THFloatTensor_storage (HType handle);

        /// <summary>
        ///  Returns the associated storage for this tensor
        /// </summary>
        
        public FloatStorage Storage => new FloatStorage (THFloatTensor_storage (handle));
        
        [DllImport ("caffe2")]
        extern static int THFloatTensor_nDimension (HType handle);
        
        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions => THFloatTensor_nDimension (handle);
        
        [DllImport ("caffe2")]
        extern static long THFloatTensor_size (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension (int dim)
        {
            return THFloatTensor_size (handle, dim);
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
        extern static long THFloatTensor_stride (HType handle, int dim);
        
        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            return THFloatTensor_stride (handle, dim);
        }
        
        [DllImport ("caffe2")]
        extern static IntPtr THFloatTensor_data (HType handle);
        
        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public unsafe float *Data => (float*) THFloatTensor_data (handle);
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newClone (HType handle);
        
        /// <summary>
        ///   Returns a deep clone of the tensor
        /// </summary>
        public FloatTensor Clone () => new FloatTensor (THFloatTensor_newClone (handle));
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newSelect (HType handle, int dim, long slideIndex);
        
        /// <summary>
        ///   Returns a new Tensor which is a tensor slice at the given index in the dimension dim. 
        /// </summary>
        /// <remarks>
        ///   The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.
        /// </remarks>
        /// <param name="dim">Dimension to select</param>
        /// <param name="slideIndex">Beginning of the tensor slice</param>
        public FloatTensor Select (int dim, long slideIndex) => new FloatTensor (THFloatTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        /// <summary>
        /// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from firstIndexto firstIndex+size-1.
        /// </summary>
        /// <param name="dim">The dimension to narrow</param>
        /// <param name="firstIndex">Initial index to narrow</param>
        /// <param name="size">Number of elements</param>
        public FloatTensor Narrow (int dim, long firstIndex, long size) => new FloatTensor (THFloatTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newTranspose (HType handle, int dim1, int dim2);
        
        /// <summary>
        /// Returns a tensor where dimensions dim1 and dim2 have been swapped. 
        /// </summary>
        /// <param name="dim1">First dimension</param>
        /// <param name="dim2">Second dimension</param>
        public FloatTensor Transpose (int dim1, int dim2) => new FloatTensor (THFloatTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newUnfold (HType handle, int dim1, long size, long step);
        
        /// <summary>
        ///   Returns a tensor which contains all slices of size size in the dimension dim. Step between two slices is given by step.
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        public FloatTensor Unfold (int dim, long size, long step) => new FloatTensor (THFloatTensor_newUnfold (handle, dim, size, step));
        
        [DllImport("caffe2")]
        extern static HType THFloatTensor_newWithStorage1d(FloatStorage.HType handle, IntPtr offset, long size, long stride);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size">Size of the first dimension.</param>     
        /// <param name="stride">Stride of the first dimension.</param>          
        public FloatTensor NewWithStorage1d(IntPtr offset, long size, long stride)
        {
            return new FloatTensor(THFloatTensor_newWithStorage1d(Storage.handle, offset, size, stride));
        }

        [DllImport("caffe2")]
        extern static HType THFloatTensor_newWithStorage2d(FloatStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        public FloatTensor NewWithStorage2d(IntPtr offset, long size0, long stride0, long size1, long stride1)
        {
            return new FloatTensor(THFloatTensor_newWithStorage2d(Storage.handle, offset, size0, stride0, size1, stride1));
        }

        [DllImport("caffe2")]
        extern static HType THFloatTensor_newWithStorage3d(FloatStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        public FloatTensor NewWithStorage3d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2)
        {
            return new FloatTensor(THFloatTensor_newWithStorage3d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2));
        }

        [DllImport("caffe2")]
        extern static HType THFloatTensor_newWithStorage4d(FloatStorage.HType handle, IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="offset">Offset within the input storage the storage of the new tensor will start from.</param> 
        /// <param name="size0">Size of the first dimension.</param>     
        /// <param name="stride0">Stride of the first dimension.</param>
        /// <param name="size1">Size of the second dimension.</param>     
        /// <param name="stride1">Stride of the second dimension.</param>
        /// <param name="size2">Size of the third dimension.</param>     
        /// <param name="stride2">Stride of the third dimension.</param>
        /// <param name="size3">Size of the forth dimension.</param>     
        /// <param name="stride3">Stride of the forth dimension.</param>
        public FloatTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new FloatTensor(THFloatTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze ()
        {
            THFloatTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Squeeze1d (FloatTensor src, int dimension)
        {
            THFloatTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Unsqueeze1d (FloatTensor src, int dimension)
        {
            THFloatTensor_unsqueeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize1d (HType handle, long size);
        
        /// <summary>
        ///   Resizes the tensor to be one dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size">The desired new size for the first dimension of the tensor.</param>
        public void Resize1d (long size)
        {
            THFloatTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize2d (HType handle, long size0, long size1);
        /// <summary>
        ///   Resizes the tensor to be two dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        public void Resize2d (long size0, long size1)
        {
            THFloatTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        /// <summary>
        ///   Resizes the tensor to be three dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        public void Resize3d (long size0, long size1, long size2)
        {
            THFloatTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        /// <summary>
        ///   Resizes the tensor to be four dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THFloatTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        /// <summary>
        ///   Resizes the tensor to be five dimensional with the specified size, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="size0">The desired new size for the first dimension of the tensor.</param>
        /// <param name="size1">The desired new size for the second dimension of the tensor.</param>
        /// <param name="size2">The desired new size for the third dimension of the tensor.</param>
        /// <param name="size3">The desired new size for the fourth dimension of the tensor.</param>
        /// <param name="size4">The desired new size for the fifth dimension of the tensor.</param>
        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THFloatTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resizeAs (HType handle, HType src);
       
        /// <summary>
        ///   Resizes the tensor to match the dimensions of the specified src tensor, the contents of the tensor after this are undetermined.
        /// </summary>
        /// <param name="src">The source tensor whose shape will be mirrored by this tensor.</param>
        public void ResizeAs (FloatTensor src)
        {
            THFloatTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_set (HType handle, HType src);
        
        /// <summary>
        ///   The tensor will use the same storage as the provided source, so any changes to that tensor are visible on this one.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data..</param>
        public void Set (FloatTensor src)
        {
            THFloatTensor_set (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_set1d (HType handle, long x0, float value);
        [DllImport ("caffe2")]
        extern static float THFloatTensor_get1d (HType handle, long x0);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>       
        /// <param name="x0">Index to access.</param> 
        public float this [long x0] {
            get => THFloatTensor_get1d (handle, x0);
            set => THFloatTensor_set1d (handle, x0, value);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_set2d (HType handle, long x0, long x1, float value);
        [DllImport ("caffe2")]
        extern static float THFloatTensor_get2d (HType handle, long x0, long x1);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>    
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        public float this [long x0, long x1] {
            get => THFloatTensor_get2d (handle, x0, x1);
            set => THFloatTensor_set2d (handle, x0, x1, value);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_set3d (HType handle, long x0, long x1, long x2, float value);
        [DllImport ("caffe2")]
        extern static float THFloatTensor_get3d (HType handle, long x0, long x1, long x2);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        public float this [long x0, long x1, long x2] {
            get => THFloatTensor_get3d (handle, x0, x1, x2);
            set => THFloatTensor_set3d (handle, x0, x1, x2, value);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_set4d (HType handle, long x0, long x1, long x2, long x3, float value);
        [DllImport ("caffe2")]
        extern static float THFloatTensor_get4d (HType handle, long x0, long x1, long x2, long x3);

        /// <summary>
        ///   Access to element at the specified position in the tensor
        /// </summary>        
        /// <param name="x0">Index in the first dimension to access.</param> 
        /// <param name="x1">Index in the second dimension to access.</param>     
        /// <param name="x2">Index in the third dimension to access.</param>     
        /// <param name="x3">Index in the fourth dimension to access.</param>     
        public float this [long x0, long x1, long x2, long x3] {
            get => THFloatTensor_get4d (handle, x0, x1, x2, x3);
            set => THFloatTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static float THFloatTensor_random (HType handle, IntPtr thgenerator);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="n">The upper limit for the values to be generated</param>
        public void Geometric (RandomGenerator source, double p)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_geometric (handle, source.handle, p);
        }
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using a newly initialized Random number geneator.
        /// </summary>
        /// <param name="n">The upper limit for the values to be generated</param>        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                CappedRandom (r, n);
        }

#if false
        [DllImport ("caffe2")]
        extern static void THFloatTensor_bernoulli_FloatTensor (HType self, IntPtr thgenerator, HType p);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void BernoulliTensor (RandomGenerator source, FloatTensor p)
        {
            THFloatTensor_bernoulli_FloatTensor(this.handle, source.handle, p.handle);
        }
#endif

        [DllImport ("caffe2")]
        extern static void THFloatTensor_uniform (HType self, IntPtr thgenerator, double min, double max);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Uniform (RandomGenerator source, double min, double max)
        {
            THFloatTensor_uniform(this.handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_exponential (HType self, IntPtr thgenerator, double lambda);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Exponential (RandomGenerator source, double lambda)
        {
            THFloatTensor_exponential(this.handle, source.handle, lambda);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_cauchy (HType self, IntPtr thgenerator, double median, double sigma);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Cauchy (RandomGenerator source, double median, double sigma)
        {
            THFloatTensor_cauchy(this.handle, source.handle, median, sigma);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_logNormal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void LogNormal (RandomGenerator source, double mean, double stdv)
        {
            THFloatTensor_logNormal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void Normal (RandomGenerator source, double mean, double stdv)
        {
            THFloatTensor_normal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal_means (HType self, IntPtr thgenerator, HType means, double stdv);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalMeans (RandomGenerator source, FloatTensor means, double stdv)
        {
            THFloatTensor_normal_means(this.handle, source.handle, means.handle, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal_stddevs (HType self, IntPtr thgenerator, double mean, HType stdvs);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalStdvs (RandomGenerator source, double mean, FloatTensor stdvs)
        {
            THFloatTensor_normal_stddevs(this.handle, source.handle, mean, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal_means_stddevs (HType self, IntPtr thgenerator, HType means, HType stdvs);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalMeansStdvs (RandomGenerator source, FloatTensor means, FloatTensor stdvs)
        {
            THFloatTensor_normal_means_stddevs(this.handle, source.handle, means.handle, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_multinomial (HType self, IntPtr thgenerator, HType prob_dist, int n_sample, int with_replacement);

        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public void NormalMeansStdvs (RandomGenerator source, FloatTensor prob_dist, int n_sample, int with_replacement)
        {
            THFloatTensor_multinomial(this.handle, source.handle, prob_dist.handle, n_sample, with_replacement);
        }
        
        /// <summary>
        ///   Returns a debuggable version of the tensor, in this case the tensor shape
        /// </summary>
        public override string ToString ()
        {
            var n = Dimensions;
            if (n == 0)
                    return "[]";

            StringBuilder sb = new StringBuilder ("[");
            for (int i = 0; i < n; i++) {
                    sb.Append (GetTensorDimension (i));
                    if (i + 1 < n)
                            sb.Append ("x");
            }
            sb.Append ("]");
            return sb.ToString ();
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_add (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Add operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Add operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Add (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_add (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Add operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Add(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Add (float value)
        {
            var result = new FloatTensor ();
            Add (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sub (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Sub operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Sub operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Sub (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_sub (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Sub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Sub(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Sub (float value)
        {
            var result = new FloatTensor ();
            Sub (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_mul (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Mul operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Mul operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Mul (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_mul (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Mul operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Mul(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Mul (float value)
        {
            var result = new FloatTensor ();
            Mul (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_div (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Div operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Div operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Div (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_div (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Div operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Div(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Div (float value)
        {
            var result = new FloatTensor ();
            Div (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_lshift (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the LShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the LShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void LShift (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_lshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the LShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.LShift(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor LShift (float value)
        {
            var result = new FloatTensor ();
            LShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_rshift (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the RShift operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the RShift operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void RShift (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_rshift (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the RShift operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.RShift(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor RShift (float value)
        {
            var result = new FloatTensor ();
            RShift (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_fmod (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Fmod operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Fmod (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_fmod (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Fmod operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Fmod(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Fmod (float value)
        {
            var result = new FloatTensor ();
            Fmod (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_remainder (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Remainder operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Remainder (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_remainder (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Remainder operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Remainder(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Remainder (float value)
        {
            var result = new FloatTensor ();
            Remainder (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_clamp (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the Clamp operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void Clamp (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_clamp (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the Clamp operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.Clamp(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor Clamp (float value)
        {
            var result = new FloatTensor ();
            Clamp (this, value, result);
            return result;
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_bitand (HType result, HType source, float value);
        
        // Not married to xthis yet - we have a few ways of solving this, sometimes
        // we could avoid allocation, but the API is ugly.  Or we could not have side-effects
        // which can also be surprising
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the source with the
        ///   provided scalar.   The result tensor specified as the last parameters
        ///   is resized to match the source.
        /// </summary>
        /// <remarks>
        ///    For each element in the <see paramref="source"/> performs the BitAnd operation
        ///    with <see paramref="value"/>.   The result is stored in the <see paramref="result"/>
        ///    tensor.
        /// </remarks>
        /// <param name="source">Source tensor on which the operation will take place.</param>
        /// <param name="value">The scalar value that the operation uses.</param>
        /// <param name="result">The tensor where the result will be placed</param>
        public static void BitAnd (FloatTensor source, float value, FloatTensor result)
        {
            // Arguments swapped to match Func<.., TResult> 
            THFloatTensor_bitand (result.handle, source.handle, value);
        }
        
        /// <summary>
        ///   Performs the BitAnd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        /// <remarks>
        ///   If you want to avoid the allocation of a new tensor, you can use the 
        ///   alternative method <see cref="M:PytorchSharp.BitAnd(PytorchSharp.FloatTensor, Float, PytorchSharp.Float)"/>.
        /// </remarks>
        public FloatTensor BitAnd (float value)
        {
            var result = new FloatTensor ();
            BitAnd (this, value, result);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cadd (HType result, HType t, float value, HType src);
        /// <summary>
        ///   Performs the CAdd operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CAdd (float value, FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cadd (result.handle, this.handle, value, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_csub (HType result, HType t, float value, HType src);
        /// <summary>
        ///   Performs the CSub operation on each element of the tensor with the
        ///   <see paramref="value"/> and returns a new tensor with the result.
        ///   where the result is t[idx] + value * src[idx]
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CSub (float value, FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_csub (result.handle, this.handle, value, src.handle);
            return result;
        }


                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sigmoid (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sigmoid of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Sigmoid (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_sigmoid (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_log (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Log (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_log (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_lgamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Lgamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Lgamma (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_lgamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_digamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Digamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Digamma (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_digamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_trigamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Trigamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Trigamma (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_trigamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_polygamma (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Polygamma of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Polygamma (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_polygamma (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_log10 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log10 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Log10 (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_log10 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_log1p (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log1p of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Log1p (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_log1p (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_log2 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Log2 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Log2 (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_log2 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_exp (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Exp of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Exp (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_exp (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_expm1 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Expm1 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Expm1 (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_expm1 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cos (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Cos of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Cos (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cos (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_acos (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Acos of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Acos (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_acos (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cosh (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Cosh of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Cosh (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cosh (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sin (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sin of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Sin (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_sin (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_asin (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Asin of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Asin (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_asin (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sinh (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sinh of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Sinh (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_sinh (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_tan (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Tan of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Tan (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_tan (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_atan (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Atan of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Atan (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_atan (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_atan2 (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Atan2 of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Atan2 (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_atan2 (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_tanh (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Tanh of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Tanh (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_tanh (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_erf (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Erf of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Erf (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_erf (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_erfc (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Erfc of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Erfc (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_erfc (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_erfinv (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Erfinv of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Erfinv (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_erfinv (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sqrt (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Sqrt of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Sqrt (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_sqrt (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_rsqrt (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Rsqrt of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Rsqrt (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_rsqrt (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_ceil (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Ceil of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Ceil (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_ceil (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_floor (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Floor of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Floor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_floor (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_round (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Round of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Round (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_round (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_abs (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Abs of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Abs (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_abs (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_trunc (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Trunc of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Trunc (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_trunc (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_frac (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the Frac of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Frac (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_frac (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cinv (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the cinv of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor cinv (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cinv (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_neg (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the neg of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor neg (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_neg (result.handle, this.handle);
            return result;
        }


                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_pow (HType result, HType x, float y);

        /// <summary>
        ///   Returns a new tensor with <see paramref="this"/> raised to the power of <see paramref="y"/>.
        /// </summary>
        /// <param 
        /// <param name="y">The exponent.</param>
        public FloatTensor Pow (float y)
        {
            var result = new FloatTensor ();
            THFloatTensor_pow (result.handle, this.handle, y);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_tpow (HType result, float x, HType y);

        /// <summary>
        ///   Returns a new tensor with <see paramref="x"/> raised to the power of <see paramref="this"/>.
        /// </summary>
        /// <param 
        /// <param name="x">The base.</param>
        public FloatTensor TPow (float x)
        {
            var result = new FloatTensor ();
            THFloatTensor_tpow (result.handle, x, this.handle);
            return result;
        }


        [DllImport ("caffe2")]
        extern static double THFloatTensor_dot (HType self, HType other);
        
        /// <summary>
        ///   Returns the tensor product between this tensor and the provided one
        /// </summary>
        /// <returns>
        ///   The dot product
        /// </returns>
        public double Dot (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
           
            return THFloatTensor_dot (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_match (HType result, HType m1, HType m2, float gain);
        
        /// <summary>
        ///   
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Match (FloatTensor m2, float gain)
        {
            if (m2 == null)
                throw new ArgumentNullException (nameof (m2));
            var result = new FloatTensor ();
            THFloatTensor_match (result.handle, this.handle, m2.handle, gain);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cmul (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMul of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CMul (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cmul (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cpow (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CPow of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CPow (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cpow (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cdiv (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CDiv of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CDiv (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cdiv (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_clshift (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CLShift of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CLShift (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_clshift (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cfmod (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CFMod of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CFMod (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cfmod (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cremainder (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CRemainder of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CRemainder (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cremainder (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cbitand (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitAnd of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CBitAnd (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cbitand (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cbitor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitOr of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CBitOr (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cbitor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cbitxor (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CBitXor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CBitXor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cbitxor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addcmul (HType result, HType t, float value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCMul of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddCMul (float value, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_addcmul (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addcdiv (HType result, HType t, float value, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddCDiv of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddCDiv (float value, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_addcdiv (result.handle, this.handle, value, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addmv (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMV of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddMV (float beta, float alpha, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_addmv (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addmm (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddMM (float beta, float alpha, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_addmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addr (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddR (float beta, float alpha, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_addr (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addbmm (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddBMM (float beta, float alpha, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_addbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_baddbmm (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor BAddBMM (float beta, float alpha, FloatTensor src1, FloatTensor src2)
        {
            if (src1 == null)
                throw new ArgumentNullException (nameof (src1));
            if (src2 == null)
                throw new ArgumentNullException (nameof (src2));
            var result = new FloatTensor ();
            THFloatTensor_baddbmm (result.handle, beta, this.handle, alpha, src1.handle, src2.handle);
            return result;
        }

 
        [DllImport ("caffe2")]
        extern static float THFloatTensor_minall (HType result);

        /// <summary>
        ///   Returns the minimum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The minimum value of the tensor.
        /// </returns>
        public float MinAll ()
        {
            return THFloatTensor_minall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static float THFloatTensor_maxall (HType result);

        /// <summary>
        ///   Returns the maximum value of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The maximum value of the tensor.
        /// </returns>
        public float MaxAll ()
        {
            return THFloatTensor_maxall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static float THFloatTensor_medianall (HType result);

        /// <summary>
        ///   Returns the median of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The median of the tensor.
        /// </returns>
        public float MedianAll ()
        {
            return THFloatTensor_medianall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THFloatTensor_sumall (HType result);

        /// <summary>
        ///   Returns the sum of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The sum of the tensor.
        /// </returns>
        public double SumAll ()
        {
            return THFloatTensor_sumall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THFloatTensor_prodall (HType result);

        /// <summary>
        ///   Returns the product of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The product of the tensor.
        /// </returns>
        public double ProdAll ()
        {
            return THFloatTensor_prodall (this.handle);
        }
 
        [DllImport ("caffe2")]
        extern static double THFloatTensor_meanall (HType result);

        /// <summary>
        ///   Returns the mean of the elements in the tensor.
        /// </summary>
        /// <returns>
        ///   The mean of the tensor.
        /// </returns>
        public double MeanAll ()
        {
            return THFloatTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_indexSelect (HType tensor, HType src, int dim, LongTensor.HType index);
        
        /// <summary>
        ///   Returns a new Tensor which indexes the original Tensor along dimension dim
        ///   using the entries in index.  The returned Tensor has the same number of dimensions as the 
        ///   original Tensor. The returned Tensor does not use the same storage as the original Tensor.
        /// </summary>
        /// <param name="dim">Dimension to select</param>
        /// <param name="index">Entries to extract</param>
        public FloatTensor IndexSelect (int dim, LongTensor index)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
             
            var res = new FloatTensor ();
            THFloatTensor_indexSelect (res.handle, handle, dim, index.handle);
            return res;
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_indexCopy (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Copies the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexCopy (int dim, LongTensor index, FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            
            THFloatTensor_indexCopy (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void Copy (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copy (this.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_copyByte (HType tensor, ByteTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a byte tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyByte (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyByte (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_copyShort (HType tensor, ShortTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a short tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyShort (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyShort (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_copyInt (HType tensor, IntTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a int tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyInt (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyInt (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_copyLong (HType tensor, LongTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a long tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyLong (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyLong (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_copyFloat (HType tensor, FloatTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a float tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyFloat (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyFloat (this.handle, src.handle);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_copyDouble (HType tensor, DoubleTensor.HType src);
        
        /// <summary>
        ///   Copies the elements of a double tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the copy</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyDouble (this.handle, src.handle);
        }
        
    }
}