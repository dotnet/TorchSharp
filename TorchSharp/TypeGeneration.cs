using System;
using System.Linq;
using System.Collections.Generic;
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
        extern static long THByteTensor_numel (HType handle);
     
        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumElements ()
        {
            return THByteTensor_numel (handle);
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
        extern static void THByteTensor_nonzero (LongTensor.HType subscript, HType handle);
     
        /// <summary>
        ///  Finds the indices of all non-zero elements.
        /// </summary>
        public LongTensor NonZero ()
        {
            var result = new LongTensor();
            THByteTensor_nonzero (result.handle, this.handle);
            return result;
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
        /// <param name="size3">Size of the fourth dimension.</param>     
        /// <param name="stride3">Stride of the fourth dimension.</param>
        public ByteTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new ByteTensor(THByteTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        public void Squeeze ()
        {
            THByteTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to remove.</param>
        public void Squeeze1d (ByteTensor src, int dimension)
        {
            THByteTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to insert.</param>
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
        ///  Populates the tensor with random values using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from min to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower limit for the values to be generated</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_randperm (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void RandPerm (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_randperm (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static byte THByteTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
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
        ///   Get a string representation of the tensor.
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
        /// <param name="src">The right-hand-side operand.</param>
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
        ///   Match
        /// </summary>
        /// <param name="m2"></param>
        /// <param name="gain"></param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        extern static void THByteTensor_cmax (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMax of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CMax (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cmax (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cmin (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMin of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CMin (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_cmin (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_ltTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_ltTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_leTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_leTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_gtTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_gtTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_geTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_geTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_eqTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_eqTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_neTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensor (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_neTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_ltTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensorT (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_ltTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_leTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensorT (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_leTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_gtTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensorT (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_gtTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_geTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensorT (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_geTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_eqTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensorT (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_eqTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_neTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensorT (ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THByteTensor_neTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cmaxvalue (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an CMaxValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CMaxValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_cmaxvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_cminvalue (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an CMinValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor CMinValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_cminvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_ltValue (ByteTensor.HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an LtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_ltValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_leValue (ByteTensor.HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an LeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_leValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_gtValue (ByteTensor.HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an GtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_gtValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_geValue (ByteTensor.HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an GeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_geValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_eqValue (ByteTensor.HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an EqValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_eqValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_neValue (ByteTensor.HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an NeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValue (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_neValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_ltValueT (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an LtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValueT (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_ltValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_leValueT (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an LeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValueT (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_leValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_gtValueT (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an GtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValueT (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_gtValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_geValueT (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an GeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValueT (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_geValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_eqValueT (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an EqValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValueT (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_eqValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_neValueT (HType result, HType t, byte value);
        
        /// <summary>
        ///   Performs an NeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValueT (byte src)
        {
            var result = new ByteTensor ();
            THByteTensor_neValueT (result.handle, this.handle, src);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_lerp (HType result, HType self, HType other, byte weight);
        
        /// <summary>
        ///   LERP
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        /// <param name="weight"></param>
        public ByteTensor LERP (ByteTensor other, byte weight)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            var result = new ByteTensor();
            THByteTensor_lerp (result.handle, this.handle, other.handle, weight);
            return result;
        }

        [DllImport ("caffe2")]
        extern static int THByteTensor_equal (HType t, HType src);
        
        /// <summary>
        ///   Compare the tensor with another for complete equality.
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        public int Equal (ByteTensor other)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THByteTensor_equal (this.handle, other.handle);
        }
                
        [DllImport ("caffe2")]
        extern static void THByteTensor_add_scaled (HType result, HType t, byte value1, byte value2);
        
        /// <summary>
        ///   Performs an AddScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor AddScaled (byte value1, byte value2)
        {
            var result = new ByteTensor ();
            THByteTensor_add_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_sub_scaled (HType result, HType t, byte value1, byte value2);
        
        /// <summary>
        ///   Performs an SubScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor SubScaled (byte value1, byte value2)
        {
            var result = new ByteTensor ();
            THByteTensor_sub_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_clamp (HType result, HType t, byte value1, byte value2);
        
        /// <summary>
        ///   Performs an Clamp of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor Clamp (byte value1, byte value2)
        {
            var result = new ByteTensor ();
            THByteTensor_clamp (result.handle, this.handle, value1, value2);
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
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for matxvec (α)</param>
        /// <param name="mat">Matrix to be multiplied</param>
        /// <param name="vec">Vector to be multiplied</param>
        /// <remarks>
        /// β tensor+α (mat@vec)

        /// </remarks>
        /// <returns>
        ///   β tensor+α (mat@vec)
        /// </returns>
        public ByteTensor AddMV (byte beta, byte alpha, ByteTensor mat, ByteTensor vec)
        {
            if (mat == null)
                throw new ArgumentNullException (nameof (mat));
            if (vec == null)
                throw new ArgumentNullException (nameof (vec));
            var result = new ByteTensor ();
            THByteTensor_addmv (result.handle, beta, this.handle, alpha, mat.handle, vec.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addmm (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for mat1xmat2 (α)</param>
        /// <param name="mat1">First matrix to  be multiplied</param>
        /// <param name="mat2">Second matrix to  be multiplied</param>
        /// <remarks>
        /// β mat+α (mat1i@mat2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (mat1i@mat2i)
        /// </returns>
        public ByteTensor AddMM (byte beta, byte alpha, ByteTensor mat1, ByteTensor mat2)
        {
            if (mat1 == null)
                throw new ArgumentNullException (nameof (mat1));
            if (mat2 == null)
                throw new ArgumentNullException (nameof (mat2));
            var result = new ByteTensor ();
            THByteTensor_addmm (result.handle, beta, this.handle, alpha, mat1.handle, mat2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addbmm (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mat+α (∑i=0bbatch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (∑i=0bbatch1i@batch2i)
        /// </returns>
        public ByteTensor AddBMM (byte beta, byte alpha, ByteTensor batch1, ByteTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new ByteTensor ();
            THByteTensor_addbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_addr (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for vec1xvec2 (α)</param>
        /// <param name="vec1">the first vector of the outer product</param>
        /// <param name="vec2">the second vector of the outer product</param>
        /// <remarks>
        /// β mat+α (vec1⊗vec2)

        /// </remarks>
        /// <returns>
        ///   β mat+α (vec1⊗vec2)
        /// </returns>
        public ByteTensor AddR (byte beta, byte alpha, ByteTensor vec1, ByteTensor vec2)
        {
            if (vec1 == null)
                throw new ArgumentNullException (nameof (vec1));
            if (vec2 == null)
                throw new ArgumentNullException (nameof (vec2));
            var result = new ByteTensor ();
            THByteTensor_addr (result.handle, beta, this.handle, alpha, vec1.handle, vec2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THByteTensor_baddbmm (HType result, byte beta, HType t, byte alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mati+α (batch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mati+α (batch1i@batch2i)
        /// </returns>
        public ByteTensor BAddBMM (byte beta, byte alpha, ByteTensor batch1, ByteTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new ByteTensor ();
            THByteTensor_baddbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
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
        extern static void THByteTensor_indexAdd (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Adds the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the add</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexAdd (int dim, LongTensor index, ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_indexAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_indexFill (HType tensor, int dim, LongTensor.HType index, byte value);
        
        /// <summary>
        ///   Uses the given value to overwrite the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the fill</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="value">The value to write.</param>
        public void IndexFill (int dim, LongTensor index, byte value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_indexFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_take (HType self, HType src, LongTensor.HType index);

        /// <summary>
        ///   Take
        /// </summary>        
        /// <param name="src"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Take (ByteTensor src, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_take (handle, src.handle, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_put (HType self, LongTensor.HType index, HType src, int accumulate);

        /// <summary>
        ///   Put
        /// </summary>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        /// <param name="accumulate"></param>
        public void Put (LongTensor index, ByteTensor src, int accumulate)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_put (handle, index.handle, src.handle, accumulate);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_gather (HType self, HType src, int dim, LongTensor.HType index);

        /// <summary>
        ///   Gather
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Gather (ByteTensor src, int dim, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_gather (handle, src.handle, dim, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_scatter (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   Scatter
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void Scatter (int dim, LongTensor index, ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_scatter (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_scatterAdd (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void ScatterAdd (int dim, LongTensor index, ByteTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_scatterAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_scatterFill (HType self, int dim, LongTensor.HType index, byte value);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="value"></param>
        public void ScatterFill (int dim, LongTensor index, byte value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THByteTensor_scatterFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
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
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THByteTensor_copyDouble (this.handle, src.handle);
        }

        
        [DllImport ("caffe2")]
        extern static int THByteTensor_logicalAndAll (HType self);
        
        /// <summary>
        ///   Compares all the elements of the tensor using 'AND' and returns the result as an integer, either 0 or 1. 
        /// </summary>
        public int LogicalAndAll ()
        {
            return THByteTensor_logicalAndAll (this.handle);
        }
        [DllImport ("caffe2")]
        extern static int THByteTensor_logicalAnyAll (HType self);
        
        /// <summary>
        ///   Compares all the elements of the tensor using 'OR' and returns the result as an integer, either 0 or 1. 
        /// </summary>
        public int LogicalAnyAll ()
        {
            return THByteTensor_logicalAnyAll (this.handle);
        }
        
        [DllImport ("caffe2")]
        extern static int THByteTensor_logicalAnd (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Compares all the elements of the tensor using 'AND' and returns the result as an integer, either 0 or 1. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public ByteTensor LogicalAnd (int dimension, int keepdim)
        {
            var result = new ByteTensor ();
            THByteTensor_logicalAnd (result.handle, this.handle, dimension, keepdim);
            return result;
        }
        [DllImport ("caffe2")]
        extern static int THByteTensor_logicalAny (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Compares all the elements of the tensor using 'OR' and returns the result as an integer, either 0 or 1. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public ByteTensor LogicalAny (int dimension, int keepdim)
        {
            var result = new ByteTensor ();
            THByteTensor_logicalAny (result.handle, this.handle, dimension, keepdim);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THByteTensor_sum (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public ByteTensor Sum (int dimension, int keepdim)
        {
            var result = new ByteTensor ();
            THByteTensor_sum (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_cumsum (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public ByteTensor CumulativeSum (int dimension)
        {
            var result = new ByteTensor ();
            THByteTensor_cumsum (result.handle, this.handle, dimension);
            return result;
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_prod (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public ByteTensor Prod (int dimension, int keepdim)
        {
            var result = new ByteTensor ();
            THByteTensor_prod (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_cumprod (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public ByteTensor CumulativeProd (int dimension)
        {
            var result = new ByteTensor ();
            THByteTensor_cumprod (result.handle, this.handle, dimension);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THByteTensor_max (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the max of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ByteTensor, LongTensor> Max (int dimension, int keepdim)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_max (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_min (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the min of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ByteTensor, LongTensor> Min (int dimension, int keepdim)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_min (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_mode (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the mode of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ByteTensor, LongTensor> Mode (int dimension, int keepdim)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_mode (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THByteTensor_median (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the median of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ByteTensor, LongTensor> Median (int dimension, int keepdim)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_median (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }
     

        [DllImport ("caffe2")]
        extern static void THByteTensor_kthvalue (HType values, LongTensor.HType indices, HType self, long k, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the kth value of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The value for 'k' in 'kth'.</param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the kth element of each dimension.</returns>
        public System.Tuple<ByteTensor, LongTensor> KthValue (long k, int dimension, int keepdim)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_kthvalue (values.handle, indices.handle, this.handle, k, dimension, keepdim);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static long THByteTensor_trace (HType self);
        
        /// <summary>
        ///   Computes the trace of the tensor. 
        /// </summary>
        public long Trace ()
        {
            return THByteTensor_trace(this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_sign (HType result, HType self);
        
        /// <summary>
        ///   Computes the sign of the tensor. 
        /// </summary>
        public ByteTensor Sign ()
        {
            var result = new ByteTensor();
            THByteTensor_sign(result.handle, this.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_cross (HType result, HType a, HType b);
        
        /// <summary>
        ///   Computes the cross product of two tensors. 
        /// </summary>
        /// <param name="other">The right-hand-side tensor.</param>
        public ByteTensor CrossProduct (ByteTensor other)
        {
            var result = new ByteTensor();
            THByteTensor_cross(result.handle, this.handle, other.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_diag (HType result, HType self, int k);
        
        /// <summary>
        ///   Gets the diagonal of the tensor. 
        /// </summary>
        /// <param name="k"></param>
        public ByteTensor Diagonal (int k)
        {
            var result = new ByteTensor();
            THByteTensor_diag(result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_eye (HType result, long m, long n);
        
        /// <summary>
        ///   Eye. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static ByteTensor Eye (long m, long n)
        {
            var result = new ByteTensor();
            THByteTensor_eye(result.handle, m, n);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_range (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static ByteTensor Range (long xmin, long xmax, long step)
        {
            var result = new ByteTensor();
            THByteTensor_range(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_arange (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static ByteTensor ARange (long xmin, long xmax, long step)
        {
            var result = new ByteTensor();
            THByteTensor_arange(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_sort (HType values, LongTensor.HType indices, HType self, int dimension, int descending);
        
        /// <summary>
        ///   Sorts the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to sort along.</param>
        /// <param name="descending">0 if ascending, 1 if descending.</param>
        /// <returns>A tuple containing the values and indices of the sorted elements.</returns>
        public System.Tuple<ByteTensor, LongTensor> Sort (int dimension, int descending)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_sort (values.handle, indices.handle, this.handle, dimension, descending);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_topk (HType values, LongTensor.HType indices, HType self, long k, int dim, int dir, int sorted);
        
        /// <summary>
        ///   Finds the top k of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The number of elements to fetch.</param>
        /// <param name="dim">The dimension along which to sort and find k elements.</param>
        /// <param name="dir">0 if ascending, 1 if descending.</param>
        /// <param name="sorted">1 if the result should be sorted, 0 if they should keep their original order.</param>
        /// <returns>A tuple containing the values and indices of the top 'k' elements.</returns>
        public System.Tuple<ByteTensor, LongTensor> TopK (long k, int dim, int dir, int sorted)
        {
            var values = new ByteTensor ();
            var indices = new LongTensor ();
            THByteTensor_topk (values.handle, indices.handle, this.handle, k, dim, dir, sorted);
            return new System.Tuple<ByteTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_tril (HType result, HType self, long k);
        
        /// <summary>
        ///   Lower triangle. 
        /// </summary>
        /// <param name="k"></param>
        public ByteTensor TriL (long k)
        {
            var result = new ByteTensor ();
            THByteTensor_tril (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_triu (HType result, HType self, long k);
        
        /// <summary>
        ///   Upper triangle. 
        /// </summary>
        /// <param name="k"></param>
        public ByteTensor TriU (long k)
        {
            var result = new ByteTensor ();
            THByteTensor_triu (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_cat (HType result, HType ta, HType tb, int dimension);
        
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="other">The second tensor.</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public ByteTensor Concatenate (ByteTensor other, int dimension)
        {
            var result = new ByteTensor ();
            THByteTensor_cat (result.handle, this.handle, other.handle, dimension);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_catArray (HType result, HType[] ta, int count, int dimension);
#if false        
// NOTE: We need to determine the right marshalling for an array of handles.
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="tensors">A collection of tensors..</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public static ByteTensor Concatenate (IEnumerable<ByteTensor> tensors, int dimension)
        {
            var result = new ByteTensor ();
            var handleArray = tensors.Select(t => t.handle).ToArray();
            THByteTensor_catArray (result.handle, handleArray, (int)handleArray.Length, dimension);
            return result;
        }
#endif
     
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
        extern static long THShortTensor_numel (HType handle);
     
        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumElements ()
        {
            return THShortTensor_numel (handle);
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
        extern static void THShortTensor_nonzero (LongTensor.HType subscript, HType handle);
     
        /// <summary>
        ///  Finds the indices of all non-zero elements.
        /// </summary>
        public LongTensor NonZero ()
        {
            var result = new LongTensor();
            THShortTensor_nonzero (result.handle, this.handle);
            return result;
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
        /// <param name="size3">Size of the fourth dimension.</param>     
        /// <param name="stride3">Stride of the fourth dimension.</param>
        public ShortTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new ShortTensor(THShortTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        public void Squeeze ()
        {
            THShortTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to remove.</param>
        public void Squeeze1d (ShortTensor src, int dimension)
        {
            THShortTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to insert.</param>
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
        ///  Populates the tensor with random values using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from min to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower limit for the values to be generated</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_randperm (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void RandPerm (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_randperm (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static short THShortTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
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
        ///   Get a string representation of the tensor.
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
        /// <param name="src">The right-hand-side operand.</param>
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
        ///   Match
        /// </summary>
        /// <param name="m2"></param>
        /// <param name="gain"></param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        extern static void THShortTensor_cmax (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMax of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CMax (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cmax (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cmin (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMin of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CMin (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_cmin (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_ltTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THShortTensor_ltTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_leTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THShortTensor_leTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_gtTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THShortTensor_gtTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_geTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THShortTensor_geTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_eqTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THShortTensor_eqTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_neTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensor (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THShortTensor_neTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_ltTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor LtTensorT (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_ltTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_leTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor LeTensorT (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_leTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_gtTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor GtTensorT (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_gtTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_geTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor GeTensorT (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_geTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_eqTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor EqTensorT (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_eqTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_neTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor NeTensorT (ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ShortTensor ();
            THShortTensor_neTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cmaxvalue (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an CMaxValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CMaxValue (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_cmaxvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_cminvalue (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an CMinValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor CMinValue (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_cminvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_ltValue (ByteTensor.HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an LtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValue (short src)
        {
            var result = new ByteTensor ();
            THShortTensor_ltValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_leValue (ByteTensor.HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an LeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValue (short src)
        {
            var result = new ByteTensor ();
            THShortTensor_leValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_gtValue (ByteTensor.HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an GtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValue (short src)
        {
            var result = new ByteTensor ();
            THShortTensor_gtValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_geValue (ByteTensor.HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an GeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValue (short src)
        {
            var result = new ByteTensor ();
            THShortTensor_geValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_eqValue (ByteTensor.HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an EqValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValue (short src)
        {
            var result = new ByteTensor ();
            THShortTensor_eqValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_neValue (ByteTensor.HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an NeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValue (short src)
        {
            var result = new ByteTensor ();
            THShortTensor_neValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_ltValueT (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an LtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor LtValueT (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_ltValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_leValueT (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an LeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor LeValueT (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_leValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_gtValueT (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an GtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor GtValueT (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_gtValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_geValueT (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an GeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor GeValueT (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_geValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_eqValueT (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an EqValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor EqValueT (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_eqValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_neValueT (HType result, HType t, short value);
        
        /// <summary>
        ///   Performs an NeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor NeValueT (short src)
        {
            var result = new ShortTensor ();
            THShortTensor_neValueT (result.handle, this.handle, src);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_lerp (HType result, HType self, HType other, short weight);
        
        /// <summary>
        ///   LERP
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        /// <param name="weight"></param>
        public ShortTensor LERP (ShortTensor other, short weight)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            var result = new ShortTensor();
            THShortTensor_lerp (result.handle, this.handle, other.handle, weight);
            return result;
        }

        [DllImport ("caffe2")]
        extern static int THShortTensor_equal (HType t, HType src);
        
        /// <summary>
        ///   Compare the tensor with another for complete equality.
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        public int Equal (ShortTensor other)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THShortTensor_equal (this.handle, other.handle);
        }
                
        [DllImport ("caffe2")]
        extern static void THShortTensor_add_scaled (HType result, HType t, short value1, short value2);
        
        /// <summary>
        ///   Performs an AddScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor AddScaled (short value1, short value2)
        {
            var result = new ShortTensor ();
            THShortTensor_add_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_sub_scaled (HType result, HType t, short value1, short value2);
        
        /// <summary>
        ///   Performs an SubScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor SubScaled (short value1, short value2)
        {
            var result = new ShortTensor ();
            THShortTensor_sub_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_clamp (HType result, HType t, short value1, short value2);
        
        /// <summary>
        ///   Performs an Clamp of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ShortTensor Clamp (short value1, short value2)
        {
            var result = new ShortTensor ();
            THShortTensor_clamp (result.handle, this.handle, value1, value2);
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
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for matxvec (α)</param>
        /// <param name="mat">Matrix to be multiplied</param>
        /// <param name="vec">Vector to be multiplied</param>
        /// <remarks>
        /// β tensor+α (mat@vec)

        /// </remarks>
        /// <returns>
        ///   β tensor+α (mat@vec)
        /// </returns>
        public ShortTensor AddMV (short beta, short alpha, ShortTensor mat, ShortTensor vec)
        {
            if (mat == null)
                throw new ArgumentNullException (nameof (mat));
            if (vec == null)
                throw new ArgumentNullException (nameof (vec));
            var result = new ShortTensor ();
            THShortTensor_addmv (result.handle, beta, this.handle, alpha, mat.handle, vec.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addmm (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for mat1xmat2 (α)</param>
        /// <param name="mat1">First matrix to  be multiplied</param>
        /// <param name="mat2">Second matrix to  be multiplied</param>
        /// <remarks>
        /// β mat+α (mat1i@mat2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (mat1i@mat2i)
        /// </returns>
        public ShortTensor AddMM (short beta, short alpha, ShortTensor mat1, ShortTensor mat2)
        {
            if (mat1 == null)
                throw new ArgumentNullException (nameof (mat1));
            if (mat2 == null)
                throw new ArgumentNullException (nameof (mat2));
            var result = new ShortTensor ();
            THShortTensor_addmm (result.handle, beta, this.handle, alpha, mat1.handle, mat2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addbmm (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mat+α (∑i=0bbatch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (∑i=0bbatch1i@batch2i)
        /// </returns>
        public ShortTensor AddBMM (short beta, short alpha, ShortTensor batch1, ShortTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new ShortTensor ();
            THShortTensor_addbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_addr (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for vec1xvec2 (α)</param>
        /// <param name="vec1">the first vector of the outer product</param>
        /// <param name="vec2">the second vector of the outer product</param>
        /// <remarks>
        /// β mat+α (vec1⊗vec2)

        /// </remarks>
        /// <returns>
        ///   β mat+α (vec1⊗vec2)
        /// </returns>
        public ShortTensor AddR (short beta, short alpha, ShortTensor vec1, ShortTensor vec2)
        {
            if (vec1 == null)
                throw new ArgumentNullException (nameof (vec1));
            if (vec2 == null)
                throw new ArgumentNullException (nameof (vec2));
            var result = new ShortTensor ();
            THShortTensor_addr (result.handle, beta, this.handle, alpha, vec1.handle, vec2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THShortTensor_baddbmm (HType result, short beta, HType t, short alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mati+α (batch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mati+α (batch1i@batch2i)
        /// </returns>
        public ShortTensor BAddBMM (short beta, short alpha, ShortTensor batch1, ShortTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new ShortTensor ();
            THShortTensor_baddbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
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
        extern static void THShortTensor_indexAdd (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Adds the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the add</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexAdd (int dim, LongTensor index, ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_indexAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_indexFill (HType tensor, int dim, LongTensor.HType index, short value);
        
        /// <summary>
        ///   Uses the given value to overwrite the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the fill</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="value">The value to write.</param>
        public void IndexFill (int dim, LongTensor index, short value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_indexFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_take (HType self, HType src, LongTensor.HType index);

        /// <summary>
        ///   Take
        /// </summary>        
        /// <param name="src"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Take (ShortTensor src, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_take (handle, src.handle, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_put (HType self, LongTensor.HType index, HType src, int accumulate);

        /// <summary>
        ///   Put
        /// </summary>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        /// <param name="accumulate"></param>
        public void Put (LongTensor index, ShortTensor src, int accumulate)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_put (handle, index.handle, src.handle, accumulate);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_gather (HType self, HType src, int dim, LongTensor.HType index);

        /// <summary>
        ///   Gather
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Gather (ShortTensor src, int dim, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_gather (handle, src.handle, dim, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_scatter (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   Scatter
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void Scatter (int dim, LongTensor index, ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_scatter (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_scatterAdd (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void ScatterAdd (int dim, LongTensor index, ShortTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_scatterAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_scatterFill (HType self, int dim, LongTensor.HType index, short value);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="value"></param>
        public void ScatterFill (int dim, LongTensor index, short value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THShortTensor_scatterFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
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
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THShortTensor_copyDouble (this.handle, src.handle);
        }

        
        
     
        [DllImport ("caffe2")]
        extern static void THShortTensor_sum (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public ShortTensor Sum (int dimension, int keepdim)
        {
            var result = new ShortTensor ();
            THShortTensor_sum (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_cumsum (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public ShortTensor CumulativeSum (int dimension)
        {
            var result = new ShortTensor ();
            THShortTensor_cumsum (result.handle, this.handle, dimension);
            return result;
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_prod (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public ShortTensor Prod (int dimension, int keepdim)
        {
            var result = new ShortTensor ();
            THShortTensor_prod (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_cumprod (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public ShortTensor CumulativeProd (int dimension)
        {
            var result = new ShortTensor ();
            THShortTensor_cumprod (result.handle, this.handle, dimension);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THShortTensor_max (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the max of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ShortTensor, LongTensor> Max (int dimension, int keepdim)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_max (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_min (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the min of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ShortTensor, LongTensor> Min (int dimension, int keepdim)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_min (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_mode (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the mode of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ShortTensor, LongTensor> Mode (int dimension, int keepdim)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_mode (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THShortTensor_median (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the median of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<ShortTensor, LongTensor> Median (int dimension, int keepdim)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_median (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }
     

        [DllImport ("caffe2")]
        extern static void THShortTensor_kthvalue (HType values, LongTensor.HType indices, HType self, long k, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the kth value of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The value for 'k' in 'kth'.</param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the kth element of each dimension.</returns>
        public System.Tuple<ShortTensor, LongTensor> KthValue (long k, int dimension, int keepdim)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_kthvalue (values.handle, indices.handle, this.handle, k, dimension, keepdim);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static long THShortTensor_trace (HType self);
        
        /// <summary>
        ///   Computes the trace of the tensor. 
        /// </summary>
        public long Trace ()
        {
            return THShortTensor_trace(this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_sign (HType result, HType self);
        
        /// <summary>
        ///   Computes the sign of the tensor. 
        /// </summary>
        public ShortTensor Sign ()
        {
            var result = new ShortTensor();
            THShortTensor_sign(result.handle, this.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_cross (HType result, HType a, HType b);
        
        /// <summary>
        ///   Computes the cross product of two tensors. 
        /// </summary>
        /// <param name="other">The right-hand-side tensor.</param>
        public ShortTensor CrossProduct (ShortTensor other)
        {
            var result = new ShortTensor();
            THShortTensor_cross(result.handle, this.handle, other.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_diag (HType result, HType self, int k);
        
        /// <summary>
        ///   Gets the diagonal of the tensor. 
        /// </summary>
        /// <param name="k"></param>
        public ShortTensor Diagonal (int k)
        {
            var result = new ShortTensor();
            THShortTensor_diag(result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_eye (HType result, long m, long n);
        
        /// <summary>
        ///   Eye. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static ShortTensor Eye (long m, long n)
        {
            var result = new ShortTensor();
            THShortTensor_eye(result.handle, m, n);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_range (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static ShortTensor Range (long xmin, long xmax, long step)
        {
            var result = new ShortTensor();
            THShortTensor_range(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_arange (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static ShortTensor ARange (long xmin, long xmax, long step)
        {
            var result = new ShortTensor();
            THShortTensor_arange(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_sort (HType values, LongTensor.HType indices, HType self, int dimension, int descending);
        
        /// <summary>
        ///   Sorts the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to sort along.</param>
        /// <param name="descending">0 if ascending, 1 if descending.</param>
        /// <returns>A tuple containing the values and indices of the sorted elements.</returns>
        public System.Tuple<ShortTensor, LongTensor> Sort (int dimension, int descending)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_sort (values.handle, indices.handle, this.handle, dimension, descending);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_topk (HType values, LongTensor.HType indices, HType self, long k, int dim, int dir, int sorted);
        
        /// <summary>
        ///   Finds the top k of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The number of elements to fetch.</param>
        /// <param name="dim">The dimension along which to sort and find k elements.</param>
        /// <param name="dir">0 if ascending, 1 if descending.</param>
        /// <param name="sorted">1 if the result should be sorted, 0 if they should keep their original order.</param>
        /// <returns>A tuple containing the values and indices of the top 'k' elements.</returns>
        public System.Tuple<ShortTensor, LongTensor> TopK (long k, int dim, int dir, int sorted)
        {
            var values = new ShortTensor ();
            var indices = new LongTensor ();
            THShortTensor_topk (values.handle, indices.handle, this.handle, k, dim, dir, sorted);
            return new System.Tuple<ShortTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_tril (HType result, HType self, long k);
        
        /// <summary>
        ///   Lower triangle. 
        /// </summary>
        /// <param name="k"></param>
        public ShortTensor TriL (long k)
        {
            var result = new ShortTensor ();
            THShortTensor_tril (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_triu (HType result, HType self, long k);
        
        /// <summary>
        ///   Upper triangle. 
        /// </summary>
        /// <param name="k"></param>
        public ShortTensor TriU (long k)
        {
            var result = new ShortTensor ();
            THShortTensor_triu (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_cat (HType result, HType ta, HType tb, int dimension);
        
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="other">The second tensor.</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public ShortTensor Concatenate (ShortTensor other, int dimension)
        {
            var result = new ShortTensor ();
            THShortTensor_cat (result.handle, this.handle, other.handle, dimension);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_catArray (HType result, HType[] ta, int count, int dimension);
#if false        
// NOTE: We need to determine the right marshalling for an array of handles.
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="tensors">A collection of tensors..</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public static ShortTensor Concatenate (IEnumerable<ShortTensor> tensors, int dimension)
        {
            var result = new ShortTensor ();
            var handleArray = tensors.Select(t => t.handle).ToArray();
            THShortTensor_catArray (result.handle, handleArray, (int)handleArray.Length, dimension);
            return result;
        }
#endif
     
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
        extern static long THIntTensor_numel (HType handle);
     
        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumElements ()
        {
            return THIntTensor_numel (handle);
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
        extern static void THIntTensor_nonzero (LongTensor.HType subscript, HType handle);
     
        /// <summary>
        ///  Finds the indices of all non-zero elements.
        /// </summary>
        public LongTensor NonZero ()
        {
            var result = new LongTensor();
            THIntTensor_nonzero (result.handle, this.handle);
            return result;
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
        /// <param name="size3">Size of the fourth dimension.</param>     
        /// <param name="stride3">Stride of the fourth dimension.</param>
        public IntTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new IntTensor(THIntTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        public void Squeeze ()
        {
            THIntTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to remove.</param>
        public void Squeeze1d (IntTensor src, int dimension)
        {
            THIntTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to insert.</param>
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
        ///  Populates the tensor with random values using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from min to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower limit for the values to be generated</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_randperm (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void RandPerm (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_randperm (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
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
        ///   Get a string representation of the tensor.
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
        /// <param name="src">The right-hand-side operand.</param>
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
        ///   Match
        /// </summary>
        /// <param name="m2"></param>
        /// <param name="gain"></param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        extern static void THIntTensor_cmax (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMax of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CMax (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cmax (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cmin (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMin of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CMin (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_cmin (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_ltTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THIntTensor_ltTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_leTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THIntTensor_leTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_gtTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THIntTensor_gtTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_geTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THIntTensor_geTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_eqTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THIntTensor_eqTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_neTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensor (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THIntTensor_neTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_ltTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor LtTensorT (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_ltTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_leTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor LeTensorT (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_leTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_gtTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor GtTensorT (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_gtTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_geTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor GeTensorT (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_geTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_eqTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor EqTensorT (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_eqTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_neTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor NeTensorT (IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new IntTensor ();
            THIntTensor_neTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cmaxvalue (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an CMaxValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CMaxValue (int src)
        {
            var result = new IntTensor ();
            THIntTensor_cmaxvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_cminvalue (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an CMinValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor CMinValue (int src)
        {
            var result = new IntTensor ();
            THIntTensor_cminvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_ltValue (ByteTensor.HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an LtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValue (int src)
        {
            var result = new ByteTensor ();
            THIntTensor_ltValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_leValue (ByteTensor.HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an LeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValue (int src)
        {
            var result = new ByteTensor ();
            THIntTensor_leValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_gtValue (ByteTensor.HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an GtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValue (int src)
        {
            var result = new ByteTensor ();
            THIntTensor_gtValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_geValue (ByteTensor.HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an GeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValue (int src)
        {
            var result = new ByteTensor ();
            THIntTensor_geValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_eqValue (ByteTensor.HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an EqValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValue (int src)
        {
            var result = new ByteTensor ();
            THIntTensor_eqValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_neValue (ByteTensor.HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an NeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValue (int src)
        {
            var result = new ByteTensor ();
            THIntTensor_neValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_ltValueT (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an LtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor LtValueT (int src)
        {
            var result = new IntTensor ();
            THIntTensor_ltValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_leValueT (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an LeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor LeValueT (int src)
        {
            var result = new IntTensor ();
            THIntTensor_leValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_gtValueT (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an GtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor GtValueT (int src)
        {
            var result = new IntTensor ();
            THIntTensor_gtValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_geValueT (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an GeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor GeValueT (int src)
        {
            var result = new IntTensor ();
            THIntTensor_geValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_eqValueT (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an EqValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor EqValueT (int src)
        {
            var result = new IntTensor ();
            THIntTensor_eqValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_neValueT (HType result, HType t, int value);
        
        /// <summary>
        ///   Performs an NeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor NeValueT (int src)
        {
            var result = new IntTensor ();
            THIntTensor_neValueT (result.handle, this.handle, src);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_lerp (HType result, HType self, HType other, int weight);
        
        /// <summary>
        ///   LERP
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        /// <param name="weight"></param>
        public IntTensor LERP (IntTensor other, int weight)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            var result = new IntTensor();
            THIntTensor_lerp (result.handle, this.handle, other.handle, weight);
            return result;
        }

        [DllImport ("caffe2")]
        extern static int THIntTensor_equal (HType t, HType src);
        
        /// <summary>
        ///   Compare the tensor with another for complete equality.
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        public int Equal (IntTensor other)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THIntTensor_equal (this.handle, other.handle);
        }
                
        [DllImport ("caffe2")]
        extern static void THIntTensor_add_scaled (HType result, HType t, int value1, int value2);
        
        /// <summary>
        ///   Performs an AddScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor AddScaled (int value1, int value2)
        {
            var result = new IntTensor ();
            THIntTensor_add_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_sub_scaled (HType result, HType t, int value1, int value2);
        
        /// <summary>
        ///   Performs an SubScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor SubScaled (int value1, int value2)
        {
            var result = new IntTensor ();
            THIntTensor_sub_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_clamp (HType result, HType t, int value1, int value2);
        
        /// <summary>
        ///   Performs an Clamp of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public IntTensor Clamp (int value1, int value2)
        {
            var result = new IntTensor ();
            THIntTensor_clamp (result.handle, this.handle, value1, value2);
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
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for matxvec (α)</param>
        /// <param name="mat">Matrix to be multiplied</param>
        /// <param name="vec">Vector to be multiplied</param>
        /// <remarks>
        /// β tensor+α (mat@vec)

        /// </remarks>
        /// <returns>
        ///   β tensor+α (mat@vec)
        /// </returns>
        public IntTensor AddMV (int beta, int alpha, IntTensor mat, IntTensor vec)
        {
            if (mat == null)
                throw new ArgumentNullException (nameof (mat));
            if (vec == null)
                throw new ArgumentNullException (nameof (vec));
            var result = new IntTensor ();
            THIntTensor_addmv (result.handle, beta, this.handle, alpha, mat.handle, vec.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addmm (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for mat1xmat2 (α)</param>
        /// <param name="mat1">First matrix to  be multiplied</param>
        /// <param name="mat2">Second matrix to  be multiplied</param>
        /// <remarks>
        /// β mat+α (mat1i@mat2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (mat1i@mat2i)
        /// </returns>
        public IntTensor AddMM (int beta, int alpha, IntTensor mat1, IntTensor mat2)
        {
            if (mat1 == null)
                throw new ArgumentNullException (nameof (mat1));
            if (mat2 == null)
                throw new ArgumentNullException (nameof (mat2));
            var result = new IntTensor ();
            THIntTensor_addmm (result.handle, beta, this.handle, alpha, mat1.handle, mat2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addbmm (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mat+α (∑i=0bbatch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (∑i=0bbatch1i@batch2i)
        /// </returns>
        public IntTensor AddBMM (int beta, int alpha, IntTensor batch1, IntTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new IntTensor ();
            THIntTensor_addbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_addr (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for vec1xvec2 (α)</param>
        /// <param name="vec1">the first vector of the outer product</param>
        /// <param name="vec2">the second vector of the outer product</param>
        /// <remarks>
        /// β mat+α (vec1⊗vec2)

        /// </remarks>
        /// <returns>
        ///   β mat+α (vec1⊗vec2)
        /// </returns>
        public IntTensor AddR (int beta, int alpha, IntTensor vec1, IntTensor vec2)
        {
            if (vec1 == null)
                throw new ArgumentNullException (nameof (vec1));
            if (vec2 == null)
                throw new ArgumentNullException (nameof (vec2));
            var result = new IntTensor ();
            THIntTensor_addr (result.handle, beta, this.handle, alpha, vec1.handle, vec2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THIntTensor_baddbmm (HType result, int beta, HType t, int alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mati+α (batch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mati+α (batch1i@batch2i)
        /// </returns>
        public IntTensor BAddBMM (int beta, int alpha, IntTensor batch1, IntTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new IntTensor ();
            THIntTensor_baddbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
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
        extern static void THIntTensor_indexAdd (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Adds the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the add</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexAdd (int dim, LongTensor index, IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_indexAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_indexFill (HType tensor, int dim, LongTensor.HType index, int value);
        
        /// <summary>
        ///   Uses the given value to overwrite the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the fill</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="value">The value to write.</param>
        public void IndexFill (int dim, LongTensor index, int value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_indexFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_take (HType self, HType src, LongTensor.HType index);

        /// <summary>
        ///   Take
        /// </summary>        
        /// <param name="src"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Take (IntTensor src, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_take (handle, src.handle, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_put (HType self, LongTensor.HType index, HType src, int accumulate);

        /// <summary>
        ///   Put
        /// </summary>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        /// <param name="accumulate"></param>
        public void Put (LongTensor index, IntTensor src, int accumulate)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_put (handle, index.handle, src.handle, accumulate);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_gather (HType self, HType src, int dim, LongTensor.HType index);

        /// <summary>
        ///   Gather
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Gather (IntTensor src, int dim, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_gather (handle, src.handle, dim, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_scatter (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   Scatter
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void Scatter (int dim, LongTensor index, IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_scatter (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_scatterAdd (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void ScatterAdd (int dim, LongTensor index, IntTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_scatterAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_scatterFill (HType self, int dim, LongTensor.HType index, int value);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="value"></param>
        public void ScatterFill (int dim, LongTensor index, int value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THIntTensor_scatterFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
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
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THIntTensor_copyDouble (this.handle, src.handle);
        }

        
        
     
        [DllImport ("caffe2")]
        extern static void THIntTensor_sum (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public IntTensor Sum (int dimension, int keepdim)
        {
            var result = new IntTensor ();
            THIntTensor_sum (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_cumsum (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public IntTensor CumulativeSum (int dimension)
        {
            var result = new IntTensor ();
            THIntTensor_cumsum (result.handle, this.handle, dimension);
            return result;
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_prod (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public IntTensor Prod (int dimension, int keepdim)
        {
            var result = new IntTensor ();
            THIntTensor_prod (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_cumprod (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public IntTensor CumulativeProd (int dimension)
        {
            var result = new IntTensor ();
            THIntTensor_cumprod (result.handle, this.handle, dimension);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THIntTensor_max (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the max of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<IntTensor, LongTensor> Max (int dimension, int keepdim)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_max (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_min (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the min of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<IntTensor, LongTensor> Min (int dimension, int keepdim)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_min (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_mode (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the mode of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<IntTensor, LongTensor> Mode (int dimension, int keepdim)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_mode (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THIntTensor_median (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the median of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<IntTensor, LongTensor> Median (int dimension, int keepdim)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_median (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }
     

        [DllImport ("caffe2")]
        extern static void THIntTensor_kthvalue (HType values, LongTensor.HType indices, HType self, long k, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the kth value of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The value for 'k' in 'kth'.</param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the kth element of each dimension.</returns>
        public System.Tuple<IntTensor, LongTensor> KthValue (long k, int dimension, int keepdim)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_kthvalue (values.handle, indices.handle, this.handle, k, dimension, keepdim);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static long THIntTensor_trace (HType self);
        
        /// <summary>
        ///   Computes the trace of the tensor. 
        /// </summary>
        public long Trace ()
        {
            return THIntTensor_trace(this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_sign (HType result, HType self);
        
        /// <summary>
        ///   Computes the sign of the tensor. 
        /// </summary>
        public IntTensor Sign ()
        {
            var result = new IntTensor();
            THIntTensor_sign(result.handle, this.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_cross (HType result, HType a, HType b);
        
        /// <summary>
        ///   Computes the cross product of two tensors. 
        /// </summary>
        /// <param name="other">The right-hand-side tensor.</param>
        public IntTensor CrossProduct (IntTensor other)
        {
            var result = new IntTensor();
            THIntTensor_cross(result.handle, this.handle, other.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_diag (HType result, HType self, int k);
        
        /// <summary>
        ///   Gets the diagonal of the tensor. 
        /// </summary>
        /// <param name="k"></param>
        public IntTensor Diagonal (int k)
        {
            var result = new IntTensor();
            THIntTensor_diag(result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_eye (HType result, long m, long n);
        
        /// <summary>
        ///   Eye. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static IntTensor Eye (long m, long n)
        {
            var result = new IntTensor();
            THIntTensor_eye(result.handle, m, n);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_range (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static IntTensor Range (long xmin, long xmax, long step)
        {
            var result = new IntTensor();
            THIntTensor_range(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_arange (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static IntTensor ARange (long xmin, long xmax, long step)
        {
            var result = new IntTensor();
            THIntTensor_arange(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_sort (HType values, LongTensor.HType indices, HType self, int dimension, int descending);
        
        /// <summary>
        ///   Sorts the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to sort along.</param>
        /// <param name="descending">0 if ascending, 1 if descending.</param>
        /// <returns>A tuple containing the values and indices of the sorted elements.</returns>
        public System.Tuple<IntTensor, LongTensor> Sort (int dimension, int descending)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_sort (values.handle, indices.handle, this.handle, dimension, descending);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_topk (HType values, LongTensor.HType indices, HType self, long k, int dim, int dir, int sorted);
        
        /// <summary>
        ///   Finds the top k of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The number of elements to fetch.</param>
        /// <param name="dim">The dimension along which to sort and find k elements.</param>
        /// <param name="dir">0 if ascending, 1 if descending.</param>
        /// <param name="sorted">1 if the result should be sorted, 0 if they should keep their original order.</param>
        /// <returns>A tuple containing the values and indices of the top 'k' elements.</returns>
        public System.Tuple<IntTensor, LongTensor> TopK (long k, int dim, int dir, int sorted)
        {
            var values = new IntTensor ();
            var indices = new LongTensor ();
            THIntTensor_topk (values.handle, indices.handle, this.handle, k, dim, dir, sorted);
            return new System.Tuple<IntTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_tril (HType result, HType self, long k);
        
        /// <summary>
        ///   Lower triangle. 
        /// </summary>
        /// <param name="k"></param>
        public IntTensor TriL (long k)
        {
            var result = new IntTensor ();
            THIntTensor_tril (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_triu (HType result, HType self, long k);
        
        /// <summary>
        ///   Upper triangle. 
        /// </summary>
        /// <param name="k"></param>
        public IntTensor TriU (long k)
        {
            var result = new IntTensor ();
            THIntTensor_triu (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_cat (HType result, HType ta, HType tb, int dimension);
        
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="other">The second tensor.</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public IntTensor Concatenate (IntTensor other, int dimension)
        {
            var result = new IntTensor ();
            THIntTensor_cat (result.handle, this.handle, other.handle, dimension);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_catArray (HType result, HType[] ta, int count, int dimension);
#if false        
// NOTE: We need to determine the right marshalling for an array of handles.
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="tensors">A collection of tensors..</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public static IntTensor Concatenate (IEnumerable<IntTensor> tensors, int dimension)
        {
            var result = new IntTensor ();
            var handleArray = tensors.Select(t => t.handle).ToArray();
            THIntTensor_catArray (result.handle, handleArray, (int)handleArray.Length, dimension);
            return result;
        }
#endif
     
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
        extern static long THLongTensor_numel (HType handle);
     
        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumElements ()
        {
            return THLongTensor_numel (handle);
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
        extern static void THLongTensor_nonzero (LongTensor.HType subscript, HType handle);
     
        /// <summary>
        ///  Finds the indices of all non-zero elements.
        /// </summary>
        public LongTensor NonZero ()
        {
            var result = new LongTensor();
            THLongTensor_nonzero (result.handle, this.handle);
            return result;
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
        /// <param name="size3">Size of the fourth dimension.</param>     
        /// <param name="stride3">Stride of the fourth dimension.</param>
        public LongTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new LongTensor(THLongTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        public void Squeeze ()
        {
            THLongTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to remove.</param>
        public void Squeeze1d (LongTensor src, int dimension)
        {
            THLongTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to insert.</param>
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
        ///  Populates the tensor with random values using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from min to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower limit for the values to be generated</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_randperm (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void RandPerm (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_randperm (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
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
        ///   Get a string representation of the tensor.
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
        /// <param name="src">The right-hand-side operand.</param>
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
        ///   Match
        /// </summary>
        /// <param name="m2"></param>
        /// <param name="gain"></param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        extern static void THLongTensor_cmax (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMax of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CMax (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cmax (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cmin (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMin of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CMin (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_cmin (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_ltTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THLongTensor_ltTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_leTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THLongTensor_leTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_gtTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THLongTensor_gtTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_geTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THLongTensor_geTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_eqTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THLongTensor_eqTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_neTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensor (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THLongTensor_neTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_ltTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor LtTensorT (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_ltTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_leTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor LeTensorT (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_leTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_gtTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor GtTensorT (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_gtTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_geTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor GeTensorT (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_geTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_eqTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor EqTensorT (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_eqTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_neTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor NeTensorT (LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new LongTensor ();
            THLongTensor_neTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cmaxvalue (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an CMaxValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CMaxValue (long src)
        {
            var result = new LongTensor ();
            THLongTensor_cmaxvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_cminvalue (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an CMinValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor CMinValue (long src)
        {
            var result = new LongTensor ();
            THLongTensor_cminvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_ltValue (ByteTensor.HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an LtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValue (long src)
        {
            var result = new ByteTensor ();
            THLongTensor_ltValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_leValue (ByteTensor.HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an LeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValue (long src)
        {
            var result = new ByteTensor ();
            THLongTensor_leValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_gtValue (ByteTensor.HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an GtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValue (long src)
        {
            var result = new ByteTensor ();
            THLongTensor_gtValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_geValue (ByteTensor.HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an GeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValue (long src)
        {
            var result = new ByteTensor ();
            THLongTensor_geValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_eqValue (ByteTensor.HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an EqValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValue (long src)
        {
            var result = new ByteTensor ();
            THLongTensor_eqValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_neValue (ByteTensor.HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an NeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValue (long src)
        {
            var result = new ByteTensor ();
            THLongTensor_neValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_ltValueT (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an LtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor LtValueT (long src)
        {
            var result = new LongTensor ();
            THLongTensor_ltValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_leValueT (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an LeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor LeValueT (long src)
        {
            var result = new LongTensor ();
            THLongTensor_leValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_gtValueT (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an GtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor GtValueT (long src)
        {
            var result = new LongTensor ();
            THLongTensor_gtValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_geValueT (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an GeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor GeValueT (long src)
        {
            var result = new LongTensor ();
            THLongTensor_geValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_eqValueT (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an EqValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor EqValueT (long src)
        {
            var result = new LongTensor ();
            THLongTensor_eqValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_neValueT (HType result, HType t, long value);
        
        /// <summary>
        ///   Performs an NeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor NeValueT (long src)
        {
            var result = new LongTensor ();
            THLongTensor_neValueT (result.handle, this.handle, src);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_lerp (HType result, HType self, HType other, long weight);
        
        /// <summary>
        ///   LERP
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        /// <param name="weight"></param>
        public LongTensor LERP (LongTensor other, long weight)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            var result = new LongTensor();
            THLongTensor_lerp (result.handle, this.handle, other.handle, weight);
            return result;
        }

        [DllImport ("caffe2")]
        extern static int THLongTensor_equal (HType t, HType src);
        
        /// <summary>
        ///   Compare the tensor with another for complete equality.
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        public int Equal (LongTensor other)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THLongTensor_equal (this.handle, other.handle);
        }
                
        [DllImport ("caffe2")]
        extern static void THLongTensor_add_scaled (HType result, HType t, long value1, long value2);
        
        /// <summary>
        ///   Performs an AddScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor AddScaled (long value1, long value2)
        {
            var result = new LongTensor ();
            THLongTensor_add_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_sub_scaled (HType result, HType t, long value1, long value2);
        
        /// <summary>
        ///   Performs an SubScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor SubScaled (long value1, long value2)
        {
            var result = new LongTensor ();
            THLongTensor_sub_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_clamp (HType result, HType t, long value1, long value2);
        
        /// <summary>
        ///   Performs an Clamp of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public LongTensor Clamp (long value1, long value2)
        {
            var result = new LongTensor ();
            THLongTensor_clamp (result.handle, this.handle, value1, value2);
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
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for matxvec (α)</param>
        /// <param name="mat">Matrix to be multiplied</param>
        /// <param name="vec">Vector to be multiplied</param>
        /// <remarks>
        /// β tensor+α (mat@vec)

        /// </remarks>
        /// <returns>
        ///   β tensor+α (mat@vec)
        /// </returns>
        public LongTensor AddMV (long beta, long alpha, LongTensor mat, LongTensor vec)
        {
            if (mat == null)
                throw new ArgumentNullException (nameof (mat));
            if (vec == null)
                throw new ArgumentNullException (nameof (vec));
            var result = new LongTensor ();
            THLongTensor_addmv (result.handle, beta, this.handle, alpha, mat.handle, vec.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addmm (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for mat1xmat2 (α)</param>
        /// <param name="mat1">First matrix to  be multiplied</param>
        /// <param name="mat2">Second matrix to  be multiplied</param>
        /// <remarks>
        /// β mat+α (mat1i@mat2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (mat1i@mat2i)
        /// </returns>
        public LongTensor AddMM (long beta, long alpha, LongTensor mat1, LongTensor mat2)
        {
            if (mat1 == null)
                throw new ArgumentNullException (nameof (mat1));
            if (mat2 == null)
                throw new ArgumentNullException (nameof (mat2));
            var result = new LongTensor ();
            THLongTensor_addmm (result.handle, beta, this.handle, alpha, mat1.handle, mat2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addbmm (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mat+α (∑i=0bbatch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (∑i=0bbatch1i@batch2i)
        /// </returns>
        public LongTensor AddBMM (long beta, long alpha, LongTensor batch1, LongTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new LongTensor ();
            THLongTensor_addbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_addr (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for vec1xvec2 (α)</param>
        /// <param name="vec1">the first vector of the outer product</param>
        /// <param name="vec2">the second vector of the outer product</param>
        /// <remarks>
        /// β mat+α (vec1⊗vec2)

        /// </remarks>
        /// <returns>
        ///   β mat+α (vec1⊗vec2)
        /// </returns>
        public LongTensor AddR (long beta, long alpha, LongTensor vec1, LongTensor vec2)
        {
            if (vec1 == null)
                throw new ArgumentNullException (nameof (vec1));
            if (vec2 == null)
                throw new ArgumentNullException (nameof (vec2));
            var result = new LongTensor ();
            THLongTensor_addr (result.handle, beta, this.handle, alpha, vec1.handle, vec2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THLongTensor_baddbmm (HType result, long beta, HType t, long alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mati+α (batch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mati+α (batch1i@batch2i)
        /// </returns>
        public LongTensor BAddBMM (long beta, long alpha, LongTensor batch1, LongTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new LongTensor ();
            THLongTensor_baddbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
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
        extern static void THLongTensor_indexAdd (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Adds the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the add</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexAdd (int dim, LongTensor index, LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_indexAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_indexFill (HType tensor, int dim, LongTensor.HType index, long value);
        
        /// <summary>
        ///   Uses the given value to overwrite the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the fill</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="value">The value to write.</param>
        public void IndexFill (int dim, LongTensor index, long value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_indexFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_take (HType self, HType src, LongTensor.HType index);

        /// <summary>
        ///   Take
        /// </summary>        
        /// <param name="src"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Take (LongTensor src, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_take (handle, src.handle, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_put (HType self, LongTensor.HType index, HType src, int accumulate);

        /// <summary>
        ///   Put
        /// </summary>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        /// <param name="accumulate"></param>
        public void Put (LongTensor index, LongTensor src, int accumulate)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_put (handle, index.handle, src.handle, accumulate);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_gather (HType self, HType src, int dim, LongTensor.HType index);

        /// <summary>
        ///   Gather
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Gather (LongTensor src, int dim, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_gather (handle, src.handle, dim, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_scatter (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   Scatter
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void Scatter (int dim, LongTensor index, LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_scatter (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_scatterAdd (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void ScatterAdd (int dim, LongTensor index, LongTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_scatterAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_scatterFill (HType self, int dim, LongTensor.HType index, long value);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="value"></param>
        public void ScatterFill (int dim, LongTensor index, long value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THLongTensor_scatterFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
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
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THLongTensor_copyDouble (this.handle, src.handle);
        }

        
        
     
        [DllImport ("caffe2")]
        extern static void THLongTensor_sum (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public LongTensor Sum (int dimension, int keepdim)
        {
            var result = new LongTensor ();
            THLongTensor_sum (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_cumsum (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public LongTensor CumulativeSum (int dimension)
        {
            var result = new LongTensor ();
            THLongTensor_cumsum (result.handle, this.handle, dimension);
            return result;
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_prod (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public LongTensor Prod (int dimension, int keepdim)
        {
            var result = new LongTensor ();
            THLongTensor_prod (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_cumprod (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public LongTensor CumulativeProd (int dimension)
        {
            var result = new LongTensor ();
            THLongTensor_cumprod (result.handle, this.handle, dimension);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THLongTensor_max (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the max of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<LongTensor, LongTensor> Max (int dimension, int keepdim)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_max (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_min (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the min of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<LongTensor, LongTensor> Min (int dimension, int keepdim)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_min (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_mode (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the mode of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<LongTensor, LongTensor> Mode (int dimension, int keepdim)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_mode (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THLongTensor_median (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the median of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<LongTensor, LongTensor> Median (int dimension, int keepdim)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_median (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }
     

        [DllImport ("caffe2")]
        extern static void THLongTensor_kthvalue (HType values, LongTensor.HType indices, HType self, long k, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the kth value of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The value for 'k' in 'kth'.</param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the kth element of each dimension.</returns>
        public System.Tuple<LongTensor, LongTensor> KthValue (long k, int dimension, int keepdim)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_kthvalue (values.handle, indices.handle, this.handle, k, dimension, keepdim);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static long THLongTensor_trace (HType self);
        
        /// <summary>
        ///   Computes the trace of the tensor. 
        /// </summary>
        public long Trace ()
        {
            return THLongTensor_trace(this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_sign (HType result, HType self);
        
        /// <summary>
        ///   Computes the sign of the tensor. 
        /// </summary>
        public LongTensor Sign ()
        {
            var result = new LongTensor();
            THLongTensor_sign(result.handle, this.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_cross (HType result, HType a, HType b);
        
        /// <summary>
        ///   Computes the cross product of two tensors. 
        /// </summary>
        /// <param name="other">The right-hand-side tensor.</param>
        public LongTensor CrossProduct (LongTensor other)
        {
            var result = new LongTensor();
            THLongTensor_cross(result.handle, this.handle, other.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_diag (HType result, HType self, int k);
        
        /// <summary>
        ///   Gets the diagonal of the tensor. 
        /// </summary>
        /// <param name="k"></param>
        public LongTensor Diagonal (int k)
        {
            var result = new LongTensor();
            THLongTensor_diag(result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_eye (HType result, long m, long n);
        
        /// <summary>
        ///   Eye. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static LongTensor Eye (long m, long n)
        {
            var result = new LongTensor();
            THLongTensor_eye(result.handle, m, n);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_range (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static LongTensor Range (long xmin, long xmax, long step)
        {
            var result = new LongTensor();
            THLongTensor_range(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_arange (HType result, long xmin, long xmax, long step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static LongTensor ARange (long xmin, long xmax, long step)
        {
            var result = new LongTensor();
            THLongTensor_arange(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_sort (HType values, LongTensor.HType indices, HType self, int dimension, int descending);
        
        /// <summary>
        ///   Sorts the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to sort along.</param>
        /// <param name="descending">0 if ascending, 1 if descending.</param>
        /// <returns>A tuple containing the values and indices of the sorted elements.</returns>
        public System.Tuple<LongTensor, LongTensor> Sort (int dimension, int descending)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_sort (values.handle, indices.handle, this.handle, dimension, descending);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_topk (HType values, LongTensor.HType indices, HType self, long k, int dim, int dir, int sorted);
        
        /// <summary>
        ///   Finds the top k of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The number of elements to fetch.</param>
        /// <param name="dim">The dimension along which to sort and find k elements.</param>
        /// <param name="dir">0 if ascending, 1 if descending.</param>
        /// <param name="sorted">1 if the result should be sorted, 0 if they should keep their original order.</param>
        /// <returns>A tuple containing the values and indices of the top 'k' elements.</returns>
        public System.Tuple<LongTensor, LongTensor> TopK (long k, int dim, int dir, int sorted)
        {
            var values = new LongTensor ();
            var indices = new LongTensor ();
            THLongTensor_topk (values.handle, indices.handle, this.handle, k, dim, dir, sorted);
            return new System.Tuple<LongTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_tril (HType result, HType self, long k);
        
        /// <summary>
        ///   Lower triangle. 
        /// </summary>
        /// <param name="k"></param>
        public LongTensor TriL (long k)
        {
            var result = new LongTensor ();
            THLongTensor_tril (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_triu (HType result, HType self, long k);
        
        /// <summary>
        ///   Upper triangle. 
        /// </summary>
        /// <param name="k"></param>
        public LongTensor TriU (long k)
        {
            var result = new LongTensor ();
            THLongTensor_triu (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_cat (HType result, HType ta, HType tb, int dimension);
        
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="other">The second tensor.</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public LongTensor Concatenate (LongTensor other, int dimension)
        {
            var result = new LongTensor ();
            THLongTensor_cat (result.handle, this.handle, other.handle, dimension);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_catArray (HType result, HType[] ta, int count, int dimension);
#if false        
// NOTE: We need to determine the right marshalling for an array of handles.
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="tensors">A collection of tensors..</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public static LongTensor Concatenate (IEnumerable<LongTensor> tensors, int dimension)
        {
            var result = new LongTensor ();
            var handleArray = tensors.Select(t => t.handle).ToArray();
            THLongTensor_catArray (result.handle, handleArray, (int)handleArray.Length, dimension);
            return result;
        }
#endif
     
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
        extern static long THDoubleTensor_numel (HType handle);
     
        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumElements ()
        {
            return THDoubleTensor_numel (handle);
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
        extern static void THDoubleTensor_nonzero (LongTensor.HType subscript, HType handle);
     
        /// <summary>
        ///  Finds the indices of all non-zero elements.
        /// </summary>
        public LongTensor NonZero ()
        {
            var result = new LongTensor();
            THDoubleTensor_nonzero (result.handle, this.handle);
            return result;
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
        /// <param name="size3">Size of the fourth dimension.</param>     
        /// <param name="stride3">Stride of the fourth dimension.</param>
        public DoubleTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new DoubleTensor(THDoubleTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        public void Squeeze ()
        {
            THDoubleTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to remove.</param>
        public void Squeeze1d (DoubleTensor src, int dimension)
        {
            THDoubleTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to insert.</param>
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
        ///  Populates the tensor with random values using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from min to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower limit for the values to be generated</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_randperm (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void RandPerm (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_randperm (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
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
        ///   Fills the tensor with values according to a Bernoulli distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
        public void BernoulliTensor (RandomGenerator source, DoubleTensor p)
        {
            THDoubleTensor_bernoulli_DoubleTensor(this.handle, source.handle, p.handle);
        }
#endif

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_uniform (HType self, IntPtr thgenerator, double min, double max);

        /// <summary>
        ///   Fills the tensor with values according to a Bernoulli distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower bound for the random number range.</param>
        /// <param name="max">The upper bound for the random number range.</param>
        public void Uniform (RandomGenerator source, double min, double max)
        {
            THDoubleTensor_uniform(this.handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_exponential (HType self, IntPtr thgenerator, double lambda);

        /// <summary>
        ///   Fills the tensor with values according to a exponential distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="lambda"></param>
        public void Exponential (RandomGenerator source, double lambda)
        {
            THDoubleTensor_exponential(this.handle, source.handle, lambda);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cauchy (HType self, IntPtr thgenerator, double median, double sigma);

        /// <summary>
        ///   Fills the tensor with values according to a Cauchy-Lorentz distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="median"></param>
        /// <param name="sigma"></param>
        public void Cauchy (RandomGenerator source, double median, double sigma)
        {
            THDoubleTensor_cauchy(this.handle, source.handle, median, sigma);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_logNormal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Fills the tensor with values according to a log-normal distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdv">The standard deviation of the distribution.</param>
        public void LogNormal (RandomGenerator source, double mean, double stdv)
        {
            THDoubleTensor_logNormal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdv">The standard deviation of the distribution.</param>
        public void Normal (RandomGenerator source, double mean, double stdv)
        {
            THDoubleTensor_normal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal_means (HType self, IntPtr thgenerator, HType means, double stdv);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution. This version uses multiple means.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="means">The means of the distribution.</param>
        /// <param name="stdv">The standard deviation of the distribution.</param>
        public void NormalMeans (RandomGenerator source, DoubleTensor means, double stdv)
        {
            THDoubleTensor_normal_means(this.handle, source.handle, means.handle, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal_stddevs (HType self, IntPtr thgenerator, double mean, HType stdvs);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution. This version uses multiple standard deviations.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdvs">The standard deviations of the distribution.</param>
        public void NormalStdvs (RandomGenerator source, double mean, DoubleTensor stdvs)
        {
            THDoubleTensor_normal_stddevs(this.handle, source.handle, mean, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_normal_means_stddevs (HType self, IntPtr thgenerator, HType means, HType stdvs);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution. This version uses multiple means and standard deviations.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="means">The means of the distribution.</param>
        /// <param name="stdvs">The standards deviation of the distribution.</param>
        public void NormalMeansStdvs (RandomGenerator source, DoubleTensor means, DoubleTensor stdvs)
        {
            THDoubleTensor_normal_means_stddevs(this.handle, source.handle, means.handle, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_multinomial (HType self, IntPtr thgenerator, HType prob_dist, int n_sample, int with_replacement);

        /// <summary>
        ///   Fills the tensor with values according to a multinomial distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="prob_dist">The probability for each bucket.</param>
        /// <param name="n_sample">The number of samples to generate.</param>
        /// <param name="with_replacement"></param>
        public void Multinomial (RandomGenerator source, DoubleTensor prob_dist, int n_sample, int with_replacement)
        {
            THDoubleTensor_multinomial(this.handle, source.handle, prob_dist.handle, n_sample, with_replacement);
        }
        
        /// <summary>
        ///   Get a string representation of the tensor.
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
        public DoubleTensor Sigmoid ()
        {
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
        public DoubleTensor Log ()
        {
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
        public DoubleTensor Lgamma ()
        {
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
        public DoubleTensor Digamma ()
        {
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
        public DoubleTensor Trigamma ()
        {
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
        public DoubleTensor Polygamma ()
        {
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
        public DoubleTensor Log10 ()
        {
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
        public DoubleTensor Log1p ()
        {
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
        public DoubleTensor Log2 ()
        {
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
        public DoubleTensor Exp ()
        {
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
        public DoubleTensor Expm1 ()
        {
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
        public DoubleTensor Cos ()
        {
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
        public DoubleTensor Acos ()
        {
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
        public DoubleTensor Cosh ()
        {
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
        public DoubleTensor Sin ()
        {
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
        public DoubleTensor Asin ()
        {
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
        public DoubleTensor Sinh ()
        {
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
        public DoubleTensor Tan ()
        {
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
        public DoubleTensor Atan ()
        {
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
        public DoubleTensor Atan2 ()
        {
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
        public DoubleTensor Tanh ()
        {
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
        public DoubleTensor Erf ()
        {
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
        public DoubleTensor Erfc ()
        {
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
        public DoubleTensor Erfinv ()
        {
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
        public DoubleTensor Sqrt ()
        {
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
        public DoubleTensor Rsqrt ()
        {
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
        public DoubleTensor Ceil ()
        {
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
        public DoubleTensor Floor ()
        {
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
        public DoubleTensor Round ()
        {
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
        public DoubleTensor Abs ()
        {
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
        public DoubleTensor Trunc ()
        {
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
        public DoubleTensor Frac ()
        {
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
        public DoubleTensor cinv ()
        {
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
        public DoubleTensor neg ()
        {
            var result = new DoubleTensor ();
            THDoubleTensor_neg (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_zerosLike (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the ZerosLike of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor ZerosLike ()
        {
            var result = new DoubleTensor ();
            THDoubleTensor_zerosLike (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_onesLike (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the OnesLike of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor OnesLike ()
        {
            var result = new DoubleTensor ();
            THDoubleTensor_onesLike (result.handle, this.handle);
            return result;
        }


                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_pow (HType result, HType x, double y);

        /// <summary>
        ///   Returns a new tensor with <see paramref="this"/> raised to the power of <see paramref="y"/>.
        /// </summary>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        ///   Match
        /// </summary>
        /// <param name="m2"></param>
        /// <param name="gain"></param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        extern static void THDoubleTensor_cmax (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMax of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CMax (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cmax (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cmin (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMin of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CMin (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_cmin (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_ltTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THDoubleTensor_ltTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_leTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THDoubleTensor_leTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_gtTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THDoubleTensor_gtTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_geTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THDoubleTensor_geTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_eqTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THDoubleTensor_eqTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_neTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensor (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THDoubleTensor_neTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_ltTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor LtTensorT (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_ltTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_leTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor LeTensorT (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_leTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_gtTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor GtTensorT (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_gtTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_geTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor GeTensorT (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_geTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_eqTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor EqTensorT (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_eqTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_neTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor NeTensorT (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new DoubleTensor ();
            THDoubleTensor_neTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cmaxvalue (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an CMaxValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CMaxValue (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_cmaxvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cminvalue (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an CMinValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor CMinValue (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_cminvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_ltValue (ByteTensor.HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an LtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValue (double src)
        {
            var result = new ByteTensor ();
            THDoubleTensor_ltValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_leValue (ByteTensor.HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an LeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValue (double src)
        {
            var result = new ByteTensor ();
            THDoubleTensor_leValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_gtValue (ByteTensor.HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an GtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValue (double src)
        {
            var result = new ByteTensor ();
            THDoubleTensor_gtValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_geValue (ByteTensor.HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an GeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValue (double src)
        {
            var result = new ByteTensor ();
            THDoubleTensor_geValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_eqValue (ByteTensor.HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an EqValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValue (double src)
        {
            var result = new ByteTensor ();
            THDoubleTensor_eqValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_neValue (ByteTensor.HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an NeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValue (double src)
        {
            var result = new ByteTensor ();
            THDoubleTensor_neValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_ltValueT (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an LtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor LtValueT (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_ltValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_leValueT (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an LeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor LeValueT (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_leValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_gtValueT (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an GtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor GtValueT (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_gtValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_geValueT (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an GeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor GeValueT (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_geValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_eqValueT (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an EqValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor EqValueT (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_eqValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_neValueT (HType result, HType t, double value);
        
        /// <summary>
        ///   Performs an NeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor NeValueT (double src)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_neValueT (result.handle, this.handle, src);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_lerp (HType result, HType self, HType other, double weight);
        
        /// <summary>
        ///   LERP
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        /// <param name="weight"></param>
        public DoubleTensor LERP (DoubleTensor other, double weight)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            var result = new DoubleTensor();
            THDoubleTensor_lerp (result.handle, this.handle, other.handle, weight);
            return result;
        }

        [DllImport ("caffe2")]
        extern static int THDoubleTensor_equal (HType t, HType src);
        
        /// <summary>
        ///   Compare the tensor with another for complete equality.
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        public int Equal (DoubleTensor other)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THDoubleTensor_equal (this.handle, other.handle);
        }
                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_add_scaled (HType result, HType t, double value1, double value2);
        
        /// <summary>
        ///   Performs an AddScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor AddScaled (double value1, double value2)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_add_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sub_scaled (HType result, HType t, double value1, double value2);
        
        /// <summary>
        ///   Performs an SubScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor SubScaled (double value1, double value2)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_sub_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_clamp (HType result, HType t, double value1, double value2);
        
        /// <summary>
        ///   Performs an Clamp of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Clamp (double value1, double value2)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_clamp (result.handle, this.handle, value1, value2);
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
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for matxvec (α)</param>
        /// <param name="mat">Matrix to be multiplied</param>
        /// <param name="vec">Vector to be multiplied</param>
        /// <remarks>
        /// β tensor+α (mat@vec)

        /// </remarks>
        /// <returns>
        ///   β tensor+α (mat@vec)
        /// </returns>
        public DoubleTensor AddMV (double beta, double alpha, DoubleTensor mat, DoubleTensor vec)
        {
            if (mat == null)
                throw new ArgumentNullException (nameof (mat));
            if (vec == null)
                throw new ArgumentNullException (nameof (vec));
            var result = new DoubleTensor ();
            THDoubleTensor_addmv (result.handle, beta, this.handle, alpha, mat.handle, vec.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addmm (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for mat1xmat2 (α)</param>
        /// <param name="mat1">First matrix to  be multiplied</param>
        /// <param name="mat2">Second matrix to  be multiplied</param>
        /// <remarks>
        /// β mat+α (mat1i@mat2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (mat1i@mat2i)
        /// </returns>
        public DoubleTensor AddMM (double beta, double alpha, DoubleTensor mat1, DoubleTensor mat2)
        {
            if (mat1 == null)
                throw new ArgumentNullException (nameof (mat1));
            if (mat2 == null)
                throw new ArgumentNullException (nameof (mat2));
            var result = new DoubleTensor ();
            THDoubleTensor_addmm (result.handle, beta, this.handle, alpha, mat1.handle, mat2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addbmm (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mat+α (∑i=0bbatch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (∑i=0bbatch1i@batch2i)
        /// </returns>
        public DoubleTensor AddBMM (double beta, double alpha, DoubleTensor batch1, DoubleTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new DoubleTensor ();
            THDoubleTensor_addbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_addr (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for vec1xvec2 (α)</param>
        /// <param name="vec1">the first vector of the outer product</param>
        /// <param name="vec2">the second vector of the outer product</param>
        /// <remarks>
        /// β mat+α (vec1⊗vec2)

        /// </remarks>
        /// <returns>
        ///   β mat+α (vec1⊗vec2)
        /// </returns>
        public DoubleTensor AddR (double beta, double alpha, DoubleTensor vec1, DoubleTensor vec2)
        {
            if (vec1 == null)
                throw new ArgumentNullException (nameof (vec1));
            if (vec2 == null)
                throw new ArgumentNullException (nameof (vec2));
            var result = new DoubleTensor ();
            THDoubleTensor_addr (result.handle, beta, this.handle, alpha, vec1.handle, vec2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_baddbmm (HType result, double beta, HType t, double alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mati+α (batch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mati+α (batch1i@batch2i)
        /// </returns>
        public DoubleTensor BAddBMM (double beta, double alpha, DoubleTensor batch1, DoubleTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new DoubleTensor ();
            THDoubleTensor_baddbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
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
        extern static void THDoubleTensor_linspace (HType result, double a, double b, long n);
        
        /// <summary>
        ///   Performs Linspace of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="n"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Linspace (double a, double b, long n)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_linspace (result.handle, a, b, n);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_logspace (HType result, double a, double b, long n);
        
        /// <summary>
        ///   Performs Logspace of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="n"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public DoubleTensor Logspace (double a, double b, long n)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_logspace (result.handle, a, b, n);
            return result;
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
        extern static void THDoubleTensor_indexAdd (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Adds the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the add</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexAdd (int dim, LongTensor index, DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_indexAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_indexFill (HType tensor, int dim, LongTensor.HType index, double value);
        
        /// <summary>
        ///   Uses the given value to overwrite the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the fill</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="value">The value to write.</param>
        public void IndexFill (int dim, LongTensor index, double value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_indexFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_take (HType self, HType src, LongTensor.HType index);

        /// <summary>
        ///   Take
        /// </summary>        
        /// <param name="src"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Take (DoubleTensor src, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_take (handle, src.handle, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_put (HType self, LongTensor.HType index, HType src, int accumulate);

        /// <summary>
        ///   Put
        /// </summary>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        /// <param name="accumulate"></param>
        public void Put (LongTensor index, DoubleTensor src, int accumulate)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_put (handle, index.handle, src.handle, accumulate);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_gather (HType self, HType src, int dim, LongTensor.HType index);

        /// <summary>
        ///   Gather
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Gather (DoubleTensor src, int dim, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_gather (handle, src.handle, dim, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_scatter (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   Scatter
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void Scatter (int dim, LongTensor index, DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_scatter (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_scatterAdd (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void ScatterAdd (int dim, LongTensor index, DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_scatterAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_scatterFill (HType self, int dim, LongTensor.HType index, double value);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="value"></param>
        public void ScatterFill (int dim, LongTensor index, double value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THDoubleTensor_scatterFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
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
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THDoubleTensor_copyDouble (this.handle, src.handle);
        }

        
        
     
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sum (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public DoubleTensor Sum (int dimension, int keepdim)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_sum (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cumsum (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public DoubleTensor CumulativeSum (int dimension)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_cumsum (result.handle, this.handle, dimension);
            return result;
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_prod (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public DoubleTensor Prod (int dimension, int keepdim)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_prod (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cumprod (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public DoubleTensor CumulativeProd (int dimension)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_cumprod (result.handle, this.handle, dimension);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_max (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the max of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<DoubleTensor, LongTensor> Max (int dimension, int keepdim)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_max (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_min (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the min of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<DoubleTensor, LongTensor> Min (int dimension, int keepdim)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_min (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_mode (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the mode of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<DoubleTensor, LongTensor> Mode (int dimension, int keepdim)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_mode (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_median (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the median of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<DoubleTensor, LongTensor> Median (int dimension, int keepdim)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_median (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }
     

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_kthvalue (HType values, LongTensor.HType indices, HType self, long k, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the kth value of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The value for 'k' in 'kth'.</param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the kth element of each dimension.</returns>
        public System.Tuple<DoubleTensor, LongTensor> KthValue (long k, int dimension, int keepdim)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_kthvalue (values.handle, indices.handle, this.handle, k, dimension, keepdim);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_trace (HType self);
        
        /// <summary>
        ///   Computes the trace of the tensor. 
        /// </summary>
        public double Trace ()
        {
            return THDoubleTensor_trace(this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sign (HType result, HType self);
        
        /// <summary>
        ///   Computes the sign of the tensor. 
        /// </summary>
        public DoubleTensor Sign ()
        {
            var result = new DoubleTensor();
            THDoubleTensor_sign(result.handle, this.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cross (HType result, HType a, HType b);
        
        /// <summary>
        ///   Computes the cross product of two tensors. 
        /// </summary>
        /// <param name="other">The right-hand-side tensor.</param>
        public DoubleTensor CrossProduct (DoubleTensor other)
        {
            var result = new DoubleTensor();
            THDoubleTensor_cross(result.handle, this.handle, other.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_diag (HType result, HType self, int k);
        
        /// <summary>
        ///   Gets the diagonal of the tensor. 
        /// </summary>
        /// <param name="k"></param>
        public DoubleTensor Diagonal (int k)
        {
            var result = new DoubleTensor();
            THDoubleTensor_diag(result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_eye (HType result, long m, long n);
        
        /// <summary>
        ///   Eye. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static DoubleTensor Eye (long m, long n)
        {
            var result = new DoubleTensor();
            THDoubleTensor_eye(result.handle, m, n);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_range (HType result, double xmin, double xmax, double step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static DoubleTensor Range (double xmin, double xmax, double step)
        {
            var result = new DoubleTensor();
            THDoubleTensor_range(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_arange (HType result, double xmin, double xmax, double step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static DoubleTensor ARange (double xmin, double xmax, double step)
        {
            var result = new DoubleTensor();
            THDoubleTensor_arange(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_sort (HType values, LongTensor.HType indices, HType self, int dimension, int descending);
        
        /// <summary>
        ///   Sorts the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to sort along.</param>
        /// <param name="descending">0 if ascending, 1 if descending.</param>
        /// <returns>A tuple containing the values and indices of the sorted elements.</returns>
        public System.Tuple<DoubleTensor, LongTensor> Sort (int dimension, int descending)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_sort (values.handle, indices.handle, this.handle, dimension, descending);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_topk (HType values, LongTensor.HType indices, HType self, long k, int dim, int dir, int sorted);
        
        /// <summary>
        ///   Finds the top k of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The number of elements to fetch.</param>
        /// <param name="dim">The dimension along which to sort and find k elements.</param>
        /// <param name="dir">0 if ascending, 1 if descending.</param>
        /// <param name="sorted">1 if the result should be sorted, 0 if they should keep their original order.</param>
        /// <returns>A tuple containing the values and indices of the top 'k' elements.</returns>
        public System.Tuple<DoubleTensor, LongTensor> TopK (long k, int dim, int dir, int sorted)
        {
            var values = new DoubleTensor ();
            var indices = new LongTensor ();
            THDoubleTensor_topk (values.handle, indices.handle, this.handle, k, dim, dir, sorted);
            return new System.Tuple<DoubleTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_tril (HType result, HType self, long k);
        
        /// <summary>
        ///   Lower triangle. 
        /// </summary>
        /// <param name="k"></param>
        public DoubleTensor TriL (long k)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_tril (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_triu (HType result, HType self, long k);
        
        /// <summary>
        ///   Upper triangle. 
        /// </summary>
        /// <param name="k"></param>
        public DoubleTensor TriU (long k)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_triu (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_cat (HType result, HType ta, HType tb, int dimension);
        
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="other">The second tensor.</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public DoubleTensor Concatenate (DoubleTensor other, int dimension)
        {
            var result = new DoubleTensor ();
            THDoubleTensor_cat (result.handle, this.handle, other.handle, dimension);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_catArray (HType result, HType[] ta, int count, int dimension);
#if false        
// NOTE: We need to determine the right marshalling for an array of handles.
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="tensors">A collection of tensors..</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public static DoubleTensor Concatenate (IEnumerable<DoubleTensor> tensors, int dimension)
        {
            var result = new DoubleTensor ();
            var handleArray = tensors.Select(t => t.handle).ToArray();
            THDoubleTensor_catArray (result.handle, handleArray, (int)handleArray.Length, dimension);
            return result;
        }
#endif
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_mean (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Compute the mean of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public DoubleTensor Mean (int dimension, int keepdim)
        {
            var result = new DoubleTensor();
            THDoubleTensor_mean (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_std (HType result, HType self, int dimension, int biased, int keepdim);

        /// <summary>
        ///   Compute the standard deviation of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="biased"></param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public DoubleTensor Std (int dimension, int biased, int keepdim)
        {
            var result = new DoubleTensor();
            THDoubleTensor_std (result.handle, this.handle, dimension, biased, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_var (HType result, HType self, int dimension, int biased, int keepdim);

        /// <summary>
        ///   Compute the variance of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="biased"></param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public DoubleTensor Var (int dimension, int biased, int keepdim)
        {
            var result = new DoubleTensor();
            THDoubleTensor_var (result.handle, this.handle, dimension, biased, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_norm (HType result, HType self, double value,  int dimension, int keepdim);

        /// <summary>
        ///   Compute the norm of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public DoubleTensor Norm (double value,  int dimension, int keepdim)
        {
            var result = new DoubleTensor();
            THDoubleTensor_norm (result.handle, this.handle, value, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_renorm (HType result, HType self, double value,  int dimension, double maxnorm);

        /// <summary>
        ///   Compute the renorm of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="maxnorm"></param>
        public DoubleTensor Renorm (double value,  int dimension, double maxnorm)
        {
            var result = new DoubleTensor();
            THDoubleTensor_renorm (result.handle, this.handle, value, dimension, maxnorm);
            return result;
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_dist (HType a, HType b, double value);

        /// <summary>
        ///   Compute the dist of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="other">The other tensor.</param>
        /// <param name="value"></param>
        public double Dist (DoubleTensor other, double value)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THDoubleTensor_dist (this.handle, other.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_histc (HType hist, HType self, long nbins, double minvalue, double maxvalue);

        /// <summary>
        ///   Create a histogram of all tensor elements. 
        /// </summary>
        /// <param name="nbins">The number of bins in the output histogram.</param>
        /// <param name="minvalue">Only consider values equal to or greater than this.</param>
        /// <param name="maxvalue">Only consider values equal to or less than this.</param>
        public DoubleTensor Histc (long nbins, double minvalue, double maxvalue)
        {
            var result = new DoubleTensor();
            THDoubleTensor_histc (result.handle, this.handle, nbins, minvalue, maxvalue);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_bhistc (HType hist, HType self, long nbins, double minvalue, double maxvalue);

        /// <summary>
        ///   Create a histogram of all tensor elements. 
        /// </summary>
        /// <param name="nbins">The number of bins in the output histogram.</param>
        /// <param name="minvalue">Only consider values equal to or greater than this.</param>
        /// <param name="maxvalue">Only consider values equal to or less than this.</param>
        public DoubleTensor BHistc (long nbins, double minvalue, double maxvalue)
        {
            var result = new DoubleTensor();
            THDoubleTensor_bhistc (result.handle, this.handle, nbins, minvalue, maxvalue);
            return result;
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_meanall (HType self);

        /// <summary>
        ///   Compute the mean of all tensor elements. 
        /// </summary>
        public double MeanAll ()
        {
            return THDoubleTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_varall (HType self, int biased);

        /// <summary>
        ///   Compute the variance of all tensor elements. 
        /// </summary>
        /// <param name="biased"></param>
        public double VarAll (int biased)
        {
            return THDoubleTensor_varall (this.handle, biased);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_stdall (HType self, int biased);

        /// <summary>
        ///   Compute the standard deviation of all tensor elements. 
        /// </summary>
        /// <param name="biased"></param>
        public double StdAll (int biased)
        {
            return THDoubleTensor_stdall (this.handle, biased);
        }

        [DllImport ("caffe2")]
        extern static double THDoubleTensor_normall (HType self, double value);

        /// <summary>
        ///   Compute the norm of all tensor elements. 
        /// </summary>
        /// <param name="value"></param>
        public double NormAll (double value)
        {
            return THDoubleTensor_normall (this.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THDoubledirichlet_grad (HType self, HType x, HType alpha, HType total);
        
        /// <summary>
        ///    DirichletGrad
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="total"></param>
        public DoubleTensor DirichletGrad (DoubleTensor alpha, DoubleTensor total)
        {
            var result = new DoubleTensor();
            THDoubledirichlet_grad (result.handle, this.handle, alpha.handle, total.handle);
            return result;
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
        extern static long THFloatTensor_numel (HType handle);
     
        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumElements ()
        {
            return THFloatTensor_numel (handle);
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
        extern static void THFloatTensor_nonzero (LongTensor.HType subscript, HType handle);
     
        /// <summary>
        ///  Finds the indices of all non-zero elements.
        /// </summary>
        public LongTensor NonZero ()
        {
            var result = new LongTensor();
            THFloatTensor_nonzero (result.handle, this.handle);
            return result;
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
        /// <param name="size3">Size of the fourth dimension.</param>     
        /// <param name="stride3">Stride of the fourth dimension.</param>
        public FloatTensor NewWithStorage4d(IntPtr offset, long size0, long stride0, long size1, long stride1, long size2, long stride2, long size3, long stride3)
        {
            return new FloatTensor(THFloatTensor_newWithStorage4d(Storage.handle, offset, size0, stride0, size1, stride1, size2, stride2, size3, stride3));
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_squeeze (HType handle, HType src);
        
        /// <summary>
        ///   Squeeze the tensor, i.e. remove all 1-sized dimensions.   
        /// </summary>
        public void Squeeze ()
        {
            THFloatTensor_squeeze (handle, handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_squeeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Squeeze the tensor, by removing the specified dimension.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to remove.</param>
        public void Squeeze1d (FloatTensor src, int dimension)
        {
            THFloatTensor_squeeze1d (handle, src.handle, dimension);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_unsqueeze1d (HType handle, HType src, int dimension);
        
        /// <summary>
        ///   Unsqueeze the tensor, by inserting the specified dimension of size 1.   
        /// </summary>
        /// <param name="src">The source tensor which contains the data.</param>
        /// <param name="dimension">The dimension to insert.</param>
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
        ///  Populates the tensor with random values using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        public void Random (RandomGenerator source)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_random (handle, source.handle);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_clampedRandom (HType handle, IntPtr thgenerator, long min, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from min to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower limit for the values to be generated</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void ClampedRandom (RandomGenerator source, long min, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_clampedRandom (handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_cappedRandom (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to max, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void CappedRandom (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_cappedRandom (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_randperm (HType handle, IntPtr thgenerator, long max);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="max">The upper limit for the values to be generated</param>
        public void RandPerm (RandomGenerator source, long max)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_randperm (handle, source.handle, max);
        }

        [DllImport ("caffe2")]
        extern static float THFloatTensor_geometric (HType handle, IntPtr thgenerator, double p);
        
        /// <summary>
        ///  Populates the tensor with random values from 0 to n, using the provided random source generator.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
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
        ///   Fills the tensor with values according to a Bernoulli distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="p"></param>
        public void BernoulliTensor (RandomGenerator source, FloatTensor p)
        {
            THFloatTensor_bernoulli_FloatTensor(this.handle, source.handle, p.handle);
        }
#endif

        [DllImport ("caffe2")]
        extern static void THFloatTensor_uniform (HType self, IntPtr thgenerator, double min, double max);

        /// <summary>
        ///   Fills the tensor with values according to a Bernoulli distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="min">The lower bound for the random number range.</param>
        /// <param name="max">The upper bound for the random number range.</param>
        public void Uniform (RandomGenerator source, double min, double max)
        {
            THFloatTensor_uniform(this.handle, source.handle, min, max);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_exponential (HType self, IntPtr thgenerator, double lambda);

        /// <summary>
        ///   Fills the tensor with values according to a exponential distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="lambda"></param>
        public void Exponential (RandomGenerator source, double lambda)
        {
            THFloatTensor_exponential(this.handle, source.handle, lambda);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_cauchy (HType self, IntPtr thgenerator, double median, double sigma);

        /// <summary>
        ///   Fills the tensor with values according to a Cauchy-Lorentz distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="median"></param>
        /// <param name="sigma"></param>
        public void Cauchy (RandomGenerator source, double median, double sigma)
        {
            THFloatTensor_cauchy(this.handle, source.handle, median, sigma);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_logNormal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Fills the tensor with values according to a log-normal distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdv">The standard deviation of the distribution.</param>
        public void LogNormal (RandomGenerator source, double mean, double stdv)
        {
            THFloatTensor_logNormal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal (HType self, IntPtr thgenerator, double mean, double stdv);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdv">The standard deviation of the distribution.</param>
        public void Normal (RandomGenerator source, double mean, double stdv)
        {
            THFloatTensor_normal(this.handle, source.handle, mean, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal_means (HType self, IntPtr thgenerator, HType means, double stdv);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution. This version uses multiple means.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="means">The means of the distribution.</param>
        /// <param name="stdv">The standard deviation of the distribution.</param>
        public void NormalMeans (RandomGenerator source, FloatTensor means, double stdv)
        {
            THFloatTensor_normal_means(this.handle, source.handle, means.handle, stdv);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal_stddevs (HType self, IntPtr thgenerator, double mean, HType stdvs);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution. This version uses multiple standard deviations.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdvs">The standard deviations of the distribution.</param>
        public void NormalStdvs (RandomGenerator source, double mean, FloatTensor stdvs)
        {
            THFloatTensor_normal_stddevs(this.handle, source.handle, mean, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_normal_means_stddevs (HType self, IntPtr thgenerator, HType means, HType stdvs);

        /// <summary>
        ///   Fills the tensor with values according to a normal distribution. This version uses multiple means and standard deviations.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="means">The means of the distribution.</param>
        /// <param name="stdvs">The standards deviation of the distribution.</param>
        public void NormalMeansStdvs (RandomGenerator source, FloatTensor means, FloatTensor stdvs)
        {
            THFloatTensor_normal_means_stddevs(this.handle, source.handle, means.handle, stdvs.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_multinomial (HType self, IntPtr thgenerator, HType prob_dist, int n_sample, int with_replacement);

        /// <summary>
        ///   Fills the tensor with values according to a multinomial distribution.
        /// </summary>
        /// <param name="source">The random generator source</param>
        /// <param name="prob_dist">The probability for each bucket.</param>
        /// <param name="n_sample">The number of samples to generate.</param>
        /// <param name="with_replacement"></param>
        public void Multinomial (RandomGenerator source, FloatTensor prob_dist, int n_sample, int with_replacement)
        {
            THFloatTensor_multinomial(this.handle, source.handle, prob_dist.handle, n_sample, with_replacement);
        }
        
        /// <summary>
        ///   Get a string representation of the tensor.
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
        public FloatTensor Sigmoid ()
        {
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
        public FloatTensor Log ()
        {
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
        public FloatTensor Lgamma ()
        {
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
        public FloatTensor Digamma ()
        {
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
        public FloatTensor Trigamma ()
        {
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
        public FloatTensor Polygamma ()
        {
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
        public FloatTensor Log10 ()
        {
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
        public FloatTensor Log1p ()
        {
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
        public FloatTensor Log2 ()
        {
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
        public FloatTensor Exp ()
        {
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
        public FloatTensor Expm1 ()
        {
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
        public FloatTensor Cos ()
        {
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
        public FloatTensor Acos ()
        {
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
        public FloatTensor Cosh ()
        {
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
        public FloatTensor Sin ()
        {
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
        public FloatTensor Asin ()
        {
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
        public FloatTensor Sinh ()
        {
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
        public FloatTensor Tan ()
        {
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
        public FloatTensor Atan ()
        {
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
        public FloatTensor Atan2 ()
        {
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
        public FloatTensor Tanh ()
        {
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
        public FloatTensor Erf ()
        {
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
        public FloatTensor Erfc ()
        {
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
        public FloatTensor Erfinv ()
        {
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
        public FloatTensor Sqrt ()
        {
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
        public FloatTensor Rsqrt ()
        {
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
        public FloatTensor Ceil ()
        {
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
        public FloatTensor Floor ()
        {
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
        public FloatTensor Round ()
        {
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
        public FloatTensor Abs ()
        {
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
        public FloatTensor Trunc ()
        {
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
        public FloatTensor Frac ()
        {
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
        public FloatTensor cinv ()
        {
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
        public FloatTensor neg ()
        {
            var result = new FloatTensor ();
            THFloatTensor_neg (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_zerosLike (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the ZerosLike of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor ZerosLike ()
        {
            var result = new FloatTensor ();
            THFloatTensor_zerosLike (result.handle, this.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_onesLike (HType result, HType t);
        
        /// <summary>
        ///   Returns a new tensor with the OnesLike of the elements of <see paramref="src"/>
        /// </summary>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor OnesLike ()
        {
            var result = new FloatTensor ();
            THFloatTensor_onesLike (result.handle, this.handle);
            return result;
        }


                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_pow (HType result, HType x, float y);

        /// <summary>
        ///   Returns a new tensor with <see paramref="this"/> raised to the power of <see paramref="y"/>.
        /// </summary>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        ///   Match
        /// </summary>
        /// <param name="m2"></param>
        /// <param name="gain"></param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        /// <param name="src">The right-hand-side operand.</param>
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
        extern static void THFloatTensor_cmax (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMax of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CMax (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cmax (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cmin (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an CMin of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CMin (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_cmin (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_ltTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtTensor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THFloatTensor_ltTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_leTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeTensor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THFloatTensor_leTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_gtTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtTensor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THFloatTensor_gtTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_geTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeTensor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THFloatTensor_geTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_eqTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqTensor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THFloatTensor_eqTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_neTensor (ByteTensor.HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensor of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeTensor (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new ByteTensor ();
            THFloatTensor_neTensor (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_ltTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor LtTensorT (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_ltTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_leTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an LeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor LeTensorT (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_leTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_gtTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GtTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor GtTensorT (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_gtTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_geTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an GeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor GeTensorT (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_geTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_eqTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an EqTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor EqTensorT (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_eqTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_neTensorT (HType result, HType t, HType src);
        
        /// <summary>
        ///   Performs an NeTensorT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor NeTensorT (FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            var result = new FloatTensor ();
            THFloatTensor_neTensorT (result.handle, this.handle, src.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cmaxvalue (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an CMaxValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CMaxValue (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_cmaxvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_cminvalue (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an CMinValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor CMinValue (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_cminvalue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_ltValue (ByteTensor.HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an LtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LtValue (float src)
        {
            var result = new ByteTensor ();
            THFloatTensor_ltValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_leValue (ByteTensor.HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an LeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor LeValue (float src)
        {
            var result = new ByteTensor ();
            THFloatTensor_leValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_gtValue (ByteTensor.HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an GtValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GtValue (float src)
        {
            var result = new ByteTensor ();
            THFloatTensor_gtValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_geValue (ByteTensor.HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an GeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor GeValue (float src)
        {
            var result = new ByteTensor ();
            THFloatTensor_geValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_eqValue (ByteTensor.HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an EqValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor EqValue (float src)
        {
            var result = new ByteTensor ();
            THFloatTensor_eqValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_neValue (ByteTensor.HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an NeValue of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public ByteTensor NeValue (float src)
        {
            var result = new ByteTensor ();
            THFloatTensor_neValue (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_ltValueT (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an LtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor LtValueT (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_ltValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_leValueT (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an LeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor LeValueT (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_leValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_gtValueT (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an GtValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor GtValueT (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_gtValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_geValueT (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an GeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor GeValueT (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_geValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_eqValueT (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an EqValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor EqValueT (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_eqValueT (result.handle, this.handle, src);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_neValueT (HType result, HType t, float value);
        
        /// <summary>
        ///   Performs an NeValueT of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="src">The right-hand-side operand.</param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor NeValueT (float src)
        {
            var result = new FloatTensor ();
            THFloatTensor_neValueT (result.handle, this.handle, src);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_lerp (HType result, HType self, HType other, float weight);
        
        /// <summary>
        ///   LERP
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        /// <param name="weight"></param>
        public FloatTensor LERP (FloatTensor other, float weight)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            var result = new FloatTensor();
            THFloatTensor_lerp (result.handle, this.handle, other.handle, weight);
            return result;
        }

        [DllImport ("caffe2")]
        extern static int THFloatTensor_equal (HType t, HType src);
        
        /// <summary>
        ///   Compare the tensor with another for complete equality.
        /// </summary>
        /// <param name="other">The right-hand-side operand.</param>
        public int Equal (FloatTensor other)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THFloatTensor_equal (this.handle, other.handle);
        }
                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_add_scaled (HType result, HType t, float value1, float value2);
        
        /// <summary>
        ///   Performs an AddScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor AddScaled (float value1, float value2)
        {
            var result = new FloatTensor ();
            THFloatTensor_add_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sub_scaled (HType result, HType t, float value1, float value2);
        
        /// <summary>
        ///   Performs an SubScaled of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor SubScaled (float value1, float value2)
        {
            var result = new FloatTensor ();
            THFloatTensor_sub_scaled (result.handle, this.handle, value1, value2);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_clamp (HType result, HType t, float value1, float value2);
        
        /// <summary>
        ///   Performs an Clamp of the tensor with the provided 
        ///   <see paramref="src"/> tensor and returns a new tensor with the result.
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Clamp (float value1, float value2)
        {
            var result = new FloatTensor ();
            THFloatTensor_clamp (result.handle, this.handle, value1, value2);
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
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for matxvec (α)</param>
        /// <param name="mat">Matrix to be multiplied</param>
        /// <param name="vec">Vector to be multiplied</param>
        /// <remarks>
        /// β tensor+α (mat@vec)

        /// </remarks>
        /// <returns>
        ///   β tensor+α (mat@vec)
        /// </returns>
        public FloatTensor AddMV (float beta, float alpha, FloatTensor mat, FloatTensor vec)
        {
            if (mat == null)
                throw new ArgumentNullException (nameof (mat));
            if (vec == null)
                throw new ArgumentNullException (nameof (vec));
            var result = new FloatTensor ();
            THFloatTensor_addmv (result.handle, beta, this.handle, alpha, mat.handle, vec.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addmm (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for mat1xmat2 (α)</param>
        /// <param name="mat1">First matrix to  be multiplied</param>
        /// <param name="mat2">Second matrix to  be multiplied</param>
        /// <remarks>
        /// β mat+α (mat1i@mat2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (mat1i@mat2i)
        /// </returns>
        public FloatTensor AddMM (float beta, float alpha, FloatTensor mat1, FloatTensor mat2)
        {
            if (mat1 == null)
                throw new ArgumentNullException (nameof (mat1));
            if (mat2 == null)
                throw new ArgumentNullException (nameof (mat2));
            var result = new FloatTensor ();
            THFloatTensor_addmm (result.handle, beta, this.handle, alpha, mat1.handle, mat2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addbmm (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mat+α (∑i=0bbatch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mat+α (∑i=0bbatch1i@batch2i)
        /// </returns>
        public FloatTensor AddBMM (float beta, float alpha, FloatTensor batch1, FloatTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new FloatTensor ();
            THFloatTensor_addbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_addr (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs AddR of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for vec1xvec2 (α)</param>
        /// <param name="vec1">the first vector of the outer product</param>
        /// <param name="vec2">the second vector of the outer product</param>
        /// <remarks>
        /// β mat+α (vec1⊗vec2)

        /// </remarks>
        /// <returns>
        ///   β mat+α (vec1⊗vec2)
        /// </returns>
        public FloatTensor AddR (float beta, float alpha, FloatTensor vec1, FloatTensor vec2)
        {
            if (vec1 == null)
                throw new ArgumentNullException (nameof (vec1));
            if (vec2 == null)
                throw new ArgumentNullException (nameof (vec2));
            var result = new FloatTensor ();
            THFloatTensor_addr (result.handle, beta, this.handle, alpha, vec1.handle, vec2.handle);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_baddbmm (HType result, float beta, HType t, float alpha, HType src1, HType src2);
        
        /// <summary>
        ///   Performs BAddBMM of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="beta">Multiplier for this tensor (β).</param>
        /// <param name="alpha">Multiplier for batch1xbatch2 (α)</param>
        /// <param name="batch1">the first batch of matrices to be multiplied</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        /// <remarks>
        /// β mati+α (batch1i@batch2i)

        /// </remarks>
        /// <returns>
        ///   β mati+α (batch1i@batch2i)
        /// </returns>
        public FloatTensor BAddBMM (float beta, float alpha, FloatTensor batch1, FloatTensor batch2)
        {
            if (batch1 == null)
                throw new ArgumentNullException (nameof (batch1));
            if (batch2 == null)
                throw new ArgumentNullException (nameof (batch2));
            var result = new FloatTensor ();
            THFloatTensor_baddbmm (result.handle, beta, this.handle, alpha, batch1.handle, batch2.handle);
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
        extern static void THFloatTensor_linspace (HType result, float a, float b, long n);
        
        /// <summary>
        ///   Performs Linspace of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="n"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Linspace (float a, float b, long n)
        {
            var result = new FloatTensor ();
            THFloatTensor_linspace (result.handle, a, b, n);
            return result;
        }

                
        [DllImport ("caffe2")]
        extern static void THFloatTensor_logspace (HType result, float a, float b, long n);
        
        /// <summary>
        ///   Performs Logspace of the tensor with the provided 
        ///   <see paramref="src1"/> and <see paramref="src1"/> tensors and returns a new tensor with the result.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="n"></param>
        /// <returns>
        ///   This returns a new tensor with the same shape as the tensor this operates on.
        /// </returns>
        public FloatTensor Logspace (float a, float b, long n)
        {
            var result = new FloatTensor ();
            THFloatTensor_logspace (result.handle, a, b, n);
            return result;
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
        extern static void THFloatTensor_indexAdd (HType tensor, int dim, LongTensor.HType index, HType src);
        
        /// <summary>
        ///   Adds the elements of tensor into the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the add</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="src">Tensor to copy the data from.</param>
        public void IndexAdd (int dim, LongTensor index, FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_indexAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_indexFill (HType tensor, int dim, LongTensor.HType index, float value);
        
        /// <summary>
        ///   Uses the given value to overwrite the original tensor by selecting the indices in the order 
        ///   given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
        /// </summary>
        /// <param name="dim">Dimension to select for the fill</param>
        /// <param name="index">Entries to copy</param>
        /// <param name="value">The value to write.</param>
        public void IndexFill (int dim, LongTensor index, float value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_indexFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_take (HType self, HType src, LongTensor.HType index);

        /// <summary>
        ///   Take
        /// </summary>        
        /// <param name="src"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Take (FloatTensor src, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_take (handle, src.handle, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_put (HType self, LongTensor.HType index, HType src, int accumulate);

        /// <summary>
        ///   Put
        /// </summary>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        /// <param name="accumulate"></param>
        public void Put (LongTensor index, FloatTensor src, int accumulate)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_put (handle, index.handle, src.handle, accumulate);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_gather (HType self, HType src, int dim, LongTensor.HType index);

        /// <summary>
        ///   Gather
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        public void Gather (FloatTensor src, int dim, LongTensor index)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_gather (handle, src.handle, dim, index.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_scatter (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   Scatter
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void Scatter (int dim, LongTensor index, FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_scatter (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_scatterAdd (HType self, int dim, LongTensor.HType index, HType src);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="src"></param>
        public void ScatterAdd (int dim, LongTensor index, FloatTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_scatterAdd (handle, dim, index.handle, src.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_scatterFill (HType self, int dim, LongTensor.HType index, float value);

        /// <summary>
        ///   ScatterAdd
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="index">Indices of entries to copy.</param>
        /// <param name="value"></param>
        public void ScatterFill (int dim, LongTensor index, float value)
        {
            if (index == null)
                throw new ArgumentNullException (nameof (index));
            THFloatTensor_scatterFill (handle, dim, index.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_copy (HType tensor, HType src);
        
        /// <summary>
        ///   Copies the elements of a tensor into the original tensor. 
        ///   The shape of the tensors must exactly match or an error will be thrown.
        /// </summary>
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
        /// <param name="src">Tensor to copy the data from.</param>
        public void CopyDouble (DoubleTensor src)
        {
            if (src == null)
                throw new ArgumentNullException (nameof (src));
            THFloatTensor_copyDouble (this.handle, src.handle);
        }

        
        
     
        [DllImport ("caffe2")]
        extern static void THFloatTensor_sum (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public FloatTensor Sum (int dimension, int keepdim)
        {
            var result = new FloatTensor ();
            THFloatTensor_sum (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_cumsum (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative sum of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public FloatTensor CumulativeSum (int dimension)
        {
            var result = new FloatTensor ();
            THFloatTensor_cumsum (result.handle, this.handle, dimension);
            return result;
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_prod (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public FloatTensor Prod (int dimension, int keepdim)
        {
            var result = new FloatTensor ();
            THFloatTensor_prod (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_cumprod (HType result, HType self, int dimension);
        
        /// <summary>
        ///   Computes the cumulative product of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        public FloatTensor CumulativeProd (int dimension)
        {
            var result = new FloatTensor ();
            THFloatTensor_cumprod (result.handle, this.handle, dimension);
            return result;
        }
     
        [DllImport ("caffe2")]
        extern static void THFloatTensor_max (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the max of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<FloatTensor, LongTensor> Max (int dimension, int keepdim)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_max (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_min (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the min of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<FloatTensor, LongTensor> Min (int dimension, int keepdim)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_min (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_mode (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the mode of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<FloatTensor, LongTensor> Mode (int dimension, int keepdim)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_mode (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }
        [DllImport ("caffe2")]
        extern static void THFloatTensor_median (HType values, LongTensor.HType indices, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the median of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the resulting elements.</returns>
        public System.Tuple<FloatTensor, LongTensor> Median (int dimension, int keepdim)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_median (values.handle, indices.handle, this.handle, dimension, keepdim);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }
     

        [DllImport ("caffe2")]
        extern static void THFloatTensor_kthvalue (HType values, LongTensor.HType indices, HType self, long k, int dimension, int keepdim);
        
        /// <summary>
        ///   Computes the kth value of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The value for 'k' in 'kth'.</param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        /// <returns>A tuple containing the values and indices of the kth element of each dimension.</returns>
        public System.Tuple<FloatTensor, LongTensor> KthValue (long k, int dimension, int keepdim)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_kthvalue (values.handle, indices.handle, this.handle, k, dimension, keepdim);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static double THFloatTensor_trace (HType self);
        
        /// <summary>
        ///   Computes the trace of the tensor. 
        /// </summary>
        public double Trace ()
        {
            return THFloatTensor_trace(this.handle);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_sign (HType result, HType self);
        
        /// <summary>
        ///   Computes the sign of the tensor. 
        /// </summary>
        public FloatTensor Sign ()
        {
            var result = new FloatTensor();
            THFloatTensor_sign(result.handle, this.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_cross (HType result, HType a, HType b);
        
        /// <summary>
        ///   Computes the cross product of two tensors. 
        /// </summary>
        /// <param name="other">The right-hand-side tensor.</param>
        public FloatTensor CrossProduct (FloatTensor other)
        {
            var result = new FloatTensor();
            THFloatTensor_cross(result.handle, this.handle, other.handle);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_diag (HType result, HType self, int k);
        
        /// <summary>
        ///   Gets the diagonal of the tensor. 
        /// </summary>
        /// <param name="k"></param>
        public FloatTensor Diagonal (int k)
        {
            var result = new FloatTensor();
            THFloatTensor_diag(result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_eye (HType result, long m, long n);
        
        /// <summary>
        ///   Eye. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static FloatTensor Eye (long m, long n)
        {
            var result = new FloatTensor();
            THFloatTensor_eye(result.handle, m, n);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_range (HType result, double xmin, double xmax, double step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static FloatTensor Range (double xmin, double xmax, double step)
        {
            var result = new FloatTensor();
            THFloatTensor_range(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_arange (HType result, double xmin, double xmax, double step);
        
        /// <summary>
        ///   Create a range spanning from xmin to xmax, with 'step' between each value.
        /// </summary>
        /// <param name="xmin">The lower bound of the range.</param>
        /// <param name="xmax">The upper bound of the range.</param>
        /// <param name="step">The value step.</param>
        public static FloatTensor ARange (double xmin, double xmax, double step)
        {
            var result = new FloatTensor();
            THFloatTensor_arange(result.handle, xmin, xmax, step);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_sort (HType values, LongTensor.HType indices, HType self, int dimension, int descending);
        
        /// <summary>
        ///   Sorts the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to sort along.</param>
        /// <param name="descending">0 if ascending, 1 if descending.</param>
        /// <returns>A tuple containing the values and indices of the sorted elements.</returns>
        public System.Tuple<FloatTensor, LongTensor> Sort (int dimension, int descending)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_sort (values.handle, indices.handle, this.handle, dimension, descending);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_topk (HType values, LongTensor.HType indices, HType self, long k, int dim, int dir, int sorted);
        
        /// <summary>
        ///   Finds the top k of all the elements of the tensor along the given dimension. 
        /// </summary>
        /// <param name="k">The number of elements to fetch.</param>
        /// <param name="dim">The dimension along which to sort and find k elements.</param>
        /// <param name="dir">0 if ascending, 1 if descending.</param>
        /// <param name="sorted">1 if the result should be sorted, 0 if they should keep their original order.</param>
        /// <returns>A tuple containing the values and indices of the top 'k' elements.</returns>
        public System.Tuple<FloatTensor, LongTensor> TopK (long k, int dim, int dir, int sorted)
        {
            var values = new FloatTensor ();
            var indices = new LongTensor ();
            THFloatTensor_topk (values.handle, indices.handle, this.handle, k, dim, dir, sorted);
            return new System.Tuple<FloatTensor, LongTensor>(values, indices);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_tril (HType result, HType self, long k);
        
        /// <summary>
        ///   Lower triangle. 
        /// </summary>
        /// <param name="k"></param>
        public FloatTensor TriL (long k)
        {
            var result = new FloatTensor ();
            THFloatTensor_tril (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_triu (HType result, HType self, long k);
        
        /// <summary>
        ///   Upper triangle. 
        /// </summary>
        /// <param name="k"></param>
        public FloatTensor TriU (long k)
        {
            var result = new FloatTensor ();
            THFloatTensor_triu (result.handle, this.handle, k);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_cat (HType result, HType ta, HType tb, int dimension);
        
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="other">The second tensor.</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public FloatTensor Concatenate (FloatTensor other, int dimension)
        {
            var result = new FloatTensor ();
            THFloatTensor_cat (result.handle, this.handle, other.handle, dimension);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_catArray (HType result, HType[] ta, int count, int dimension);
#if false        
// NOTE: We need to determine the right marshalling for an array of handles.
        /// <summary>
        ///   Concatenate tensors along the given dimesion.
        /// </summary>
        /// <param name="tensors">A collection of tensors..</param>
        /// <param name="dimension">The dimension along which to concatenate.</param>
        public static FloatTensor Concatenate (IEnumerable<FloatTensor> tensors, int dimension)
        {
            var result = new FloatTensor ();
            var handleArray = tensors.Select(t => t.handle).ToArray();
            THFloatTensor_catArray (result.handle, handleArray, (int)handleArray.Length, dimension);
            return result;
        }
#endif
        [DllImport ("caffe2")]
        extern static void THFloatTensor_mean (HType result, HType self, int dimension, int keepdim);
        
        /// <summary>
        ///   Compute the mean of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public FloatTensor Mean (int dimension, int keepdim)
        {
            var result = new FloatTensor();
            THFloatTensor_mean (result.handle, this.handle, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_std (HType result, HType self, int dimension, int biased, int keepdim);

        /// <summary>
        ///   Compute the standard deviation of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="biased"></param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public FloatTensor Std (int dimension, int biased, int keepdim)
        {
            var result = new FloatTensor();
            THFloatTensor_std (result.handle, this.handle, dimension, biased, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_var (HType result, HType self, int dimension, int biased, int keepdim);

        /// <summary>
        ///   Compute the variance of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="biased"></param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public FloatTensor Var (int dimension, int biased, int keepdim)
        {
            var result = new FloatTensor();
            THFloatTensor_var (result.handle, this.handle, dimension, biased, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_norm (HType result, HType self, float value,  int dimension, int keepdim);

        /// <summary>
        ///   Compute the norm of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="keepdim">1 if the reduction dimension should be kept, 0 otherwise.</param>
        public FloatTensor Norm (float value,  int dimension, int keepdim)
        {
            var result = new FloatTensor();
            THFloatTensor_norm (result.handle, this.handle, value, dimension, keepdim);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_renorm (HType result, HType self, float value,  int dimension, float maxnorm);

        /// <summary>
        ///   Compute the renorm of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dimension">The dimension to process along.</param>
        /// <param name="maxnorm"></param>
        public FloatTensor Renorm (float value,  int dimension, float maxnorm)
        {
            var result = new FloatTensor();
            THFloatTensor_renorm (result.handle, this.handle, value, dimension, maxnorm);
            return result;
        }

        [DllImport ("caffe2")]
        extern static double THFloatTensor_dist (HType a, HType b, float value);

        /// <summary>
        ///   Compute the dist of all tensor elements along the given dimension. 
        /// </summary>
        /// <param name="other">The other tensor.</param>
        /// <param name="value"></param>
        public double Dist (FloatTensor other, float value)
        {
            if (other == null)
                throw new ArgumentNullException (nameof (other));
            return THFloatTensor_dist (this.handle, other.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_histc (HType hist, HType self, long nbins, float minvalue, float maxvalue);

        /// <summary>
        ///   Create a histogram of all tensor elements. 
        /// </summary>
        /// <param name="nbins">The number of bins in the output histogram.</param>
        /// <param name="minvalue">Only consider values equal to or greater than this.</param>
        /// <param name="maxvalue">Only consider values equal to or less than this.</param>
        public FloatTensor Histc (long nbins, float minvalue, float maxvalue)
        {
            var result = new FloatTensor();
            THFloatTensor_histc (result.handle, this.handle, nbins, minvalue, maxvalue);
            return result;
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_bhistc (HType hist, HType self, long nbins, float minvalue, float maxvalue);

        /// <summary>
        ///   Create a histogram of all tensor elements. 
        /// </summary>
        /// <param name="nbins">The number of bins in the output histogram.</param>
        /// <param name="minvalue">Only consider values equal to or greater than this.</param>
        /// <param name="maxvalue">Only consider values equal to or less than this.</param>
        public FloatTensor BHistc (long nbins, float minvalue, float maxvalue)
        {
            var result = new FloatTensor();
            THFloatTensor_bhistc (result.handle, this.handle, nbins, minvalue, maxvalue);
            return result;
        }

        [DllImport ("caffe2")]
        extern static double THFloatTensor_meanall (HType self);

        /// <summary>
        ///   Compute the mean of all tensor elements. 
        /// </summary>
        public double MeanAll ()
        {
            return THFloatTensor_meanall (this.handle);
        }

        [DllImport ("caffe2")]
        extern static double THFloatTensor_varall (HType self, int biased);

        /// <summary>
        ///   Compute the variance of all tensor elements. 
        /// </summary>
        /// <param name="biased"></param>
        public double VarAll (int biased)
        {
            return THFloatTensor_varall (this.handle, biased);
        }

        [DllImport ("caffe2")]
        extern static double THFloatTensor_stdall (HType self, int biased);

        /// <summary>
        ///   Compute the standard deviation of all tensor elements. 
        /// </summary>
        /// <param name="biased"></param>
        public double StdAll (int biased)
        {
            return THFloatTensor_stdall (this.handle, biased);
        }

        [DllImport ("caffe2")]
        extern static double THFloatTensor_normall (HType self, float value);

        /// <summary>
        ///   Compute the norm of all tensor elements. 
        /// </summary>
        /// <param name="value"></param>
        public double NormAll (float value)
        {
            return THFloatTensor_normall (this.handle, value);
        }

        [DllImport ("caffe2")]
        extern static void THFloatdirichlet_grad (HType self, HType x, HType alpha, HType total);
        
        /// <summary>
        ///    DirichletGrad
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="total"></param>
        public FloatTensor DirichletGrad (FloatTensor alpha, FloatTensor total)
        {
            var result = new FloatTensor();
            THFloatdirichlet_grad (result.handle, this.handle, alpha.handle, total.handle);
            return result;
        }
     
    }
}