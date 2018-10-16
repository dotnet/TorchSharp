using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using HType=TorchSharp.TorchHandle;
using System.Text;

namespace TorchSharp {

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
        extern static HType THByteStorage_new_withSize (IntPtr size);
        
        public ByteStorage (long size)
        {
            handle = THByteStorage_new_withSize ((IntPtr) size);
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
        extern static void THByteStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THByteStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static byte THByteStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        extern static void THByteStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  byte value);
        
        public byte this [long index] {
            get => THByteStorage_get (handle, (IntPtr) (index));
            set {
                THByteStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static byte THByteStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            THByteStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void THByteStorage_fill (HType handle, byte value);
        
        public void Fill (byte value)
        {
            THByteStorage_fill (handle, value);
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
        extern static void THByteTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THByteTensor_free (handle);
                handle.Dispose ();
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
        extern static HType THByteTensor_storage (HType handle);

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
        
        public ByteTensor Select (int dim, long slideIndex) => new ByteTensor (THByteTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THByteTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public ByteTensor Narrow (int dim, long firstIndex, long size) => new ByteTensor (THByteTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newTranspose (HType handle, int dim1, int dim2);
        public ByteTensor Transpose (int dim1, int dim2) => new ByteTensor (THByteTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THByteTensor_newUnfold (HType handle, int dim1, long size, long step);
        public ByteTensor Unfold (int dim, long size, long step) => new ByteTensor (THByteTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            THByteTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            THByteTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            THByteTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THByteTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THByteTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THByteTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (ByteTensor src)
        {
            THByteTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THByteTensor_set (HType handle, HType src);
        
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
        public byte this [long x0, long x1, long x2, long x3] {
            get => THByteTensor_get4d (handle, x0, x1, x2, x3);
            set => THByteTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static byte THByteTensor_randperm (HType handle, IntPtr thgenerator, long n);
        public void Random (RandomGenerator source, long n)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THByteTensor_randperm (handle, source.handle, n);
        }
        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                Random (r, n);
        }
        
        
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
        #if false
        [DllImport ("caffe2")]
        extern static string THByteTensor_
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
        extern static HType THShortStorage_new_withSize (IntPtr size);
        
        public ShortStorage (long size)
        {
            handle = THShortStorage_new_withSize ((IntPtr) size);
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
        extern static void THShortStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THShortStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static short THShortStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        extern static void THShortStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  short value);
        
        public short this [long index] {
            get => THShortStorage_get (handle, (IntPtr) (index));
            set {
                THShortStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static short THShortStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            THShortStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void THShortStorage_fill (HType handle, short value);
        
        public void Fill (short value)
        {
            THShortStorage_fill (handle, value);
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
        extern static void THShortTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THShortTensor_free (handle);
                handle.Dispose ();
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
        extern static HType THShortTensor_storage (HType handle);

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
        
        public ShortTensor Select (int dim, long slideIndex) => new ShortTensor (THShortTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THShortTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public ShortTensor Narrow (int dim, long firstIndex, long size) => new ShortTensor (THShortTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newTranspose (HType handle, int dim1, int dim2);
        public ShortTensor Transpose (int dim1, int dim2) => new ShortTensor (THShortTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THShortTensor_newUnfold (HType handle, int dim1, long size, long step);
        public ShortTensor Unfold (int dim, long size, long step) => new ShortTensor (THShortTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            THShortTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            THShortTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            THShortTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THShortTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THShortTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THShortTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (ShortTensor src)
        {
            THShortTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THShortTensor_set (HType handle, HType src);
        
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
        public short this [long x0, long x1, long x2, long x3] {
            get => THShortTensor_get4d (handle, x0, x1, x2, x3);
            set => THShortTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static short THShortTensor_randperm (HType handle, IntPtr thgenerator, long n);
        public void Random (RandomGenerator source, long n)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THShortTensor_randperm (handle, source.handle, n);
        }
        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                Random (r, n);
        }
        
        
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
        #if false
        [DllImport ("caffe2")]
        extern static string THShortTensor_
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
        extern static HType THIntStorage_new_withSize (IntPtr size);
        
        public IntStorage (long size)
        {
            handle = THIntStorage_new_withSize ((IntPtr) size);
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
        extern static void THIntStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THIntStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static int THIntStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        extern static void THIntStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  int value);
        
        public int this [long index] {
            get => THIntStorage_get (handle, (IntPtr) (index));
            set {
                THIntStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static int THIntStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            THIntStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void THIntStorage_fill (HType handle, int value);
        
        public void Fill (int value)
        {
            THIntStorage_fill (handle, value);
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
        extern static void THIntTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THIntTensor_free (handle);
                handle.Dispose ();
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
        extern static HType THIntTensor_storage (HType handle);

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
        
        public IntTensor Select (int dim, long slideIndex) => new IntTensor (THIntTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THIntTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public IntTensor Narrow (int dim, long firstIndex, long size) => new IntTensor (THIntTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newTranspose (HType handle, int dim1, int dim2);
        public IntTensor Transpose (int dim1, int dim2) => new IntTensor (THIntTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THIntTensor_newUnfold (HType handle, int dim1, long size, long step);
        public IntTensor Unfold (int dim, long size, long step) => new IntTensor (THIntTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            THIntTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            THIntTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            THIntTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THIntTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THIntTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THIntTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (IntTensor src)
        {
            THIntTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THIntTensor_set (HType handle, HType src);
        
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
        public int this [long x0, long x1, long x2, long x3] {
            get => THIntTensor_get4d (handle, x0, x1, x2, x3);
            set => THIntTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static int THIntTensor_randperm (HType handle, IntPtr thgenerator, long n);
        public void Random (RandomGenerator source, long n)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THIntTensor_randperm (handle, source.handle, n);
        }
        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                Random (r, n);
        }
        
        
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
        #if false
        [DllImport ("caffe2")]
        extern static string THIntTensor_
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
        extern static HType THLongStorage_new_withSize (IntPtr size);
        
        public LongStorage (long size)
        {
            handle = THLongStorage_new_withSize ((IntPtr) size);
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
        extern static void THLongStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THLongStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static long THLongStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        extern static void THLongStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  long value);
        
        public long this [long index] {
            get => THLongStorage_get (handle, (IntPtr) (index));
            set {
                THLongStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static long THLongStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            THLongStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void THLongStorage_fill (HType handle, long value);
        
        public void Fill (long value)
        {
            THLongStorage_fill (handle, value);
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
        extern static void THLongTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THLongTensor_free (handle);
                handle.Dispose ();
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
        extern static HType THLongTensor_storage (HType handle);

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
        
        public LongTensor Select (int dim, long slideIndex) => new LongTensor (THLongTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THLongTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public LongTensor Narrow (int dim, long firstIndex, long size) => new LongTensor (THLongTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newTranspose (HType handle, int dim1, int dim2);
        public LongTensor Transpose (int dim1, int dim2) => new LongTensor (THLongTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THLongTensor_newUnfold (HType handle, int dim1, long size, long step);
        public LongTensor Unfold (int dim, long size, long step) => new LongTensor (THLongTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            THLongTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            THLongTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            THLongTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THLongTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THLongTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THLongTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (LongTensor src)
        {
            THLongTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THLongTensor_set (HType handle, HType src);
        
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
        public long this [long x0, long x1, long x2, long x3] {
            get => THLongTensor_get4d (handle, x0, x1, x2, x3);
            set => THLongTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static long THLongTensor_randperm (HType handle, IntPtr thgenerator, long n);
        public void Random (RandomGenerator source, long n)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THLongTensor_randperm (handle, source.handle, n);
        }
        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                Random (r, n);
        }
        
        
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
        #if false
        [DllImport ("caffe2")]
        extern static string THLongTensor_
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
        extern static HType THDoubleStorage_new_withSize (IntPtr size);
        
        public DoubleStorage (long size)
        {
            handle = THDoubleStorage_new_withSize ((IntPtr) size);
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
        extern static void THDoubleStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THDoubleStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static double THDoubleStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        extern static void THDoubleStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  double value);
        
        public double this [long index] {
            get => THDoubleStorage_get (handle, (IntPtr) (index));
            set {
                THDoubleStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static double THDoubleStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            THDoubleStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleStorage_fill (HType handle, double value);
        
        public void Fill (double value)
        {
            THDoubleStorage_fill (handle, value);
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
        extern static void THDoubleTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THDoubleTensor_free (handle);
                handle.Dispose ();
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
        extern static HType THDoubleTensor_storage (HType handle);

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
        
        public DoubleTensor Select (int dim, long slideIndex) => new DoubleTensor (THDoubleTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public DoubleTensor Narrow (int dim, long firstIndex, long size) => new DoubleTensor (THDoubleTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newTranspose (HType handle, int dim1, int dim2);
        public DoubleTensor Transpose (int dim1, int dim2) => new DoubleTensor (THDoubleTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THDoubleTensor_newUnfold (HType handle, int dim1, long size, long step);
        public DoubleTensor Unfold (int dim, long size, long step) => new DoubleTensor (THDoubleTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            THDoubleTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            THDoubleTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            THDoubleTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THDoubleTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THDoubleTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (DoubleTensor src)
        {
            THDoubleTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THDoubleTensor_set (HType handle, HType src);
        
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
        public double this [long x0, long x1, long x2, long x3] {
            get => THDoubleTensor_get4d (handle, x0, x1, x2, x3);
            set => THDoubleTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static double THDoubleTensor_randperm (HType handle, IntPtr thgenerator, long n);
        public void Random (RandomGenerator source, long n)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THDoubleTensor_randperm (handle, source.handle, n);
        }
        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                Random (r, n);
        }
        
        
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
        #if false
        [DllImport ("caffe2")]
        extern static string THDoubleTensor_
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
        extern static HType THFloatStorage_new_withSize (IntPtr size);
        
        public FloatStorage (long size)
        {
            handle = THFloatStorage_new_withSize ((IntPtr) size);
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
        extern static void THFloatStorage_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THFloatStorage_free (handle);
                handle.Dispose ();
            }
        }
        
        [DllImport ("caffe2")]
        extern static float THFloatStorage_get (HType handle, /*ptrdiff_t*/IntPtr pos);
        extern static void THFloatStorage_set (HType handle, /*ptrdiff_t*/IntPtr pos,  float value);
        
        public float this [long index] {
            get => THFloatStorage_get (handle, (IntPtr) (index));
            set {
                THFloatStorage_set (handle, (IntPtr) (index), value);
            }
        }
        
        [DllImport ("caffe2")]
        extern static float THFloatStorage_resize (HType handle, /*ptrdiff_t*/UIntPtr newSize);
        
        public void Resize (ulong size)
        {
            THFloatStorage_resize (handle, (UIntPtr) size);
        }

        [DllImport ("caffe2")]
        extern static void THFloatStorage_fill (HType handle, float value);
        
        public void Fill (float value)
        {
            THFloatStorage_fill (handle, value);
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
        extern static void THFloatTensor_free (HType handle);
        
        public void Dispose (bool disposing)
        {
            if (disposing){
                THFloatTensor_free (handle);
                handle.Dispose ();
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
        extern static HType THFloatTensor_storage (HType handle);

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
        
        public FloatTensor Select (int dim, long slideIndex) => new FloatTensor (THFloatTensor_newSelect (handle, dim, slideIndex));

        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newNarrow (HType handle, int dim, long firstIndex, long size);
        
        public FloatTensor Narrow (int dim, long firstIndex, long size) => new FloatTensor (THFloatTensor_newNarrow (handle, dim, firstIndex, size));
                
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newTranspose (HType handle, int dim1, int dim2);
        public FloatTensor Transpose (int dim1, int dim2) => new FloatTensor (THFloatTensor_newTranspose (handle, dim1, dim2));
        
        [DllImport ("caffe2")]
        extern static HType THFloatTensor_newUnfold (HType handle, int dim1, long size, long step);
        public FloatTensor Unfold (int dim, long size, long step) => new FloatTensor (THFloatTensor_newUnfold (handle, dim, size, step));
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize1d (HType handle, long size);
        
        public void Resize1d (long size)
        {
            THFloatTensor_resize1d (handle, size);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize2d (HType handle, long size0, long size1);
        public void Resize2d (long size0, long size1)
        {
            THFloatTensor_resize2d (handle, size0, size1);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize3d (HType handle, long size0, long size1, long size2);
        
        public void Resize3d (long size0, long size1, long size2)
        {
            THFloatTensor_resize3d (handle, size0, size1, size2);
        }

        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize4d (HType handle, long size0, long size1, long size2, long size4);
        public void Resize4d (long size0, long size1, long size2, long size3)
        {
            THFloatTensor_resize4d (handle, size0, size1, size2, size3);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resize5d (HType handle, long size0, long size1, long size2, long size4, long size5);

        public void Resize5d (long size0, long size1, long size2, long size3, long size4)
        {
            THFloatTensor_resize5d (handle, size0, size1, size2, size3, size4);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_resizeAs (HType handle, HType src);
       
        public void ResizeAs (FloatTensor src)
        {
            THFloatTensor_resizeAs (handle, src.handle);
        }
        
        [DllImport ("caffe2")]
        extern static void THFloatTensor_set (HType handle, HType src);
        
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
        public float this [long x0, long x1, long x2, long x3] {
            get => THFloatTensor_get4d (handle, x0, x1, x2, x3);
            set => THFloatTensor_set4d (handle, x0, x1, x2, x3, value);
        }
        
        [DllImport ("caffe2")]
        extern static float THFloatTensor_randperm (HType handle, IntPtr thgenerator, long n);
        public void Random (RandomGenerator source, long n)
        {
            if (source == null)
                throw new ArgumentNullException (nameof (source));
            THFloatTensor_randperm (handle, source.handle, n);
        }
        
        public void Random (long n)
        {
            using (var r = new RandomGenerator ())
                Random (r, n);
        }
        
        
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
        #if false
        [DllImport ("caffe2")]
        extern static string THFloatTensor_
        #endif
    }
}