using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

namespace TorchSharp
{
    using static TensorExtensionMethods;

    public static partial class torch
    {
        /// <summary>
        /// Get the current default floating point torch.dtype.
        /// </summary>
        /// <returns></returns>
        public static ScalarType get_default_dtype() => default_dtype;

        /// <summary>
        /// Sets the default floating point dtype to d. This dtype is:
        /// 1. The inferred dtype for python floats in torch.tensor().
        /// 2. Used to infer dtype for python complex numbers.
        ///    The default complex dtype is set to torch.complex128 if default floating point dtype is torch.float64, otherwise it’s set to torch.complex64
        ///    The default floating point dtype is initially torch.float32.
        /// </summary>
        /// <param name="dtype"></param>
        public static void set_default_dtype(ScalarType dtype) { default_dtype = dtype; }


        // arange()

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                if (start.Type.IsIntegral() && stop.Type.IsIntegral() && step.Type.IsIntegral()) {
                    dtype = ScalarType.Int64;
                }
                else {
                    dtype = get_default_dtype();
                }
            }

            var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, dtype, device, requiresGrad);
        }

        // randperm()

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            dtype = dtype.HasValue ? dtype : ScalarType.Int64;

            var handle = THSTensor_randperm(n, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm(n, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }


        // zeros()

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_zeros((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Returns a tensor filled with the scalar value 0, with the same size as input.
        /// </summary>
        public static Tensor zeros_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.zeros_like(dtype, device, requiresGrad);


        // ones()

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_ones((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Returns a tensor filled with the scalar value 1, with the same size as input.
        /// </summary>
        public static Tensor ones_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.ones_like(dtype, device, requiresGrad);


        // empty()

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with empty
        /// </summary>
        static public Tensor empty(long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with empty
        /// </summary>
        static public Tensor empty(long size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with empty
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with empty
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with empty
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Returns a tensor filled with uninitialized data, with the same size as input.
        /// </summary>
        public static Tensor empty_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.empty_like(dtype, device, requiresGrad);


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size, pstrides = strides) {
                    var handle = THSTensor_empty_strided((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                if (value.Type.IsIntegral()) {
                    dtype = ScalarType.Int64;
                } else {
                    dtype = get_default_dtype();
                }
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_full((IntPtr)psizes, size.Length, value.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full((IntPtr)psizes, size.Length, value.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Returns a tensor with the same size as input filled with 'value.'
        /// </summary>
        static public Tensor full_like(Tensor input, Scalar value, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.full_like(value, dtype, device, requiresGrad);


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye(rows, columns, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye(rows, columns, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            //TODO: fix randint to take a low bound, too.

            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            if (dtype == ScalarType.ComplexFloat32) {

                return randint_c32(max, size, device, requiresGrad);
            }
            else if (dtype == ScalarType.ComplexFloat64) {
                return randint_c64(max, size, device, requiresGrad);
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_randint(max, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(max, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        static private Tensor randint_c32(long max, long[] size, torch.Device device, bool requiresGrad)
        {
            var sz = new List<long>();
            sz.AddRange(size);
            sz.Add(2);
            var size2 = sz.ToArray();

            unsafe {
                fixed (long* psizes = size2) {
                    //
                    // This is a little roundabout -- the native library doesn't support 'randint' for complex types,
                    // but we can get around that by adding another dimension, creating a float tensor, and then
                    // converting it to a complex tensor in the end.
                    //
                    var handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        torch.CheckForErrors();

                    //
                    // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                    //
                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int)device.type, device.index, requiresGrad);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();

                    res = THSTensor_copy_(res, cmplx, false);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    return new Tensor(res);
                }
            }
        }

        static private Tensor randint_c64(long max, long[] size, torch.Device device, bool requiresGrad)
        {
            var sz = new List<long>();
            sz.AddRange(size);
            sz.Add(2);
            var size2 = sz.ToArray();

            unsafe {
                fixed (long* psizes = size2) {
                    //
                    // This is a little roundabout -- the native library doesn't support 'randint' for complex types,
                    // but we can get around that by adding another dimension, creating a float tensor, and then
                    // converting it to a complex tensor in the end.
                    //
                    var handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        torch.CheckForErrors();

                    //
                    // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                    //
                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int)device.type, device.index, requiresGrad);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();

                    res = THSTensor_copy_(res, cmplx, false);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    return new Tensor(res);
                }
            }
        }

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_rand((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_randn((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(bool scalar, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newBoolScalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is not null) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(byte scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newByteScalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(sbyte scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt8Scalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(short scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt16Scalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(int scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt32Scalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(long scalar, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt64Scalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is not null) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(float scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newFloat32Scalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(double scalar, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newFloat64Scalar(scalar, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is not null) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor((float Real, float Imaginary) scalar, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newComplexFloat32Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is not null) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(System.Numerics.Complex scalar, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is not null) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<bool> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Bool, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Bool, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                }
                else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<bool> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<bool> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<bool> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<bool> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<byte> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Byte, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Byte, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<byte> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<byte> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<byte> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<byte> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<sbyte> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int8, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int8, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<sbyte> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<sbyte> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<sbyte> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<sbyte> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<short> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int16, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int16, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<short> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<short> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<short> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<short> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<int> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int32, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int32, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<int> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<int> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<int> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(int[] rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<long> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_newInt64(dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_newInt64(dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<long> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<long> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<long> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<long> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<float> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float32, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float32, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<float> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<float> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<float> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<float> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<double> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float64, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float64, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<double> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<double> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<double> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<double> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat32, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat32, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long[] dimensions, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray.ToArray();
            unsafe {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                    });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat64, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat64, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                var tensor = new Tensor(handle);
                if (device is not null) {
                    tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
                } else if (dtype.HasValue) {
                    tensor = tensor.to_type(dtype.Value);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { (long)rawArray.Count }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long rows, long columns, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { rows, columns }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            return tensor(rawArray, new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = values.dtype;
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_sparse(indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse(indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }


        /// <summary>
        /// onstructs a complex tensor with its real part equal to real and its imaginary part equal to imag.
        /// </summary>
        static public Tensor complex(Tensor real, Tensor imag)
        {
            var res = THSTensor_complex(real.Handle, imag.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new Tensor(res);
        }

        /// <summary>
        /// Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value 'abs' and angle 'angle'.
        /// </summary>
        static public Tensor polar(Tensor abs, Tensor angle)
        {
            var res = THSTensor_polar(abs.Handle, angle.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new Tensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public Tensor linspace(double start, double end, long steps, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var handle = THSTensor_linspace(start, end, steps, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace(start, end, steps, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public Tensor logspace(double start, double end, long steps, double @base = 10, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var handle = THSTensor_logspace(start, end, steps, @base, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace(start, end, steps, @base, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bartlett_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Bartlett window function.
        /// </summary>
        static public Tensor bartlett_window(long len, bool periodic = true, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_bartlett_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_bartlett_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_blackman_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Blackman window function.
        /// </summary>
        static public Tensor blackman_window(long len, bool periodic = true, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_blackman_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_blackman_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hamming_window(long len, bool periodic, double alpha, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hamming window function.
        /// </summary>
        static public Tensor hamming_window(long len, bool periodic = true, float alpha = 0.54f, float beta = 0.46f, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_hamming_window(len, periodic, alpha, beta, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hamming_window(len, periodic, alpha, beta, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hann_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hann window function.
        /// </summary>
        static public Tensor hann_window(long len, bool periodic = true, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_hann_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hann_window(len, periodic, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_kaiser_window(long len, bool periodic, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Computes the Kaiser window with window length window_length and shape parameter beta.
        /// </summary>
        static public Tensor kaiser_window(long len, bool periodic = true, float beta = 12.0f, torch.ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }
            if (!dtype.Value.IsFloatingPoint()) throw new ArgumentException("Only floating point types are supported.");

            var handle = THSTensor_kaiser_window(len, periodic, beta, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_kaiser_window(len, periodic, beta, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBoolScalar(bool scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt8Scalar(sbyte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt16Scalar(short scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt32Scalar(int scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64Scalar(long scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat32Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat64Scalar(double scalar, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newComplexFloat32Scalar(float real, float imaginary, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newComplexFloat64Scalar(double real, double imaginary, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_view_as_complex(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, bool non_blocking);

        [DllImport("LibTorchSharp")]
        extern static void THSTensor_dispose(IntPtr handle);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_complex(IntPtr real, IntPtr imag);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_polar(IntPtr abs, IntPtr angle);

        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static private ScalarType default_dtype = ScalarType.Float32;

        static torch()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }
    }
}
