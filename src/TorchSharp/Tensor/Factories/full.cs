// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a new tensor filled with a given value
        /// </summary>
        public static Tensor full(long[] size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(size, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with a given value
        /// </summary>
        public static Tensor full(ReadOnlySpan<long> size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(size, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with given value
        /// </summary>
        public static Tensor full(long size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { size }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with given value
        /// </summary>
        public static Tensor full(long rows, long columns, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { rows, columns }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with given value
        /// </summary>
        public static Tensor full(long dim0, long dim1, long dim2, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with given value
        /// </summary>
        public static Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with given value
        /// </summary>
        public static Tensor full(int size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { size }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with given value
        /// </summary>
        public static Tensor full(int rows, int columns, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { rows, columns }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with given value
        /// </summary>
        public static Tensor full(int dim0, int dim1, int dim2, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with given value
        /// </summary>
        public static Tensor full(int dim0, int dim1, int dim2, int dim3, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor with the same size as input filled with 'value.'
        /// </summary>
        public static Tensor full_like(Tensor input, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.full_like(value, dtype, device, requires_grad);

        /// <summary>
        /// Create a new tensor filled with a given value
        /// </summary>
        private static Tensor _full(ReadOnlySpan<long> size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
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
                    var handle = THSTensor_full((IntPtr)psizes, size.Length, value.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full((IntPtr)psizes, size.Length, value.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }
    }
}
