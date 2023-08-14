// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a new tensor filled with ones
        /// </summary>
        public static Tensor ones(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with ones
        /// </summary>
        public static Tensor ones(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor filled with the scalar value 1, with the same size as input.
        /// </summary>
        public static Tensor ones_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.ones_like(dtype, device, requires_grad);

        /// <summary>
        /// Create a new tensor filled with ones
        /// </summary>
        private static Tensor _ones(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_ones((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
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
