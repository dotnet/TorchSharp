// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a new tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with zeros
        /// </summary>
        public static Tensor zeros(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor filled with the scalar value 0, with the same size as input.
        /// </summary>
        public static Tensor zeros_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.zeros_like(dtype, device, requires_grad);

        private static Tensor _zeros(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {

                IntPtr handle = IntPtr.Zero;

                fixed (long* psizes = size) {

                    handle = THSTensor_zeros((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);

                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
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
