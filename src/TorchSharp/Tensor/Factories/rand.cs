// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [low, high).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(long low, long high, Size size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            var shape = size.Shape;

            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = ScalarType.Int64;
            }

            ValidateIntegerRange(low, dtype.Value, nameof(low));
            ValidateIntegerRange(high - 1, dtype.Value, nameof(high));

            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            Tensor? result = null;

            if (dtype == ScalarType.ComplexFloat32) {
                result = randint_c32(genHandle, low, high, shape, device, requires_grad, names: names);
            } else if (dtype == ScalarType.ComplexFloat64) {
                result = randint_c64(genHandle, low, high, shape, device, requires_grad, names: names);
            }

            if (result is null) {
                unsafe {
                    fixed (long* psizes = shape) {
                        var handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, shape.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                        if (handle == IntPtr.Zero) {
                            GC.Collect();
                            GC.WaitForPendingFinalizers();
                            handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, shape.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                        }
                        if (handle == IntPtr.Zero) { CheckForErrors(); }
                        result = new Tensor(handle);
                    }
                }
            }

            if (names != null && names.Length > 0) {

                result.rename_(names);
            }

            return result;
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [0, high).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(long high, Size size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(0, high, size, dtype, device, requires_grad, generator, names);
        }

        // Note: Once F# implicit conversion support is broadly available, all the following overloads will be redundant.

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [0, high).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(long high, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(0, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [low, high).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <remarks>
        /// The array-size and 'int' overloads are necessary for F#, which doesn't implicitly convert types.
        /// Once implicit conversion support is broadly available, some of these overloads can be removed.
        /// </remarks>
        public static Tensor randint(long low, long high, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(low, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [low, high).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <remarks>
        /// The array-size and 'int' overloads are necessary for F#, which doesn't implicitly convert types.
        /// Once implicit conversion support is broadly available, some of these overloads can be removed.
        /// </remarks>
        public static Tensor randint(int low, int high, int[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(low, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [0, high).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(int high, int[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(0, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a random Boolean value.
        /// </summary>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static bool randint_bool(Generator? generator = null)
        {
            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            var result = THSTensor_randint_bool(genHandle);
            if (result == -1) CheckForErrors();
            return result == 1;
        }

        /// <summary>
        /// Create a random integer value taken from a uniform distribution in [0, high).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static int randint_int(int high, Generator? generator = null)
        {
            return randint_int(0, high, generator);
        }

        /// <summary>
        /// Create a random integer value taken from a uniform distribution in [low, high).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static int randint_int(int low, int high, Generator? generator = null)
        {
            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            var result = THSTensor_randint_int(genHandle, low, high);
            CheckForErrors();
            return result;
        }

        /// <summary>
        /// Create a random integer value taken from a uniform distribution in [0, high).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static long randint_long(long high, Generator? generator = null)
        {
            return randint_long(0, high, generator);
        }

        /// <summary>
        /// Create a random integer value taken from a uniform distribution in [low, high).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static long randint_long(long low, long high, Generator? generator = null)
        {
            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            var result = THSTensor_randint_long(genHandle, low, high);
            CheckForErrors();
            return result;
        }

        /// <summary>
        /// Create a new 32-bit complex tensor filled with random integer values taken from a uniform distribution in [0, high).
        /// </summary>
        /// <param name="genHandle"></param>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        private static Tensor randint_c32(IntPtr genHandle, long low, long high, long[] size, Device device, bool requires_grad, string[]? names)
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
                    var handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        CheckForErrors();

                    //
                    // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                    //
                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int)device.type, device.index, requires_grad);
                    if (res == IntPtr.Zero)
                        CheckForErrors();

                    THSTensor_copy_(res, cmplx, false);
                    CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    var result = new Tensor(res);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new 64-bit complex tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        /// <param name="genHandle"></param>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        private static Tensor randint_c64(IntPtr genHandle, long low, long high, long[] size, Device device, bool requires_grad, string[]? names)
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
                    var handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        CheckForErrors();

                    //
                    // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                    //
                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int)device.type, device.index, requires_grad);
                    if (res == IntPtr.Zero)
                        CheckForErrors();

                    THSTensor_copy_(res, cmplx, false);
                    CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    var result = new Tensor(res);
                    if (names != null && names.Length > 0) {
                        result.rename_(names);
                    }
                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        private static Tensor _rand(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            if (dtype.HasValue && torch.is_integral(dtype.Value))
                throw new ArgumentException($"torch.rand() was passed a bad dtype: {dtype}. It must be floating point or complex.", "dtype");

            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_rand(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
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

        /// <summary>
        /// Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a random floating-point value taken from a uniform distribution in [0, 1).
        /// </summary>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static double rand_float(Generator? generator = null)
        {
            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            var result = THSTensor_rand_float(genHandle);
            CheckForErrors();
            return result;
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        private static Tensor _randn(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            if (dtype.HasValue && torch.is_integral(dtype.Value))
                throw new ArgumentException($"torch.randn() was passed a bad dtype: {dtype}. It must be floating point or complex.", "dtype");

            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_randn(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
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

        /// <summary>
        /// Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a random floating-point value taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <remarks>This method does not exist in PyTorch, but is useful for getting a scalar from a torch RNG.</remarks>
        public static double randn_float(Generator? generator = null)
        {
            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            var result = THSTensor_randn_float(genHandle);
            CheckForErrors();
            return result;
        }
    }
}
