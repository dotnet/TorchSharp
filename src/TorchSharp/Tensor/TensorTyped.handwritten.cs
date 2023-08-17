// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;

// The scalar 'from' factories for complex tensors require some hand-written code, cannot be generated.

namespace TorchSharp
{
    public static partial class torch
    {
        internal partial class ComplexFloat32Tensor
        {
            /// <summary>
            /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
            /// common difference step, starting from start.
            /// </summary>
            /// <remarks>In the case of complex element types, 'arange' will create a complex tensor with img=0 in all elements.</remarks>
            public static Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);

                var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                var res = THSTensor_to_type(handle, (sbyte)ScalarType.ComplexFloat32);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();

                return new Tensor(res);
            }

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from((float Real, float Imaginary) scalar, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat32Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from(float real, float imaginary = 0.0f, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat32Scalar(real, imaginary, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
            ///  The real and imaginary parts will be filled independently of each other.
            /// </summary>
            public static Tensor randint(long max, long[] size, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);

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
                        var handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                        if (handle == IntPtr.Zero) {
                            GC.Collect();
                            GC.WaitForPendingFinalizers();
                            handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                        }
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                        var cmplx = THSTensor_view_as_complex(handle);
                        if (cmplx == IntPtr.Zero)
                            torch.CheckForErrors();

                        //
                        // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                        //
                        var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int)device.type, device.index, requires_grad);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();

                        THSTensor_copy_(res, cmplx, false);
                        torch.CheckForErrors();

                        THSTensor_dispose(handle);
                        THSTensor_dispose(cmplx);

                        return new Tensor(res);
                    }
                }
            }
        }

        internal partial class ComplexFloat64Tensor
        {
            /// <summary>
            /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
            /// common difference step, starting from start.
            /// </summary>
            /// <remarks>In the case of complex element types, 'arange' will create a complex tensor with img=0 in all elements.</remarks>
            public static Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);

                var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                var res = THSTensor_to_type(handle, (sbyte)ScalarType.ComplexFloat64);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();

                return new Tensor(res);
            }

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from(System.Numerics.Complex scalar, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from(double real, double imaginary = 0.0f, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat64Scalar(real, imaginary, (int)device.type, device.index, requires_grad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
            ///  The real and imaginary parts will be filled independently of each other.
            /// </summary>
            public static Tensor randint(long max, long[] size, torch.Device device = null, bool requires_grad = false)
            {
                device = torch.InitializeDevice(device);

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
                        var handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                        if (handle == IntPtr.Zero) {
                            GC.Collect();
                            GC.WaitForPendingFinalizers();
                            handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                        }
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                        var cmplx = THSTensor_view_as_complex(handle);
                        if (cmplx == IntPtr.Zero)
                            torch.CheckForErrors();

                        //
                        // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                        //
                        var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int)device.type, device.index, requires_grad);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();

                        THSTensor_copy_(res, cmplx, false);
                        torch.CheckForErrors();

                        THSTensor_dispose(handle);
                        THSTensor_dispose(cmplx);

                        return new Tensor(res);
                    }
                }
            }

        }
    }
}