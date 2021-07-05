using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

// The scalar 'from' factories for complex tensors require some hand-written code, cannot be generated.

namespace TorchSharp
{
    public static partial class torch
    {

        public partial class ComplexFloat32Tensor
        {
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);


            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

            /// <summary>
            /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
            /// common difference step, starting from start.
            /// </summary>
            /// <remarks>In the case of complex element types, 'arange' will create a complex tensor with img=0 in all elements.</remarks>
            static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);

                var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int)device.type, device.index, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                var res = THSTensor_to_type(handle, (sbyte)ScalarType.ComplexFloat32);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();

                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_newComplexFloat32Scalar(float real, float imaginary, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from((float Real, float Imaginary) scalar, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat32Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from(float real, float imaginary = 0.0f, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat32Scalar(real, imaginary, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_view_as_complex(IntPtr tensor);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, bool non_blocking);

            [DllImport("LibTorchSharp")]
            extern static void THSTensor_dispose(IntPtr handle);

            /// <summary>
            ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
            ///  The real and imaginary parts will be filled independently of each other.
            /// </summary>
            static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
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
        }

        public partial class ComplexFloat64Tensor
        {
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

            /// <summary>
            /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
            /// common difference step, starting from start.
            /// </summary>
            /// <remarks>In the case of complex element types, 'arange' will create a complex tensor with img=0 in all elements.</remarks>
            static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);

                var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int)device.type, device.index, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }

                var res = THSTensor_to_type(handle, (sbyte)ScalarType.ComplexFloat64);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();

                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_newComplexFloat64Scalar(double real, double imaginary, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from(System.Numerics.Complex scalar, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            /// <summary>
            /// Create a scalar tensor from a single value
            /// </summary>
            public static Tensor from(double real, double imaginary = 0.0f, torch.Device device = null, bool requiresGrad = false)
            {
                device = torch.InitializeDevice(device);
                var handle = THSTensor_newComplexFloat64Scalar(real, imaginary, (int)device.type, device.index, requiresGrad);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_view_as_complex(IntPtr tensor);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, bool non_blocking);

            [DllImport("LibTorchSharp")]
            extern static void THSTensor_dispose(IntPtr handle);

            /// <summary>
            ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
            ///  The real and imaginary parts will be filled independently of each other.
            /// </summary>
            static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
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

        }
    }
}