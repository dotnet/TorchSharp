using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

// The scalar 'from' factories for complex tensors require some hand-written code, cannot be generated.

namespace TorchSharp.Tensor
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
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int)device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int)device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }

            var res = THSTensor_to_type(handle, (sbyte)ScalarType.ComplexFloat32);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();

            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newComplexFloat32Scalar(float real, float imaginary, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from((float Real, float Imaginary) scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newComplexFloat32Scalar(scalar.Real, scalar.Imaginary, (int)device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(float real, float imaginary = 0.0f, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newComplexFloat32Scalar(real, imaginary, (int)device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
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
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var sz = new List<long>();
            sz.AddRange(size);
            sz.Add(2);
            var size2 = sz.ToArray();

            unsafe {
                fixed (long* psizes = size2) {
                    var handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        Torch.CheckForErrors();

                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int)device.Type, device.Index, requiresGrad);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();

                    res = THSTensor_copy_(res, cmplx, false);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    return new TorchTensor(res);
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
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int)device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int)device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }

            var res = THSTensor_to_type(handle, (sbyte)ScalarType.ComplexFloat64);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();

            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newComplexFloat64Scalar(double real, double imaginary, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(System.Numerics.Complex scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(double real, double imaginary = 0.0f, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(real, imaginary, (int)device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
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
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var sz = new List<long>();
            sz.AddRange(size);
            sz.Add(2);
            var size2 = sz.ToArray();

            unsafe {
                fixed (long* psizes = size2) {
                    var handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(max, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        Torch.CheckForErrors();

                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int)device.Type, device.Index, requiresGrad);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();

                    res = THSTensor_copy_(res, cmplx, false);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    return new TorchTensor(res);
                }
            }
        }

    }
}
