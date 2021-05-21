using System;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

// The scalar 'from' factories for complex tensors require some hand-written code, cannot be generated.

namespace TorchSharp.Tensor
{
    public partial class ComplexFloat32Tensor
    {
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
    }
    public partial class ComplexFloat64Tensor
    {
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
    }
}
