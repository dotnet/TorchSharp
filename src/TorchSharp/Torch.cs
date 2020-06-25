// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    using Debug = System.Diagnostics.Debug;

    public static class Torch
    {
        [DllImport("LibTorchSharp")]
        private static extern void THSTorch_seed(long seed);

        public static void SetSeed(long seed)
        {
            THSTorch_seed(seed);
        }

        internal static bool TryInitializeDeviceType(DeviceType deviceType)
        {
            if (deviceType == DeviceType.CUDA) {

                // See https://github.com/pytorch/pytorch/issues/33415
                if (System.Environment.OSVersion.Platform == PlatformID.Win32NT) {
                    LoadLibrary(@"torch_cuda.dll");
                }
                return THSTorchCuda_is_available();

            }
            return true;
        }

        internal static void InitializeDeviceType(DeviceType deviceType)
        {
            if (!TryInitializeDeviceType(deviceType)) {
                throw new InvalidOperationException($"Torch device type {deviceType} did not initialise on the current machine.");
            }

        }

        internal static bool TryInitializeDevice(DeviceType deviceType, int deviceIndex)
        {
            return TryInitializeDeviceType(deviceType);
        }

        internal static void InitializeDevice(DeviceType deviceType, int deviceIndex)
        {
            if (!TryInitializeDevice(deviceType, deviceIndex)) {
                throw new InvalidOperationException($"Torch device type {deviceType} did not initialise on the current machine.");
            }

        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTorchCuda_is_available();

        [DllImport("kernel32.dll")]
        public static extern IntPtr LoadLibrary(string dllToLoad);

        public static bool IsCudaAvailable()
        {
            TryInitializeDeviceType(DeviceType.CUDA);
            return THSTorchCuda_is_available();
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTorchCuda_cudnn_is_available();

        public static bool IsCudnnAvailable()
        {
            TryInitializeDeviceType(DeviceType.CUDA);
            return THSTorchCuda_cudnn_is_available();
        }

        [DllImport("LibTorchSharp")]
        private static extern int THSTorchCuda_device_count();

        public static int CudaDeviceCount()
        {
            TryInitializeDeviceType(DeviceType.CUDA);
            return THSTorchCuda_device_count();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTorch_get_and_reset_last_err();

        //[Conditional("DEBUG")]
        internal static void CheckForErrors()
        {
            var error = THSTorch_get_and_reset_last_err();

            if (error != IntPtr.Zero)
            {
                throw new ExternalException(Marshal.PtrToStringAnsi(error));
            }
        }
    }

    public enum DeviceType
    {
        CPU = 0,
        CUDA = 1, // CUDA.
        MKLDNN = 2, // Reserved for explicit MKLDNN
        OPENGL = 3, // OpenGL
        OPENCL = 4, // OpenCL
        IDEEP = 5, // IDEEP.
        HIP = 6, // AMD HIP
        FPGA = 7, // FPGA
        MSNPU = 8, // MSNPU
        XLA = 9 // XLA / TPU
    }
}
