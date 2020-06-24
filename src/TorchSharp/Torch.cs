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

        [DllImport("LibTorchSharp")]
        private static extern bool THSTorchCuda_is_available();

        public static bool IsCudaAvailable()
        {
            return THSTorchCuda_is_available();
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTorchCuda_cudnn_is_available();

        public static bool IsCudnnAvailable()
        {
            return THSTorchCuda_cudnn_is_available();
        }

        [DllImport("LibTorchSharp")]
        private static extern int THSTorchCuda_device_count();

        public static int DeviceCount()
        {
            return THSTorchCuda_device_count();
        }

        [DllImport("LibTorchSharp")]
        private static extern int THSTorchCuda_device_capability();

        public static int GetDeviceCapability()
        {
            return THSTorchCuda_device_capability();
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
