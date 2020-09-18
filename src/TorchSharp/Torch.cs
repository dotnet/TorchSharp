// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Data;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    using Debug = System.Diagnostics.Debug;

    public static class Torch
    {
        const string libtorchPackageVersion = "1.5.6";
        const string cudaVersion = "10.2";

        [DllImport("LibTorchSharp")]
        private static extern void THSTorch_seed(long seed);

        public static void SetSeed(long seed)
        {
            THSTorch_seed(seed);
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);

        static long initializationMask = 0;

        static string nativeRid =>
            (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) ? "win-x64" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) ? "linux-x64" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) ? "osx-x64" :
            "any";

        static string nativeGlob=>
            (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) ? "*.dll" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) ? "*.so" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) ? "*.dynlib" :
            "*.so";

        internal static bool TryInitializeDeviceType(DeviceType deviceType)
        {
            long mask = 1L << ((int)deviceType);

            if ((initializationMask & mask) == 0) {
                Console.WriteLine($"Initialising for device type {deviceType}");

                bool ok;

                if (deviceType == DeviceType.CUDA) {
                    // See https://github.com/pytorch/pytorch/issues/33415
                    ok = NativeLibrary.TryLoad("torch_cuda", typeof(Torch).Assembly, null, out var res1);
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                        NativeLibrary.TryLoad("nvrtc-builtins64_102", typeof(Torch).Assembly, null, out var res2);
                        NativeLibrary.TryLoad("caffe2_nvrtc", typeof(Torch).Assembly, null, out var res3);
                        NativeLibrary.TryLoad("nvrtc64_102_0", typeof(Torch).Assembly, null, out var res4);
                    }
                } else {
                    ok = NativeLibrary.TryLoad("torch_cpu", typeof(Torch).Assembly, null, out var res1);
                }
                if (!ok) {
                    Console.WriteLine($"Failed application init, now trying .NET Interactive init for device type {deviceType}");
                    // See https://github.com/xamarin/TorchSharp/issues/169
                    //
                    // If we are loading in .NET Interactive or F# Interactive, these load DLLs directly
                    // from package directories, instead of from a collected application directory. For
                    // managed DLLs this works OK, but native DLLs do not load transitive dependencies.
                    //
                    // So we iteratively load the native DLLs here.
                    //
                    // Assumed to be in ...\packages\torchsharp\0.3.0-local-debug-20200918\lib\netcoreapp3.0\TorchSharp.dll
                    var loc = Path.GetDirectoryName(typeof(Torch).Assembly.Location);
                    var packagesDir = Path.Combine(loc, "..", "..", "..", "..");
                    if (deviceType == DeviceType.CUDA) {
                        LoadNativeComponentsFromMultiplePackages(packagesDir, $"libtorch-cuda-{cudaVersion}-{nativeRid}-*");
                    } else if (deviceType == DeviceType.CPU) {
                        LoadNativeComponentsFromMultiplePackages(packagesDir, "libtorch-cpu");
                    }
                }
                initializationMask |= mask;
            }
            if (deviceType == DeviceType.CUDA) {
                return THSTorchCuda_is_available();
            }
            else {
                return true;
            }
        }

        /// Iteratively load the components until all are loaded.  
        private static void LoadNativeComponentsFromMultiplePackages(string packagesDir, string packagePattern)
        {
            // Some loads will fail due to missing dependencies but then
            // these will be resolved in subsequent iterations.
            Console.WriteLine($"LoadNativeComponentsFromMultiplePackages, packagesDir = {packagesDir}");
            if (Directory.Exists(packagesDir)) {
                var packages =
                    Directory.GetDirectories(packagesDir, packagePattern)
                       .Where(d => Directory.Exists(Path.Combine(d, libtorchPackageVersion)))
                       .ToArray();

                Console.WriteLine($"LoadNativeComponentsFromMultiplePackages, packages = {packages}");
                if (packages.Length > 0) {
                    for (int i = 0; i < 10; i++) {
                        var allOk = true;
                        Console.WriteLine($"LoadNativeComponentsFromMultiplePackages, iteration {i}");
                        foreach (var part in packages) {
                            var natives = Path.Combine(part, libtorchPackageVersion, "runtimes", nativeRid, "native");
                            if (Directory.Exists(natives)) {
                                foreach (var file in Directory.GetFiles(natives, nativeGlob)) {
                                    var ok = NativeLibrary.TryLoad(file, out var res);
                                    allOk &= ok;
                                    if (ok) {
                                        Console.WriteLine($"Loaded {file} on iteration {i}");
                                    } else if (i < 9) {
                                        Console.WriteLine($"** Failed to load {file} on iteration {i}, will try again on next iteration");

                                    } else {
                                        Console.WriteLine($"!!! Failed to load {file}, giving up");
                                    }
                                }
                            }
                        }
                        if (allOk)
                            break;
                    }
                }
            }
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
