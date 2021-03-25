// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    using Debug = System.Diagnostics.Debug;

    public static class Torch
    {
        const string libtorchPackageVersion = "1.8.0.7";
        const string cudaVersion = "11.1";

        [DllImport("LibTorchSharp")]
        private static extern void THSTorch_manual_seed(long seed);

        public static void SetSeed(long seed)
        {
            THSTorch_manual_seed(seed);
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);


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
        static bool nativeBackendLoaded = false;
        static bool nativeBackendCudaLoaded = false;

        public static void LoadNativeBackend(bool useCudaBackend)
        {
            bool ok = false;
            var isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            if (!(useCudaBackend ? nativeBackendCudaLoaded : nativeBackendLoaded)) {
                Debug.WriteLine($"TorchSHarp: Initialising native backend");

                // See https://github.com/pytorch/pytorch/issues/33415
                if (useCudaBackend)
                    ok = NativeLibrary.TryLoad("torch_cuda", typeof(Torch).Assembly, null, out var res1);
                if (ok) {
                    if (isWindows) {

                        // Preloading these DLLs on windows seems to iron out problems where one native DLL
                        // requests a load of another through dynamic linking techniques.  
                        // 
                        NativeLibrary.TryLoad("cudnn_adv_infer64_8", typeof(Torch).Assembly, null, out var res2);
                        NativeLibrary.TryLoad("cudnn_adv_train64_8", typeof(Torch).Assembly, null, out var res3);
                        NativeLibrary.TryLoad("cudnn_cnn_infer64_8", typeof(Torch).Assembly, null, out var res4);
                        NativeLibrary.TryLoad("cudnn_cnn_train64_8", typeof(Torch).Assembly, null, out var res5);
                        NativeLibrary.TryLoad("cudnn_ops_infer64_8", typeof(Torch).Assembly, null, out var res6);
                        NativeLibrary.TryLoad("cudnn_ops_train64_8", typeof(Torch).Assembly, null, out var res7);
                        NativeLibrary.TryLoad("nvrtc-builtins64_111", typeof(Torch).Assembly, null, out var res8);
                        NativeLibrary.TryLoad("caffe2_nvrtc", typeof(Torch).Assembly, null, out var res9);
                        NativeLibrary.TryLoad("nvrtc64_111_0", typeof(Torch).Assembly, null, out var res10);
                    }
                } else {
                    ok = NativeLibrary.TryLoad("torch_cpu", typeof(Torch).Assembly, null, out var res2);
                }
                if (!ok) {
                    Console.WriteLine($"Native backend not found in application. Trying dynamic load for .NET/F# Interactive...");

                    // See https://github.com/xamarin/TorchSharp/issues/169
                    //
                    // If we are loading in .NET Interactive or F# Interactive, these are in packages in separate
                    // package directories. For managed DLLs this works OK, but native DLLs do not load transitive dependencies.
                    //
                    // So we shadow copy the DLLs to the TorchSharp package, make a copy of the native DLL and continue
                    //
                    // Assumed to be in ...\packages\torchsharp\0.3.0-local-debug-20200918\lib\net5.0\TorchSharp.dll
                    var cpuRootPackage = "libtorch-cpu";
                    var cudaRootPackage = $"libtorch-cuda-{cudaVersion}-{nativeRid}";
                    var torchsharpLoc = Path.GetDirectoryName(typeof(Torch).Assembly.Location);
                    if (torchsharpLoc.Contains("torchsharp") && torchsharpLoc.Contains("lib")) {

                        var packagesDir = Path.GetFullPath(Path.Combine(torchsharpLoc, "..", "..", "..", ".."));
                        var torchSharpVersion = Path.GetFileName(Path.GetFullPath(Path.Combine(torchsharpLoc, "..", "..")));
                        var target = isWindows ? "LibTorchSharp.dll" : "libLibTorchSharp.so";

                        if (useCudaBackend) {
                            var cudaTarget = Path.Combine(torchsharpLoc, $"cuda-{cudaVersion}");
                            var cudaOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, $"{cudaRootPackage}-*", libtorchPackageVersion, cudaTarget);
                            if (cudaOk) {
                                ok = CopyNativeComponentsIntoSingleDirectory(packagesDir, "torchsharp", torchSharpVersion, cudaTarget);
                                if (ok) {
                                    ok = NativeLibrary.TryLoad(Path.Combine(cudaTarget, target), out var res3);
                                }
                            }
                            if (!ok)
                                throw new NotSupportedException($"The {cudaRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cudaRootPackage}, {libtorchPackageVersion}\"");
                        }
                        else {
                            var cpuTarget = Path.Combine(torchsharpLoc, $"cpu");
                            var cpuOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, cpuRootPackage, libtorchPackageVersion, cpuTarget);
                            if (cpuOk) {
                                ok = CopyNativeComponentsIntoSingleDirectory(packagesDir, "torchsharp", torchSharpVersion, cpuTarget);
                                if (ok) {
                                    ok = NativeLibrary.TryLoad(Path.Combine(cpuTarget, target), out var res4);
                                }
                            }
                            if (!ok)
                                throw new NotSupportedException($"The {cpuRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cpuRootPackage}, {libtorchPackageVersion}\"");
                        }
                    }
                    else 
                        throw new NotSupportedException($"This application uses TorchSharp but doesn't contain reference to either {cudaRootPackage} or {cpuRootPackage}, {libtorchPackageVersion}\"");
                }
                if (useCudaBackend)
                    nativeBackendCudaLoaded = true;
                else
                    nativeBackendLoaded = true;
            }
        }

        public static bool TryInitializeDeviceType(DeviceType deviceType)
        {
            LoadNativeBackend(deviceType == DeviceType.CUDA);
            if (deviceType == DeviceType.CUDA) {
                return THSTorchCuda_is_available();
            }
            else {
                return true;
            }
        }

        /// Copy all native runtime DLLs into single directory if it hasn't been done already
        private static bool CopyNativeComponentsIntoSingleDirectory(string packagesDir, string packagePattern, string packageVersion, string target)
        {
            // Some loads will fail due to missing dependencies but then
            // these will be resolved in subsequent iterations.
            Console.WriteLine($"CopyNativeComponentsIntoSingleDirectory, packagesDir = {packagesDir}");
            if (Directory.Exists(packagesDir)) {
                var packages =
                    Directory.GetDirectories(packagesDir, packagePattern)
                       .Where(d => Directory.Exists(Path.Combine(d, packageVersion)))
                       .ToArray();

                if (packages.Length > 0) {
                    foreach (var package in packages) {
                        Console.WriteLine($"CopyNativeComponentsIntoSingleDirectory, package {package}");
                        var natives = Path.Combine(package, packageVersion, "runtimes", nativeRid, "native");
                        Console.WriteLine($"CopyNativeComponentsIntoSingleDirectory, natives {natives}");
                        if (Directory.Exists(natives)) {
                            foreach (var file in Directory.GetFiles(natives, nativeGlob)) {
                                var targetFile = Path.Combine(target, Path.GetFileName(file));
                                if (!File.Exists(targetFile))
                                    File.Copy(file, targetFile);
                            }
                        }
                    }
                    return true;
                }
            }
            return false;
        }

        public static void InitializeDeviceType(DeviceType deviceType)
        {
            if (!TryInitializeDeviceType(deviceType)) {
                throw new InvalidOperationException($"Torch device type {deviceType} did not initialise on the current machine.");
            }
        }

        public static Device InitializeDevice(Device device)
        {
            if (device == null)
                device = TorchSharp.Device.CPU;
            InitializeDeviceType(device.Type);
            return device;
        }

        public static Device Device(string description)
        {
            return new Device(description);
        }

        public static Device Device(DeviceType type, int index = -1)
        {
            return new Device(type, index);
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
