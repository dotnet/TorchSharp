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

        internal static bool TryLoadNativeLibraryFromFile(string path) {
            bool ok = false;
            try {
                ok = NativeLibrary.TryLoad(path, out var res);
            }
            catch {
                ok = false;
            }
            return ok;
        }

        internal static bool TryLoadNativeLibraryByName(string name, System.Reflection.Assembly assembly)
        {
            bool ok = false;
            try {
                ok = NativeLibrary.TryLoad(name, assembly, null, out var res);
            } catch {
                ok = false;
            }
            return ok;
        }

        public static void LoadNativeBackend(bool useCudaBackend)
        {
            bool ok = false;
            var isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            if (!(useCudaBackend ? nativeBackendCudaLoaded : nativeBackendLoaded)) {
                Debug.WriteLine($"TorchSHarp: Initialising native backend");

                // See https://github.com/pytorch/pytorch/issues/33415
                if (useCudaBackend) {
                    ok = TryLoadNativeLibraryByName("torch_cuda", typeof(Torch).Assembly);
                }
                if (ok) {
                    if (isWindows) {

                        // Preloading these DLLs on windows seems to iron out problems where one native DLL
                        // requests a load of another through dynamic linking techniques.  
                        // 
                        TryLoadNativeLibraryByName("cudnn_adv_infer64_8", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_adv_train64_8", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_cnn_infer64_8", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_cnn_train64_8", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_ops_infer64_8", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_ops_train64_8", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("nvrtc-builtins64_111", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("caffe2_nvrtc", typeof(Torch).Assembly);
                        TryLoadNativeLibraryByName("nvrtc64_111_0", typeof(Torch).Assembly);
                    }
                } else {
                    ok = TryLoadNativeLibraryByName("torch_cpu", typeof(Torch).Assembly);
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
                    //
                    // TODO: on linux make these copies link not shadow-copy
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
                                    ok = TryLoadNativeLibraryFromFile(Path.Combine(cudaTarget, target));
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
                                    ok = TryLoadNativeLibraryFromFile(Path.Combine(cpuTarget, target));
                                }
                            }
                            if (!ok)
                                throw new NotSupportedException($"The {cpuRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cpuRootPackage}, {libtorchPackageVersion}\"");
                        }
                    }
                    else 
                        throw new NotSupportedException($"This application uses TorchSharp but doesn't contain reference to either {cudaRootPackage} or {cpuRootPackage}, {libtorchPackageVersion}. Consider either referncing one of these packages or call System.Runtime.InteropServices.NativeLibrary.Load explicitly for a Python install or a download of libtorch.so/torch.dll. See https://github.com/xamarin/TorchSharp/issues/169.\"");
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
