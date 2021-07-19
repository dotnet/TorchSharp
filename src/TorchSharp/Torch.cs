// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace TorchSharp
{
    using Debug = System.Diagnostics.Debug;

    public static partial class torch
    {
        const string libtorchPackageVersion = "1.9.0.7";
        const string cudaVersion = "11.1";

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);


        static string nativeRid =>
            (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) ? "win-x64" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) ? "linux-x64" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) ? "osx-x64" :
            "any";

        static string nativeGlob =>
            (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) ? @".*\.dll" :
            (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) ? @".*\.dylib\.*" :
            // must match
            //   lib.so
            //   lib.so.1
            //   lib.so.11.0
            //   lib.so.11.1
            @".*\.so(\.\d*)*";
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
            var target = 
                (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) ? "LibTorchSharp.dll":
                (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) ? "libLibTorchSharp.dylib" :
                "libLibTorchSharp.so";
            if (!(useCudaBackend ? nativeBackendCudaLoaded : nativeBackendLoaded)) {
                Trace.WriteLine($"TorchSharp: LoadNativeBackend: Initialising native backend");

                // Workarounds for weird LibTorch native stuff
                // See https://github.com/pytorch/pytorch/issues/33415
                if (useCudaBackend) {
                    var isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
                    if (isWindows) {
                        Trace.WriteLine($"Try loading Windows cuda native components");
                        // Preloading these DLLs on windows seems to iron out problems where one native DLL
                        // requests a load of another through dynamic linking techniques.  
                        // 
                        TryLoadNativeLibraryByName("cudnn_adv_infer64_8", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_adv_train64_8", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_cnn_infer64_8", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_cnn_train64_8", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_ops_infer64_8", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("cudnn_ops_train64_8", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("nvrtc-builtins64_111", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("caffe2_nvrtc", typeof(torch).Assembly);
                        TryLoadNativeLibraryByName("nvrtc64_111_0", typeof(torch).Assembly);
                    }
                    Trace.WriteLine($"TorchSharp: LoadNativeBackend: Try loading torch_cuda native component");
                    TryLoadNativeLibraryByName("torch_cuda", typeof(torch).Assembly);
                } else {
                    Trace.WriteLine($"TorchSharp: LoadNativeBackend: Loading torch_cpu");
                    TryLoadNativeLibraryByName("torch_cpu", typeof(torch).Assembly);
                }
                Trace.WriteLine($"TorchSharp: LoadNativeBackend: Loading LibTorchSharp");
                ok = TryLoadNativeLibraryByName("LibTorchSharp", typeof(torch).Assembly);

                Trace.WriteLine($"TorchSharp: LoadNativeBackend: Loaded LibTorchSharp, ok = {ok}");
                // Try dynamic load from package directories
                var cpuRootPackage = "libtorch-cpu";
                var cudaRootPackage = $"libtorch-cuda-{cudaVersion}-{nativeRid}";
                if (!ok) {

                    Console.WriteLine($"TorchSharp: LoadNativeBackend: Native backend not found in application loading TorchSharp directly from packages directory.");
                    // See https://github.com/xamarin/TorchSharp/issues/169
                    //
                    // If we are loading in .NET Interactive or F# Interactive, these are in packages in separate
                    // package directories. For managed DLLs this works OK, but native DLLs do not load transitive dependencies.
                    //
                    // So we shadow copy the DLLs into the TorchSharp package, make a copy of the native DLL and continue
                    // with the dynamic load
                    //
                    // Assumed to be in ...\packages\torchsharp\0.3.0-local-debug-20200918\lib\net5.0\TorchSharp.dll
                    //
                    // TODO: on linux make these copies link not shadow-copy
                    var torchsharpLoc = Path.GetDirectoryName(typeof(torch).Assembly.Location);
                    var packagesDir = Path.GetFullPath(Path.Combine(torchsharpLoc, "..", "..", "..", ".."));
                    var torchsharpHome = Path.GetFullPath(Path.Combine(torchsharpLoc, "..", ".."));
                    if (torchsharpLoc.Contains("torchsharp") && torchsharpLoc.Contains("lib") && Directory.Exists(packagesDir) && Directory.Exists(torchsharpHome)) {

                        var torchSharpVersion = Path.GetFileName(torchsharpHome); // really GetDirectoryName

                        if (useCudaBackend) {
                            var consolidatedDir = Path.Combine(torchsharpLoc, $"cuda-{cudaVersion}");
                            Console.WriteLine($"TorchSharp: LoadNativeBackend: Trying dynamic load for .NET/F# Interactive by consolidating native {cudaRootPackage}-* binaries to {consolidatedDir}...");
                            var cudaOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, $"{cudaRootPackage}-*", libtorchPackageVersion, consolidatedDir);
                            if (cudaOk) {
                                Trace.WriteLine($"TorchSharp: LoadNativeBackend: Consolidating native LibTorchSharp binaries to {consolidatedDir}...");
                                cudaOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, "torchsharp", torchSharpVersion, consolidatedDir);
                                if (cudaOk) {
                                    var consolidated = Path.Combine(consolidatedDir, target);
                                    Trace.WriteLine($"TorchSharp: LoadNativeBackend: Trying to load {consolidated}...");
                                    ok = TryLoadNativeLibraryFromFile(consolidated);
                                }
                            }
                            if (!cudaOk)
                                throw new NotSupportedException($"The {cudaRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cudaRootPackage}, {libtorchPackageVersion}\"");
                        }
                        else {
                            var consolidatedDir = Path.Combine(torchsharpLoc, $"cpu");
                            Console.WriteLine($"TorchSharp: LoadNativeBackend: Trying dynamic load for .NET/F# Interactive by consolidating native {cpuRootPackage}-* binaries to {consolidatedDir}...");
                            var cpuOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, cpuRootPackage, libtorchPackageVersion, consolidatedDir);
                            if (cpuOk) {
                                Trace.WriteLine($"TorchSharp: LoadNativeBackend: Consolidating native LibTorchSharp binaries to {consolidatedDir}...");
                                cpuOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, "torchsharp", torchSharpVersion, consolidatedDir);
                                if (cpuOk) {
                                    var consolidated = Path.Combine(consolidatedDir, target);
                                    Trace.WriteLine($"TorchSharp: LoadNativeBackend: Trying to load {consolidated}...");
                                    ok = TryLoadNativeLibraryFromFile(consolidated);
                                    Trace.WriteLine($"TorchSharp: LoadNativeBackend: ok = {ok}...");
                                }
                            }
                            if (!cpuOk)
                                throw new NotSupportedException($"The {cpuRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cpuRootPackage}, {libtorchPackageVersion}\"");
                        }
                    }
                }
                if (!ok)
                    throw new NotSupportedException($"This application uses TorchSharp but doesn't contain reference to either {cudaRootPackage} or {cpuRootPackage}, {libtorchPackageVersion}. Consider either referncing one of these packages or call System.Runtime.InteropServices.NativeLibrary.Load explicitly for a Python install or a download of libtorch.so/torch.dll. See https://github.com/xamarin/TorchSharp/issues/169.\"");

                // Record the successful load
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
                return cuda.CallTorchCudaIsAvailable();
            } else {
                return true;
            }
        }

        /// Copy all native runtime DLLs into single directory if it hasn't been done already
        private static bool CopyNativeComponentsIntoSingleDirectory(string packagesDir, string packagePattern, string packageVersion, string target)
        {
            // Some loads will fail due to missing dependencies but then
            // these will be resolved in subsequent iterations.
            Trace.WriteLine($"CopyNativeComponentsIntoSingleDirectory: packagesDir = {packagesDir}");
            if (Directory.Exists(packagesDir)) {
                var packages =
                    Directory.GetDirectories(packagesDir, packagePattern)
                       .Where(d => Directory.Exists(Path.Combine(d, packageVersion)))
                       .ToArray();

                if (packages.Length > 0) {
                    if (!Directory.Exists(target))
                        Directory.CreateDirectory(target);
                    foreach (var package in packages) {
                        var natives = Path.Combine(package, packageVersion, "runtimes", nativeRid, "native");
                        Trace.WriteLine($"CopyNativeComponentsIntoSingleDirectory: package={package}, natives={natives}, target={target}");
                        if (Directory.Exists(natives)) {
                            var nativeRegExp = new Regex("^"+nativeGlob+"$");
                            foreach (var file in Directory.GetFiles(natives).Where(path => nativeRegExp.IsMatch(path))) {
                                var targetFile = Path.Combine(target, Path.GetFileName(file));
                                if (!File.Exists(targetFile)) {
                                    Trace.WriteLine($"Copy {file} --> {targetFile}");
                                    File.Copy(file, targetFile);
                                }
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

        public static Device InitializeDevice(torch.Device device)
        {
            if (device == null)
                device = torch.CPU;
            InitializeDeviceType(device.type);
            return device;
        }

        public static partial class random
        {
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSGenerator_manual_seed(long seed);

            public static Generator manual_seed(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                var res = THSGenerator_manual_seed(seed);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Generator(res);
            }
        }

        public static partial class nn
        {
            public static partial class utils
            {
                [DllImport("LibTorchSharp")]
                extern static double THSTensor_clip_grad_norm_(IntPtr tensor, int len, double max_norm, double norm_type);

                /// <summary>
                /// Clips gradient norm of an iterable of parameters.
                /// The norm is computed over all gradients together, as if they were concatenated into a single vector.
                /// Gradients are modified in-place.
                /// </summary>
                /// <param name="tensors"></param>
                /// <param name="max_norm"></param>
                /// <param name="norm_type"></param>
                /// <returns></returns>
                public static double clip_grad_norm_(IList<Tensor> tensors, double max_norm, double norm_type = 2.0)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                        return THSTensor_clip_grad_norm_(tensorsRef, parray.Array.Length, max_norm, norm_type);
                    }
                }

            }
        }

        public static partial class cuda
        {
            [DllImport("LibTorchSharp")]
            private static extern bool THSTorchCuda_is_available();

            /// This must be a separate method to the failure to bind DllImport THSTorchCuda_is_available
            /// is not raised as early as a DllImportException
            [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
            internal static bool CallTorchCudaIsAvailable()
            {
                return THSTorchCuda_is_available();
            }

            /// <summary>
            /// Returns a bool indicating if CUDA is currently available.
            /// </summary>
            /// <returns></returns>
            public static bool is_available()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return CallTorchCudaIsAvailable();
            }

            [DllImport("LibTorchSharp")]
            private static extern bool THSTorchCuda_cudnn_is_available();

            public static bool is_cudnn_available()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return THSTorchCuda_cudnn_is_available();
            }

            [DllImport("LibTorchSharp")]
            private static extern int THSTorchCuda_device_count();

            /// <summary>
            /// Returns the number of GPUs available.
            /// </summary>
            /// <returns></returns>
            public static int device_count()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return THSTorchCuda_device_count();
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSCuda_manual_seed(long seed);

            /// <summary>
            /// Sets the seed for generating random numbers for the current GPU.
            /// It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
            /// </summary>
            /// <param name="seed">The desired seed.</param>
            public static void manual_seed(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                THSCuda_manual_seed(seed);
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSCuda_manual_seed_all(long seed);

            /// <summary>
            /// Sets the seed for generating random numbers on all GPUs.
            /// It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
            /// </summary>
            /// <param name="seed"></param>
            public static void manual_seed_all(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                THSCuda_manual_seed_all(seed);
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSCuda_synchronize(long device_index);

            /// <summary>
            /// Waits for all kernels in all streams on a CUDA device to complete.
            /// </summary>
            /// <param name="seed">Device for which to synchronize.
            /// It uses the current device, given by current_device(), if a device is not provided.</param>
            public static void synchronize(long seed = -1L)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                THSCuda_synchronize(seed);
            }
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
