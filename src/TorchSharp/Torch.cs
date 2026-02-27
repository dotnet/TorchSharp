// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using TorchSharp.Modules;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
#if LIBTORCH_2_2_2_0
    const string libtorchPackageVersion = "2.2.2.0";
#elif LIBTORCH_2_10_0_0
    const string libtorchPackageVersion = "2.10.0.0";
#elif LIBTORCH_2_7_1_0
    const string libtorchPackageVersion = "2.7.1.0";
#else
#error "Please update libtorchPackageVersion to match LibTorchPackageVersion"
#endif
#if CUDA_12_8
        const string cudaVersion = "12.8";
#else
#error "Please update cudaVersion to match CudaVersionDot"
#endif

        static bool isAppleSilicon =>
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) &&
            RuntimeInformation.OSArchitecture == Architecture.Arm64;

        static string nativeRid =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? (RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "win-arm64" : "win-x64") :
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? $"linux-x64" :
            isAppleSilicon ? "osx-arm64" :
            "any";

        static string nativeGlob =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? @".*\.dll" :
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? @".*\.dylib\.*" :
            // must match
            //   lib.so
            //   lib.so.1
            //   lib.so.11.0
            //   lib.so.11.3
            @".*\.so(\.\d*)*";
        static bool nativeBackendLoaded = false;
        static bool nativeBackendCudaLoaded = false;

        public static string __version__ => libtorchPackageVersion;
        public static string NormalizeNuGetVersion(string versionString)
        {
            if (string.IsNullOrWhiteSpace(versionString))
                throw new ArgumentException($"Invalid NuGet version: {versionString}. Version string is null, empty or only contains whitespaces");

            string[] parts = versionString.Split('-', '+');
            string[] versionParts = parts[0].Split('.');

            if (versionParts.Length < 2 || versionParts.Length > 4 || !versionParts.All(v => int.TryParse(v, out _)))
                throw new ArgumentException($"Invalid NuGet version: {versionString}. Please check: https://learn.microsoft.com/en-us/nuget/concepts/package-versioning");

            string normalizedVersion = versionParts[0] + "." + versionParts[1];
            if (versionParts.Length > 2) normalizedVersion += "." + versionParts[2];
            if (versionParts.Length > 3 && int.Parse(versionParts[3]) != 0) normalizedVersion += "." + versionParts[3];

            if (parts.Length > 1)
                normalizedVersion += "-" + parts[1];

            return normalizedVersion;
        }

        internal static bool TryLoadNativeLibraryFromFile(string path, StringBuilder trace) {
            bool ok;
            try {
                trace.AppendLine($"    Trying to load native component {path}");
                ok = NativeLibrary.TryLoad(path, out var res);
                if (!ok)
                    trace.AppendLine($"    Failed to load native component {path}");
            } catch (Exception exn) {
                trace.AppendLine($"    Failed to load native component {path}: {exn.Message}");
                ok = false;
            }
            return ok;
        }

        internal static bool TryLoadNativeLibraryByName(string name, Assembly assembly, StringBuilder trace)
        {
            bool ok;
            try {
                trace.AppendLine($"    Trying to load native component {name} relative to {assembly.Location}");
                ok = NativeLibrary.TryLoad(name, assembly, null, out var res);
                if (!ok)
                    trace.AppendLine($"    Failed to load native component {name} relative to {assembly.Location}");
            } catch (Exception exn) {
                trace.AppendLine($"    Failed to load native component {name} relative to {assembly.Location}: {exn.Message}");
                ok = false;
            }
            return ok;
        }

        private static void LoadNativeBackend(bool useCudaBackend, out StringBuilder? trace)
        {
            if (!System.Environment.Is64BitProcess) {
                throw new NotSupportedException("TorchSharp only supports 64-bit processes.");
            }

            var alreadyLoaded = useCudaBackend ? nativeBackendCudaLoaded : nativeBackendLoaded;
            trace = null;

            if (!alreadyLoaded) {
                bool ok;
                trace = new StringBuilder();
                trace.AppendLine($"");
                trace.AppendLine($"TorchSharp: LoadNativeBackend: Initialising native backend, useCudaBackend = {useCudaBackend}");
                trace.AppendLine($"");
                trace.AppendLine($"Step 1 - First try regular load of native libtorch binaries.");
                trace.AppendLine($"");

                // Workarounds for weird LibTorch native stuff
                // See https://github.com/pytorch/pytorch/issues/33415

                if (useCudaBackend) {
                    var isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
                    if (isWindows) {
                        trace.AppendLine($"    Try loading Windows cuda native components");
                        // Preloading these DLLs on windows seems to iron out problems where one native DLL
                        // requests a load of another through dynamic linking techniques.
                        //
                        ok = TryLoadNativeLibraryByName("cudnn_adv64_9", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cudnn_cnn64_9", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cudnn_ops64_9", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cudnn_graph64_9.dll", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cudnn_heuristic64_9.dll", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cudnn_engines_precompiled64_9.dll", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cudnn_engines_runtime_compiled64_9.dll", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("nvrtc-builtins64_128", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("caffe2_nvrtc", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("nvrtc64_120_0", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cublasLt64_12", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cufft64_11", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cusparse64_12", typeof(torch).Assembly, trace);
                        ok = TryLoadNativeLibraryByName("cusolver64_11", typeof(torch).Assembly, trace);
                    }

                    ok = TryLoadNativeLibraryByName("torch_cuda", typeof(torch).Assembly, trace);
                    ok = TryLoadNativeLibraryByName("LibTorchSharp", typeof(torch).Assembly, trace);
                } else {
                    ok = TryLoadNativeLibraryByName("torch_cpu", typeof(torch).Assembly, trace);
                    ok = TryLoadNativeLibraryByName("LibTorchSharp", typeof(torch).Assembly, trace);
                }

                trace.AppendLine($"    Result from regular native load of LibTorchSharp is {ok}");

                // Try dynamic load from package directories
                if (!ok) {

                    trace.AppendLine($"");
                    trace.AppendLine($"Step 3 - Alternative load from consolidated directory of native binaries from nuget packages");
                    trace.AppendLine($"");

                    var cpuRootPackage = $"libtorch-cpu-{nativeRid}";
                    var cudaRootPackage = $"libtorch-cuda-{cudaVersion}-{nativeRid}";
                    var target =
                        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "LibTorchSharp.dll" :
                        RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libLibTorchSharp.dylib" :
                        "libLibTorchSharp.so";

                    // See https://github.com/dotnet/TorchSharp/issues/169
                    //
                    // If we are loading in .NET Interactive or F# Interactive, these are in packages in separate
                    // package directories. For managed DLLs this works OK, but native DLLs do not load transitive dependencies.
                    //
                    // So we shadow copy the DLLs into the TorchSharp package, make a copy of the native DLL and continue
                    // with the dynamic load
                    //
                    // Assumed to be in ...\packages\torchsharp\0.3.0-local-debug-20200918\lib\net6.0\TorchSharp.dll
                    //
                    // TODO: on linux make these copies link not shadow-copy
                    var torchsharpLoc = Path.GetDirectoryName(typeof(torch).Assembly.Location);
                    var packagesDir = Path.GetFullPath(Path.Combine(torchsharpLoc!, "..", "..", "..", ".."));
                    var torchsharpHome = Path.GetFullPath(Path.Combine(torchsharpLoc!, "..", ".."));

                    trace.AppendLine($"    torchsharpLoc = {torchsharpLoc}");
                    trace.AppendLine($"    packagesDir = {packagesDir}");
                    trace.AppendLine($"    torchsharpHome = {torchsharpHome}");

                    if (torchsharpLoc!.Contains("torchsharp") && torchsharpLoc.Contains("lib") && Directory.Exists(packagesDir) && Directory.Exists(torchsharpHome)) {

                        var torchSharpVersion = NormalizeNuGetVersion(Path.GetFileName(torchsharpHome));
                        var normalizedLibtorchPackageVersion = NormalizeNuGetVersion(libtorchPackageVersion);
                        if (useCudaBackend) {
                            var consolidatedDir = Path.Combine(torchsharpLoc, $"cuda-{cudaVersion}");

                            trace.AppendLine($"    Trying dynamic load for .NET/F# Interactive by consolidating native {cudaRootPackage}-* binaries to {consolidatedDir}...");

                            var cudaOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, $"{cudaRootPackage}-*", normalizedLibtorchPackageVersion, consolidatedDir, trace);
                            if (cudaOk) {
                                cudaOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, "torchsharp", torchSharpVersion, consolidatedDir, trace);
                                if (cudaOk) {
                                    var consolidated = Path.Combine(consolidatedDir, target);
                                    ok = TryLoadNativeLibraryFromFile(consolidated, trace);
                                }
                            }
                            if (!cudaOk) {
                                var message = $"The {cudaRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cudaRootPackage}, {libtorchPackageVersion}\". Trace from LoadNativeBackend:\n{trace}";
                                Console.WriteLine(message);
                                throw new NotSupportedException(message);
                            }
                        } else {
                            var consolidatedDir = Path.Combine(torchsharpLoc, $"cpu");

                            trace.AppendLine($"    Trying dynamic load for .NET/F# Interactive by consolidating native {cpuRootPackage}-* binaries to {consolidatedDir}...");

                            var cpuOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, cpuRootPackage, normalizedLibtorchPackageVersion, consolidatedDir, trace);
                            if (cpuOk) {
                                cpuOk = CopyNativeComponentsIntoSingleDirectory(packagesDir, "torchsharp", torchSharpVersion, consolidatedDir, trace);
                                if (cpuOk) {
                                    var consolidated = Path.Combine(consolidatedDir, target);
                                    ok = TryLoadNativeLibraryFromFile(consolidated, trace);
                                }
                            }
                            if (!cpuOk) {
                                var message = $"The {cpuRootPackage} package version {libtorchPackageVersion} is not restored on this system. If using F# Interactive or .NET Interactive you may need to add a reference to this package, e.g. \n    #r \"nuget: {cpuRootPackage}, {libtorchPackageVersion}\". Trace from LoadNativeBackend:\n{trace}";
                                Console.WriteLine(message);
                                throw new NotSupportedException(message);
                            }
                        }
                    }
                    else {
                        trace.AppendLine("    Giving up, TorchSharp.dll does not appear to have been loaded from package directories");
                    }
                    if (!ok) {
                        var message = $"This application or script uses TorchSharp but doesn't contain a reference to {(useCudaBackend ? cudaRootPackage : cpuRootPackage)}, Version={libtorchPackageVersion}.\n\nConsider referencing one of the combination packages TorchSharp-cpu, TorchSharp-cuda-linux, TorchSharp-cuda-windows or call System.Runtime.InteropServices.NativeLibrary.Load(path-to-{target}) explicitly for a Python install of pytorch. See https://github.com/dotnet/TorchSharp/issues/169.\".\n\nFor CUDA, you may need to call 'TorchSharp.torch.InitializeDeviceType(TorchSharp.DeviceType.CUDA)' before any use of TorchSharp CUDA packages from scripts or notebooks.\n\nTrace from LoadNativeBackend:\n{trace}";
                        Console.WriteLine(message);
                        throw new NotSupportedException(message);
                    }
                }


                // Record the successful load
                if (useCudaBackend)
                    nativeBackendCudaLoaded = true;
                else
                    nativeBackendLoaded = true;
            }
        }

        /// Copy all native runtime DLLs into single directory if it hasn't been done already
        private static bool CopyNativeComponentsIntoSingleDirectory(string packagesDir,
            string packagePattern,
            string packageVersion,
            string target,
            StringBuilder trace)
        {
            // Some loads will fail due to missing dependencies but then
            // these will be resolved in subsequent iterations.
            trace.AppendLine($"    Consolidating native binaries, packagesDir={packagesDir}, packagePattern={packagePattern}, packageVersion={packageVersion} to target={target}...");
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
                        trace.AppendLine($"    CopyNativeComponentsIntoSingleDirectory: natives={natives}");
                        if (Directory.Exists(natives)) {
                            var nativeRegExp = new Regex("^" + nativeGlob + "$");
                            foreach (var file in Directory.GetFiles(natives).Where(path => nativeRegExp.IsMatch(path))) {
                                var targetFile = Path.Combine(target, Path.GetFileName(file));
                                if (!File.Exists(targetFile)) {
                                    trace.AppendLine($"Copy {file} --> {targetFile}");
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

        public static bool TryInitializeDeviceType(DeviceType deviceType)
        {
            if (deviceType == DeviceType.MPS && !isAppleSilicon)
            {
                return false;
            }

            LoadNativeBackend(deviceType == DeviceType.CUDA, out _);
            if (deviceType == DeviceType.CUDA) {
                return cuda.CallTorchCudaIsAvailable();
            } else {
                return true;
            }
        }

        public static void InitializeDeviceType(DeviceType deviceType)
        {
            if (deviceType == DeviceType.MPS && !isAppleSilicon)
            {
                throw new InvalidOperationException($"Torch device type 'MPS' is not available on this platform.");
            }

            LoadNativeBackend(deviceType == DeviceType.CUDA, out var trace);
            if (deviceType == DeviceType.CUDA) {

                // For CUDA, we double-check that CudaIsAvailable actually returns true.
                // If it doesn't we report the entire load trace.
                var result = cuda.CallTorchCudaIsAvailable();
                if (!result)
                    throw new InvalidOperationException($"Torch device type {deviceType} did not initialise on the current machine. Trace from LoadNativeBackend:\n{trace?.ToString() ?? string.Empty}");
            }
        }

        public static Device InitializeDevice(Device? device)
        {
            if (device is null)
            {
                device = get_default_device();
            }
            InitializeDeviceType(device.type);
            return device;
        }

        public static partial class random
        {
            /// <summary>
            /// Sets the seed for generating random numbers to a non-deterministic random number. Returns a 64 bit number used to seed the RNG.
            /// </summary>
            public static long seed() => Generator.Default.seed();

            /// <summary>
            /// Returns the initial seed for generating random numbers.
            /// </summary>
            public static long initial_seed() => Generator.Default.initial_seed();

            /// <summary>
            /// Sets the seed for generating random numbers. Returns a torch.Generator object.
            /// </summary>
            /// <param name="seed">The desired seed.</param>
            /// <returns></returns>
            public static Generator manual_seed(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                var res = THSGenerator_manual_seed(seed);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Generator(res);
            }

            /// <summary>
            /// Returns the random number generator state as a torch.ByteTensor.
            /// </summary>
            /// <returns></returns>
            public static Tensor get_rng_state()
            {
                return Generator.Default.get_state();
            }
            /// <summary>
            /// Sets the random number generator state.
            /// </summary>
            /// <param name="new_state">The desired state</param>
            public static void set_rng_state(Tensor new_state)
            {
                Generator.Default.set_state(new_state);
            }
        }

        public static partial class nn
        {
            public static partial class utils
            {
                /// <summary>
                /// Clips gradient norm of an iterable of parameters.
                /// The norm is computed over all gradients together, as if they were concatenated into a single vector.
                /// </summary>
                /// <param name="tensors"></param>
                /// <param name="max_norm"></param>
                /// <param name="norm_type"></param>
                /// <remarks>Gradients are modified in-place.</remarks>
                public static double clip_grad_norm_(IEnumerable<Modules.Parameter> tensors, double max_norm, double norm_type = 2.0)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorsRef = parray.CreateArray(tensors.ToHandleArray());
                        var value = THSTensor_clip_grad_norm_(tensorsRef, parray.Array.Length, max_norm, norm_type);
                        CheckForErrors();
                        return value;
                    }
                }

                /// <summary>
                /// Clips gradient of an iterable of parameters at specified value.
                /// </summary>
                /// <param name="tensors">An enumeration of Tensors that will have gradients normalized</param>
                /// <param name="clip_value">Maximum allowed value of the gradients. The gradients are clipped in the range [-clip_value,clip_value]</param>
                /// <remarks>Gradients are modified in-place.</remarks>
                public static void clip_grad_value_(IEnumerable<Modules.Parameter> tensors, double clip_value)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorsRef = parray.CreateArray(tensors.ToHandleArray());
                        THSTensor_clip_grad_value_(tensorsRef, parray.Array.Length, clip_value);
                        CheckForErrors();
                    }
                }

                /// <summary>
                /// Convert parameters to one vector
                /// </summary>
                /// <param name="tensors">An enumeration of Tensors that are the parameters of a model.</param>
                /// <returns>A one-dimensional tensor with the values of all the parameters.</returns>
                public static Tensor parameters_to_vector(IEnumerable<Modules.Parameter> tensors)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorsRef = parray.CreateArray(tensors.ToHandleArray());

                        var res = THSTensor_parameters_to_vector(tensorsRef, parray.Array.Length);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }

                /// <summary>
                /// Convert one vector to parameters.
                /// </summary>
                /// <param name="vec">a single vector represents the parameters of a model.</param>
                /// <param name="tensors">An enumeration of Tensors that are the parameters of a model.</param>
                /// <returns>A one-dimensional tensor with the values of all the parameters.</returns>
                public static void vector_to_parameters(Tensor vec, IEnumerable<Modules.Parameter> tensors)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorsRef = parray.CreateArray(tensors.ToHandleArray());

                        THSTensor_vector_to_parameters(vec.Handle, tensorsRef, parray.Array.Length);
                        CheckForErrors();
                    }
                }

                /// <summary>
                /// Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.
                /// </summary>
                /// <param name="conv_w">Convolutional weight.</param>
                /// <param name="conv_b">Convolutional bias.</param>
                /// <param name="bn_rm">BatchNorm running mean.</param>
                /// <param name="bn_rv">BatchNorm running variance.</param>
                /// <param name="bn_eps">BatchNorm epsilon.</param>
                /// <param name="bn_w">BatchNorm weight.</param>
                /// <param name="bn_b">BatchNorm bias.</param>
                /// <param name="transpose">If <c>true</c>, transpose the conv weight. Defaults to <c>false</c>.</param>
                /// <returns>Fused convolutional weight and bias.</returns>
                public static (Parameter weight, Parameter bias) fuse_conv_bn_weights(
                    Tensor conv_w, Tensor? conv_b,
                    Tensor bn_rm, Tensor bn_rv, double bn_eps,
                    Tensor? bn_w, Tensor? bn_b,
                    bool transpose = false)
                {
                    using var scope = NewDisposeScope();

                    var conv_weight_dtype = conv_w.dtype;
                    var conv_bias_dtype = conv_b?.dtype ?? conv_weight_dtype;
                    conv_b ??= zeros_like(bn_rm);
                    bn_w ??= ones_like(bn_rm);
                    bn_b ??= zeros_like(bn_rm);
                    var shape = conv_w.shape.Select(_ => 1L).ToArray();
                    if (transpose)
                        shape[1] = -1;
                    else
                        shape[0] = -1;

                    var bn_var_rsqrt = rsqrt(bn_rv + bn_eps);
                    var fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape))
                        .to(conv_weight_dtype);
                    var fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b)
                        .to(conv_bias_dtype);

                    var weight = new Parameter(fused_conv_w, conv_w.requires_grad);
                    var bias = new Parameter(fused_conv_b, conv_b.requires_grad);

                    return scope.MoveToOuter(weight, bias);
                }

                /// <summary>
                /// Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.
                /// </summary>
                /// <param name="linear_w">Linear weight.</param>
                /// <param name="linear_b">Linear bias.</param>
                /// <param name="bn_rm">BatchNorm running mean.</param>
                /// <param name="bn_rv">BatchNorm running variance.</param>
                /// <param name="bn_eps">BatchNorm epsilon.</param>
                /// <param name="bn_w">BatchNorm weight.</param>
                /// <param name="bn_b">BatchNorm bias.</param>
                /// <returns>Fused linear weight and bias.</returns>
                public static (Parameter weight, Parameter bias) fuse_linear_bn_weights(
                    Tensor linear_w, Tensor? linear_b,
                    Tensor bn_rm, Tensor bn_rv, double bn_eps,
                    Tensor bn_w, Tensor bn_b)
                {
                    using var scope = NewDisposeScope();

                    linear_b ??= zeros_like(bn_rm);

                    var bn_scale = bn_w * rsqrt(bn_rv + bn_eps);

                    var fused_w = linear_w * bn_scale.unsqueeze(-1);
                    var fused_b = (linear_b - bn_rm) * bn_scale + bn_b;

                    var weight = new Parameter(fused_w, linear_w.requires_grad);
                    var bias = new Parameter(fused_b, linear_b.requires_grad);

                    return scope.MoveToOuter(weight, bias);
                }

                public static Linear fuse_linear_bn_eval(Linear linear, BatchNorm bn)
                {
                    if (linear.training || bn.training)
                        throw new InvalidOperationException("Fusing operators is valid only for eval mode.");

                    var (weight, bias) = fuse_linear_bn_weights(linear.weight, linear.bias, bn.running_mean!, bn.running_var!, bn.eps, bn.weight, bn.bias!);

                    return Linear(weight, bias);
                }
            }
        }

        public static partial class cuda
        {

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

            /// <summary>
            /// Returns a bool indicating if CUDNN is currently available.
            /// </summary>
            public static bool is_cudnn_available()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return THSTorchCuda_cudnn_is_available();
            }

            /// <summary>
            /// Returns the number of GPUs available.
            /// </summary>
            /// <returns></returns>
            public static int device_count()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return THSTorchCuda_device_count();
            }

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

            /// <summary>
            /// Waits for all kernels in all streams on a CUDA device to complete.
            /// </summary>
            /// <param name="device">Device for which to synchronize.
            /// It uses the current device, given by current_device(), if a device is not provided.</param>
            public static void synchronize(Device? device = null)
            {
                TryInitializeDeviceType(device?.type ?? DeviceType.CUDA);
                THSTorchCuda_synchronize(device?.index ?? -1);
            }
        }

        /// <summary>
        /// Workaround for F# issue.
        /// </summary>
        public static bool cuda_is_available() => torch.cuda.is_available();

        /// <summary>
        /// Check whether MPS is available
        /// </summary>
        public static bool mps_is_available() => torch.isAppleSilicon;

        public static void CheckForErrors()
        {
            var error = THSTorch_get_and_reset_last_err();

            if (error != IntPtr.Zero)
            {
                throw new ExternalException(Marshal.PtrToStringAnsi(error));
            }
        }

        public static partial class backends
        {
            public static partial class cuda
            {
                public static partial class matmul
                {
                    public static bool allow_tf32 {
                        get {
                            var result = NativeMethods.THSBackend_cublas_get_allow_tf32();
                            CheckForErrors();
                            return result;
                        }
                        set {
                            NativeMethods.THSBackend_cublas_set_allow_tf32(value);
                            CheckForErrors();
                        }
                    }

                    public static bool allow_fp16_reduced_precision_reduction {
                        get {
                            var result = NativeMethods.THSBackend_cuda_get_allow_fp16_reduced_precision_reduction();
                            CheckForErrors();
                            return result;
                        }
                        set {
                            NativeMethods.THSBackend_cuda_set_allow_fp16_reduced_precision_reduction(value);
                            CheckForErrors();
                        }
                    }
                }

                public static bool flash_sdp_enabled()
                {
                    var result = NativeMethods.THSBackend_cuda_get_enable_flash_sdp();
                    CheckForErrors();
                    return result;
                }

                public static void enable_flash_sdp(bool enable)
                {
                    NativeMethods.THSBackend_cuda_set_enable_flash_sdp(enable);
                    CheckForErrors();
                }

                public static bool math_sdp_enabled()
                {
                    var result = NativeMethods.THSBackend_cuda_get_enable_math_sdp();
                    CheckForErrors();
                    return result;
                }

                public static void enable_math_sdp(bool enable)
                {
                    NativeMethods.THSBackend_cuda_set_enable_math_sdp(enable);
                    CheckForErrors();
                }
            }

            public static partial class cudnn
            {
                public static bool allow_tf32 {
                    get {
                        var result = NativeMethods.THSBackend_cudnn_get_allow_tf32();
                        CheckForErrors();
                        return result;
                    }
                    set {
                        NativeMethods.THSBackend_cudnn_set_allow_tf32(value);
                        CheckForErrors();
                    }
                }
            }
        }
    }

    /// <summary>
    /// The LibTorch device types.
    /// </summary>
    /// <remarks>TorchSharp currently only supports CPU and CUDA.</remarks>
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
        XLA = 9, // XLA / TPU
        MPS = 13, // Apple Silicon
        META = 14,
    }
}
