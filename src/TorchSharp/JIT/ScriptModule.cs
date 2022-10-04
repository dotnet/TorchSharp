// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using System.Net;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class jit
        {
            /// <summary>
            /// This class represents a TorchScript module.
            /// </summary>
            public class ScriptModule : torch.nn.Module
            {
                internal ScriptModule(IntPtr handle) : base(new HType(handle, true, THSJIT_Module_dispose), null)
                {
                }

                ~ScriptModule()
                {
                    Dispose(false);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_dispose(HType handle);

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_named_parameters(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                protected override (string name, TorchSharp.Modules.Parameter parameter)[] _named_parameters()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSJIT_Module_named_parameters(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new TorchSharp.Modules.Parameter(x))).ToArray();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_named_buffers(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                protected override (string name, Tensor buffer)[] _named_buffers()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSJIT_Module_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Tensor(x))).ToArray();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_named_modules(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                /// <summary>
                /// Returns an enumerable of all modules in the network, yielding both the name of the module as well as the module itself.
                /// </summary>
                /// <returns>(string, Module) – Tuple of name and module</returns>
                public override IEnumerable<(string name, nn.Module module)> named_modules()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSJIT_Module_named_modules(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new ScriptModule(x) as nn.Module)).Where(m => !String.IsNullOrEmpty(m.Item1));
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_named_children(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                /// <summary>
                /// Returns an enumerable of immediate children modules, yielding both the name of the module as well as the module itself.
                /// </summary>
                /// <returns>(string, Module) – Tuple containing a name and child module</returns>
                public override IEnumerable<(string name, nn.Module module)> named_children()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSJIT_Module_named_children(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new ScriptModule(x) as nn.Module));
                }

                [DllImport("LibTorchSharp")]
                private static extern long THSJIT_getNumModules(HType module);

                [DllImport("LibTorchSharp")]
                private static extern int THSJIT_Module_num_inputs(HType module);

                public int GetNumberOfInputs()
                {
                    return THSJIT_Module_num_inputs(handle);
                }

                [DllImport("LibTorchSharp")]
                private static extern int THSJIT_Module_num_outputs(HType module);

                public int GetNumberOfOutputs()
                {
                    return THSJIT_Module_num_outputs(handle);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_train(HType module, bool on);

                /// <summary>
                /// Sets the module in evaluation mode.
                /// </summary>
                /// <remarks>
                /// Any script module that was created using torch.jit.trace() will be unaffected. The behavior of such
                /// modules will be captured when traced.
                /// </remarks>
                public override void train(bool on = true)
                {
                    THSJIT_Module_train(handle, on);
                    CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_eval(HType module);

                /// <summary>
                /// Sets the module in evaluation mode.
                /// </summary>
                /// <remarks>
                /// Any script module that was created using torch.jit.trace() will be unaffected. The behavior of such
                /// modules will be captured when traced.
                /// </remarks>
                public override void eval()
                {
                    THSJIT_Module_eval(handle);
                    CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern bool THSJIT_Module_is_training(HType module);

                /// <summary>
                /// Check whether the module is set to training or evaluation mode.
                /// </summary>
                public override bool training {
                    get {
                        var res = THSJIT_Module_is_training(handle);
                        CheckForErrors();
                        return res;
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern void THSJIT_Module_to_device(HType module, long deviceType, long deviceIndex);

                [DllImport("LibTorchSharp")]
                static extern void THSJIT_Module_to_device_dtype(HType module, sbyte dtype, long deviceType, long deviceIndex);

                [DllImport("LibTorchSharp")]
                static extern void THSJIT_Module_to_dtype(HType module, sbyte dtype);

                internal protected override nn.Module _to(Device device, ScalarType dtype)
                {
                    if (device.type != DeviceType.CUDA) { device = new Device(device.type, -1); };

                    if (device.type == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");


                    InitializeDeviceType(device.type);
                    THSJIT_Module_to_device_dtype(handle, (sbyte)dtype, (int)device.type, device.index);
                    CheckForErrors();

                    _toEpilog(device, dtype);

                    return this;
                }

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
                /// <param name="deviceIndex">The optional device index.</param>
                /// <returns></returns>
                internal protected override nn.Module _to(DeviceType deviceType, int deviceIndex = -1)
                {
                    if (deviceType != DeviceType.CUDA) deviceIndex = -1;

                    if (deviceType == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                    if (deviceType != _deviceType || deviceIndex != _deviceIndex) {

                        InitializeDeviceType(deviceType);
                        THSJIT_Module_to_device(handle, (int)deviceType, deviceIndex);
                        CheckForErrors();

                        _toEpilog(deviceType, deviceIndex);
                    }

                    Debug.Assert(_deviceType == DeviceType.CUDA || _deviceIndex == -1);

                    return this;
                }

                private DeviceType _deviceType = DeviceType.CPU;
                private int _deviceIndex = -1;

                /// <summary>
                /// Convert the parameters and buffers.
                /// </summary>
                /// <returns></returns>
                internal protected override nn.Module _to(ScalarType dtype)
                {
                    THSJIT_Module_to_dtype(handle, (sbyte)dtype);
                    CheckForErrors();

                    _toEpilog(dtype);

                    return this;
                }

#if false   // These functions "work," but the native code doesn't seem to find any interesting information.

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSJIT_Module_getInputType(HType module, int index);

                public Type GetInputType(int index)
                {
                    var type = new Type(THSJIT_Module_getInputType(handle, index), Type.TypeKind.AnyType);

                    return GetType(type);
                }

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSJIT_getOutputType(HType module, int index);

                public Type GetOutputType(int index)
                {
                    var type = new Type(THSJIT_getOutputType(handle, index), Type.TypeKind.AnyType);

                    return GetType(type);
                }

                private Type GetType(Type type)
                {
                    switch (type.Kind) {
                    case Type.TypeKind.TensorType:
                        var dynamic = type.AsTensorType();
                        type.Dispose();
                        return dynamic;
                    //case Type.TypeKind.DimensionedTensorType:
                    //    var tensor = type.AsTensorType();
                    //    type.Dispose();
                    //    return tensor;
                    default:
                        return type;
                    }
                }
#endif

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_forward(HType module, IntPtr tensors, int length, AllocatePinnedArray allocator, out sbyte typeCode);

                /// <summary>
                /// Invoke the 'forward' function of the script with any number of arguments.
                /// </summary>
                /// <param name="objs"></param>
                /// <returns></returns>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                /// <exception cref="NotImplementedException"></exception>
                public object forward(params object[] objs)
                {
                    TensorOrScalar[] ptrArray = null;
                    sbyte typeCode = 0;

                    using (var parray = new PinnedArray<TensorOrScalar>()) {

                        DetermineArgumentTypeRefs(objs, out int count, out TensorOrScalar[] tensorRefs);

                        THSJIT_Module_forward(handle, parray.CreateArray(tensorRefs), count, parray.CreateArray, out typeCode);
                        torch.CheckForErrors();
                        ptrArray = parray.Array;
                    }

                    return ProcessReturnValue(name, ptrArray, typeCode);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_invoke(HType module, string name, IntPtr tensors, int length, AllocatePinnedArray allocator, out sbyte typeCode);

                [StructLayout(LayoutKind.Sequential)]
                internal struct TensorOrScalar
                {
                    public long TypeCode;
                    public IntPtr Handle;
                }

                /// <summary>
                /// Invoke a function from the script module.
                /// </summary>
                /// <param name="name">The name of the function.</param>
                /// <param name="objs">Function arguments.</param>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public object invoke(string name, params object[] objs)
                {
                    if (String.IsNullOrEmpty(name)) throw new ArgumentNullException("method name");

                    //if (!objs.All(o => typeof(Tensor).IsAssignableFrom(o.GetType()))) {
                    //    throw new NotImplementedException($"ScriptModule.{name}() is not yet taking non-tensors as input arguments");
                    //}

                    TensorOrScalar[] ptrArray = null;
                    sbyte typeCode = 0;

                    using (var parray = new PinnedArray<TensorOrScalar>()) {

                        DetermineArgumentTypeRefs(objs, out int count, out TensorOrScalar[] tensorRefs);

                        THSJIT_Module_invoke(handle, name, parray.CreateArray(tensorRefs), count, parray.CreateArray, out typeCode);
                        torch.CheckForErrors();
                        ptrArray = parray.Array;
                    }

                    return ProcessReturnValue(name, ptrArray, typeCode);
                }

                internal static void DetermineArgumentTypeRefs(object[] objs, out int count, out TensorOrScalar[] tensorRefs)
                {
                    count = objs.Length;
                    tensorRefs = new TensorOrScalar[count];
                    for (var idx = 0; idx < objs.Length; idx++) {
                        switch (objs[idx]) {
                        case Tensor t:
                            tensorRefs[idx].Handle = t.Handle;
                            tensorRefs[idx].TypeCode = 0;
                            break;
                        case Scalar s:
                            tensorRefs[idx].Handle = s.Handle;
                            tensorRefs[idx].TypeCode = 1;
                            break;
                        case float f:
                            tensorRefs[idx].Handle = ((Scalar)f).Handle;
                            tensorRefs[idx].TypeCode = 1;
                            break;
                        case double d:
                            tensorRefs[idx].Handle = ((Scalar)d).Handle;
                            tensorRefs[idx].TypeCode = 1;
                            break;
                        case bool i:
                            tensorRefs[idx].Handle = (IntPtr)(i ? 1L : 0L);
                            tensorRefs[idx].TypeCode = 2;
                            break;
                        case int i:
                            tensorRefs[idx].Handle = (IntPtr)i;
                            tensorRefs[idx].TypeCode = 3;
                            break;
                        case long l:
                            tensorRefs[idx].Handle = ((Scalar)l).Handle;
                            tensorRefs[idx].TypeCode = 1;
                            // The MacOS version of Clang doesn't like the use of int64_t, so pass as a Scalar instance, instead.
                            //tensorRefs[idx].Handle = (IntPtr)l;
                            //tensorRefs[idx].TypeCode = 4;
                            break;
                        default:
                            throw new NotImplementedException($"Passing arguments of type {objs[idx].GetType().Name} to TorchScript.");
                        }
                    }
                }

                internal static object ProcessReturnValue(string name, TensorOrScalar[] ptrArray, sbyte typeCode)
                {
                    switch (typeCode) {
                    default:
                        // Nothing.
                        throw new NotImplementedException($"ScriptModule.{name}() returning something else than a tensor, a tuple of tensors, or list of tensors.");
                    case 1:
                        // Tensor
                        return new Tensor(ptrArray[0].Handle);
                    case 2:
                        // Tuple
                        switch (ptrArray.Length) {
                        case 1:
                            return new Tensor(ptrArray[0].Handle);
                        case 2:
                            return (new Tensor(ptrArray[0].Handle), new Tensor(ptrArray[1].Handle));
                        case 3:
                            return (new Tensor(ptrArray[0].Handle), new Tensor(ptrArray[1].Handle), new Tensor(ptrArray[2].Handle));
                        case 4:
                            return (new Tensor(ptrArray[0].Handle), new Tensor(ptrArray[1].Handle), new Tensor(ptrArray[2].Handle), new Tensor(ptrArray[3].Handle));
                        case 5:
                            return (new Tensor(ptrArray[0].Handle), new Tensor(ptrArray[1].Handle), new Tensor(ptrArray[2].Handle), new Tensor(ptrArray[3].Handle), new Tensor(ptrArray[4].Handle));
                        default: {
                                // Too long a tuple, return as a list, instead.
                                var result = new Tensor[ptrArray.Length];
                                for (var i = 0; i < ptrArray.Length; i++) {
                                    result[i] = new Tensor(ptrArray[i].Handle);
                                }
                                return result;
                            }
                        }
                    case 3: {
                            // List of tensors
                            var result = new Tensor[ptrArray.Length];
                            for (var i = 0; i < ptrArray.Length; i++) {
                                result[i] = new Tensor(ptrArray[i].Handle);
                            }
                            return result;
                        }
                    case 4:
                        // Scalar
                        return new Scalar(ptrArray[0].Handle);
                    case 5:
                        // Scalar tuple
                        switch (ptrArray.Length) {
                        case 1:
                            return new Scalar(ptrArray[0].Handle);
                        case 2:
                            return (new Scalar(ptrArray[0].Handle), new Scalar(ptrArray[1].Handle));
                        case 3:
                            return (new Scalar(ptrArray[0].Handle), new Scalar(ptrArray[1].Handle), new Scalar(ptrArray[2].Handle));
                        case 4:
                            return (new Scalar(ptrArray[0].Handle), new Scalar(ptrArray[1].Handle), new Scalar(ptrArray[2].Handle), new Scalar(ptrArray[3].Handle));
                        case 5:
                            return (new Scalar(ptrArray[0].Handle), new Scalar(ptrArray[1].Handle), new Scalar(ptrArray[2].Handle), new Scalar(ptrArray[3].Handle), new Scalar(ptrArray[4].Handle));
                        default: {
                                // Too long a tuple, return as a list, instead.
                                var result = new Scalar[ptrArray.Length];
                                for (var i = 0; i < ptrArray.Length; i++) {
                                    result[i] = new Scalar(ptrArray[i].Handle);
                                }
                                return result;
                            }
                        }
                    case 6: {
                            // List of scalars
                            var result = new Scalar[ptrArray.Length];
                            for (var i = 0; i < ptrArray.Length; i++) {
                                result[i] = new Scalar(ptrArray[i].Handle);
                            }
                            return result;
                        }
                    case 7: {
                            // List of scalars and tensors
                            var result = new object[ptrArray.Length];
                            for (var i = 0; i < ptrArray.Length; i++) {
                                result[i] = ptrArray[i].TypeCode == 0 ? new Tensor(ptrArray[i].Handle) : new Scalar(ptrArray[i].Handle);
                            }
                            return result;
                        }
                    }
                }

                /// <summary>
                /// Invoke a function from the script module.
                /// </summary>
                /// <typeparam name="TResult">The return type of the TorchScript function.</typeparam>
                /// <param name="name">The name of the function.</param>
                /// <param name="inputs">Function arguments.</param>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public TResult invoke<TResult>(string name, params object[] inputs) => (TResult)invoke(name, inputs);

                /// <summary>
                /// Invoke a function from the script module.
                /// </summary>
                /// <typeparam name="T">The type of all function arguments.</typeparam>
                /// <typeparam name="TResult">The return type of the TorchScript function.</typeparam>
                /// <param name="name">The name of the function.</param>
                /// <param name="inputs">Function arguments.</param>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public TResult invoke<T, TResult>(string name, params T[] inputs) => (TResult)invoke(name, inputs);
            }

            /// <summary>
            /// A script module taking any number of tensors as input
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            public class ScriptModule<TResult> : ScriptModule, torch.nn.IModule<Tensor[], TResult>
            {
                internal ScriptModule(IntPtr handle) : base(handle) { }

                /// <summary>
                /// Invoke the 'forward' function of the script with one tensor as its argument
                /// </summary>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public TResult forward(params Tensor[] tensor)
                {
                    return (TResult)base.forward(tensor);
                }
            }

            /// <summary>
            /// A script module taking a single argument.
            /// </summary>
            /// <typeparam name="T">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            public class ScriptModule<T, TResult> : ScriptModule, torch.nn.IModule<T, TResult>
            {
                internal ScriptModule(IntPtr handle) : base(handle) { }

                /// <summary>
                /// Invoke the 'forward' function of the script with one tensor as its argument
                /// </summary>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public TResult forward(T tensor)
                {
                    return (TResult)base.forward(tensor);
                }
            }

            /// <summary>
            /// A script module taking two arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            public class ScriptModule<T1, T2, TResult> : ScriptModule, torch.nn.IModule<T1, T2, TResult>
            {
                internal ScriptModule(IntPtr handle) : base(handle) { }

                /// <summary>
                /// Invoke the 'forward' function of the script with one tensor as its argument
                /// </summary>
                /// <remarks>
                /// Only certain types can currently be passed:
                /// 1. Tensor
                /// 2. Scalar
                /// 3. int/long
                /// 4. double/float
                /// 5. bool
                ///
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public TResult forward(T1 input1, T2 input2)
                {
                    return (TResult)base.forward(input1, input2);
                }
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSJIT_load(string filename);

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="filename">The file name of the module.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule load(string filename)
            {
                return new ScriptModule(_load(filename));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<TResult> load<TResult>(string filename)
            {
                return new ScriptModule<TResult>(_load(filename));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, TResult> load<T1, TResult>(string filename)
            {
                return new ScriptModule<T1, TResult>(_load(filename));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(string filename)
            {
                return new ScriptModule<T1, T2, TResult>(_load(filename));
            }

            private static IntPtr _load(string filename)
            {
                if (!System.IO.File.Exists(filename))
                    throw new System.IO.FileNotFoundException(filename);

                var result = THSJIT_load(filename);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return result;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSJIT_save(nn.Module.HType handle, string filename);

            /// <summary>
            /// Save an offline version of a previously loaded script module.
            /// 
            /// The saved module serializes all of the methods, submodules, parameters, and attributes of this module.
            /// It can be loaded into the C++ API using torch::jit::load(filename) or into the .NET API with torch.jit.load().
            /// </summary>
            /// <param name="module">The script module to save.</param>
            /// <param name="filename">The file name of the module.</param>
            public static void save(ScriptModule module, string filename)
            {
                THSJIT_save(module.handle, filename);
                CheckForErrors();
            }

        }
    }
}
