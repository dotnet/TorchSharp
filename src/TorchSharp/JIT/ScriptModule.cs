// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class jit
        {
            /// <summary>
            /// This class represents a TorchScript module.
            /// </summary>
            public class ScriptModule : nn.Module
            {
                internal ScriptModule(IntPtr handle) : base(new HType(handle, true, THSJIT_Module_dispose), null)
                {
                }

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

                public (string name, Tensor buffer)[] named_attributes(bool recurse = true)
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSJIT_Module_named_attributes(handle, recurse, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Tensor(x))).ToArray();
                }

                public void set_attribute(string name, Tensor buffer)
                {
                    THSJIT_Module_set_attribute(handle, name, buffer.Handle);
                    CheckForErrors();
                }

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

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new ScriptModule(x) as nn.Module)).Where(m => !string.IsNullOrEmpty(m.Item1));
                }

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

                public int GetNumberOfInputs()
                {
                    return THSJIT_Module_num_inputs(handle);
                }

                public int GetNumberOfOutputs()
                {
                    return THSJIT_Module_num_outputs(handle);
                }

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

                protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking)
                {
                    if (device.type != DeviceType.CUDA) { device = new Device(device.type, -1); };

                    if (device.type == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");


                    InitializeDeviceType(device.type);
                    THSJIT_Module_to_device_dtype(handle, (sbyte)dtype, (int)device.type, device.index);
                    CheckForErrors();

                    _toEpilog(device, dtype, non_blocking);
                    _toScriptEpilog(device, dtype, non_blocking);
                    return this;
                }

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
                /// <param name="deviceIndex">The optional device index.</param>
                /// <param name="non_blocking">
                /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible, 
                /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
                /// </param>
                /// <returns></returns>
                protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
                {
                    if (deviceType != DeviceType.CUDA) deviceIndex = -1;

                    if (deviceType == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                    if (deviceType != _deviceType || deviceIndex != _deviceIndex) {

                        InitializeDeviceType(deviceType);
                        THSJIT_Module_to_device(handle, (int)deviceType, deviceIndex);
                        CheckForErrors();

                        _toEpilog(deviceType, deviceIndex, non_blocking);
                        _toScriptEpilog(deviceType, deviceIndex, non_blocking);
                    }

                    Debug.Assert(_deviceType == DeviceType.CUDA || _deviceIndex == -1);

                    return this;
                }

                /// <summary>
                /// Convert the parameters and buffers.
                /// </summary>
                /// <returns></returns>
                protected internal override nn.Module _to(ScalarType dtype, bool non_blocking)
                {
                    THSJIT_Module_to_dtype(handle, (sbyte)dtype);
                    CheckForErrors();

                    _toEpilog(dtype, non_blocking);
                    _toScriptEpilog(dtype, non_blocking);

                    return this;
                }

                protected void _toScriptEpilog(ScalarType dtype, bool non_blocking)
                {
                    _toScriptEpilog(dtype, null, non_blocking);
                }

                protected void _toScriptEpilog(Device device, ScalarType dtype, bool non_blocking)
                {
                    _toScriptEpilog(dtype, device, non_blocking);
                }

                protected void _toScriptEpilog(DeviceType deviceType, int deviceIndex, bool non_blocking)
                {
                    _toScriptEpilog(null, new Device(deviceType, deviceIndex), non_blocking);
                }

                private void _toScriptEpilog(ScalarType? dtype, Device device, bool non_blocking)
                {
                    foreach (var (name, buffer) in named_attributes(recurse: false)) {
                        if (name is null || !buffer.toWillCopy(dtype ?? buffer.dtype, device ?? buffer.device)) continue;

                        set_attribute(name, buffer.to(dtype ?? buffer.dtype, device ?? buffer.device, disposeAfter: true, non_blocking: non_blocking));
                    }
                }

#if false   // These functions "work," but the native code doesn't seem to find any interesting information.

                public Type GetInputType(int index)
                {
                    var type = new Type(THSJIT_Module_getInputType(handle, index), Type.TypeKind.AnyType);

                    return GetType(type);
                }

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

                /// <summary>
                /// Invoke the 'forward' function of the script with any number of arguments.
                /// </summary>
                /// <param name="input">Any number of parameters for the forward function.</param>
                /// <returns>An object.</returns>
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
                public object forward(params object[] input)
                {
                    TensorOrScalar[] ptrArray = null;
                    sbyte typeCode = 0;

                    using (var ntosArray = new NativeTensorOrScalarIndexedArray()) {

                        var tRefsHandle = DetermineArgumentTypeRefs(input, out var count, ntosArray);

                        var allocated = ntosArray.Count;

                        THSJIT_Module_forward(handle, tRefsHandle, count, ntosArray.CreateArray, out typeCode, allocated);
                        torch.CheckForErrors();
                        ptrArray = ntosArray.ToToSArray(allocated);

                        return ProcessReturnValue(name, ntosArray, ptrArray, typeCode);
                    }
                }

                /// <summary>
                /// Synonym for `forward`
                /// </summary>
                /// <param name="input">Any number of parameters for the forward function.</param>
                /// <returns>An object.</returns>
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
                ///
                /// Note: this currently does not support hooking the module.
                /// </remarks>
                public object call(params object[] input)
                {
                    // TODO: Call pre-hooks, if available.

                    var result = forward(input);

                    // TODO: Call post-hooks, if available.

                    return result;
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
                /// 6. 'null' object
                /// 
                /// Only certain types can currently be returned:
                /// 1. Tensor / Scalar
                /// 2. Tuple of Tensor / Scalar
                /// 3. Array (Python list) of Tensor / Scalar
                /// 4. null object
                ///
                /// For returned types, if the number of values returned in a tuple is greaterh than 5, it is returned as an array, instead.
                /// If a tuple contains both tensors and scalars, it is returned as an object[].
                /// </remarks>
                public object invoke(string name, params object[] objs)
                {
                    if (string.IsNullOrEmpty(name)) throw new ArgumentNullException("method name");

                    TensorOrScalar[] ptrArray = null;
                    sbyte typeCode = 0;

                    using (var ntosArray = new NativeTensorOrScalarIndexedArray()) {

                        var tRefsHandle = DetermineArgumentTypeRefs(objs, out var count, ntosArray);

                        var allocated = ntosArray.Count;

                        THSJIT_Module_invoke(handle, name, tRefsHandle, count, ntosArray.CreateArray, out typeCode, allocated);
                        torch.CheckForErrors();
                        ptrArray = ntosArray.ToToSArray(allocated);

                        return ScriptModule.ProcessReturnValue(name, ntosArray, ptrArray, typeCode);
                    }
                }

                internal static IntPtr DetermineArgumentTypeRefs(object[] objs, out int count, NativeTensorOrScalarIndexedArray allocator)
                {
                    count = objs.Length;
                    var tensorRefs = allocator.CreateArray(allocator.Count, objs.Length);

                    for (var idx = 0; idx < objs.Length; idx++) {
                        switch (objs[idx]) {
                        case Tensor t:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 0, 0, t.Handle);
                            break;
                        case Scalar s:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 1, 0, s.Handle);
                            break;
                        case float f:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 1, 0, ((Scalar)f).Handle);
                            break;
                        case double d:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 1, 0, ((Scalar)d).Handle);
                            break;
                        case bool i:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 2, 0, (IntPtr)(i ? 1L : 0L));
                            break;
                        case int i:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 3, 0, (IntPtr)i);
                            break;
                        case long l:
                            THSJIT_SetTensorOrScalar(tensorRefs, idx, 1, 0, ((Scalar)l).Handle);
                            // The MacOS version of Clang doesn't like the use of int64_t, so pass as a Scalar instance, instead.
                            //THSJIT_SetTensorOrScalar(tensorRefs, idx, 4, 0, (IntPtr)l);
                            break;
                        case Tensor[] tensors: {
                                THSJIT_SetTensorOrScalar(tensorRefs, idx, 5, tensors.Length, DetermineArgumentTypeRefs(tensors, out var _, allocator));
                            }
                            break;
                        default:
                            if (objs[idx] is null) {
                                THSJIT_SetTensorOrScalar(tensorRefs, idx, 8, 0, IntPtr.Zero);
                            } else {
                                throw new NotImplementedException($"Passing arguments of type {objs[idx].GetType().Name} to TorchScript.");
                            }
                            break;
                        }
                    }

                    return tensorRefs;
                }

                internal static object ProcessReturnValue(string name, NativeTensorOrScalarIndexedArray allocator, TensorOrScalar[] ptrArray, long typeCode)
                {
                    switch (typeCode) {
                    default:
                        // Nothing.
                        throw new NotImplementedException($"ScriptModule.{name}() returning something else than a tensor, a tuple of tensors, or list of tensors. TypeCode: {typeCode}");
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
                                switch (ptrArray[i].TypeCode) {
                                case 0:
                                    result[i] = new Tensor(ptrArray[i].Handle);
                                    break;
                                case 8:
                                    result[i] = null;
                                    break;
                                case 4:
                                    result[i] = new Scalar(ptrArray[i].Handle);
                                    break;
                                default:
                                    throw new NotImplementedException($"ScriptModule.{name}() returning something else than a tensor/scalar, a tuple of tensors/scalars, or list of tensors/scalars.");
                                }
                            }
                            return result;
                        }
                    case 8:
                        // The value 'null' of any reference type
                        return null;

                    case 10: {
                            switch (ptrArray.Length) {
                            case 1:
                                return ProcessReturnValue(name, allocator, ptrArray[0]);
                            case 2:
                                return (ProcessReturnValue(name, allocator, ptrArray[0]),
                                    ProcessReturnValue(name, allocator, ptrArray[1]));
                            case 3:
                                return (ProcessReturnValue(name, allocator, ptrArray[0]),
                                    ProcessReturnValue(name, allocator, ptrArray[1]),
                                    ProcessReturnValue(name, allocator, ptrArray[2]));
                            case 4:
                                return (ProcessReturnValue(name, allocator, ptrArray[0]),
                                    ProcessReturnValue(name, allocator, ptrArray[1]),
                                    ProcessReturnValue(name, allocator, ptrArray[2]),
                                    ProcessReturnValue(name, allocator, ptrArray[3]));
                            case 5:
                                return (ProcessReturnValue(name, allocator, ptrArray[0]),
                                    ProcessReturnValue(name, allocator, ptrArray[1]),
                                    ProcessReturnValue(name, allocator, ptrArray[2]),
                                    ProcessReturnValue(name, allocator, ptrArray[3]),
                                    ProcessReturnValue(name, allocator, ptrArray[4]));
                            default: {
                                    // Too long a tuple, return as a list, instead.
                                    var result = new Scalar[ptrArray.Length];
                                    for (var i = 0; i < ptrArray.Length; i++) {
                                        result[i] = new Scalar(ptrArray[i].Handle);
                                    }
                                    return result;
                                }
                            }
                        }
                    case 9: {
                            // List of anything
                            var result = new object[ptrArray.Length];
                            for (var i = 0; i < ptrArray.Length; i++) {
                                result[i] = ProcessReturnValue(name, allocator, ptrArray[i]);
                            }
                            return result;
                        }
                    }
                }

                internal static object ProcessReturnValue(string name, NativeTensorOrScalarIndexedArray allocator, TensorOrScalar value)
                {
                    switch (value.TypeCode) {
                    case 0:
                        return new Tensor(value.Handle);
                    case 4:
                        return new Scalar(value.Handle);
                    case 8:
                        return null;
                    default:
                        return ProcessReturnValue(name, allocator, allocator.ToToSArray((int)value.ArrayIndex), value.TypeCode);
                    }

                    throw new NotImplementedException($"ScriptModule.{name}() returning something else than a tensor/scalar, a tuple of tensors/scalars, or list of tensors/scalars.");
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
            /// Represents a module that accepts 'hook' to the module logic.
            /// </summary>
            public class HookableScriptModule<TPreHook, TPostHook> : ScriptModule
            {
                internal HookableScriptModule(IntPtr handle) : base(handle)
                {
                }

                public HookRemover register_forward_hook(TPostHook hook)
                {
                    var key = Guid.NewGuid().ToString();
                    post_hooks.Add(key, hook);
                    return new HookRemover(this, key);
                }

                public HookRemover register_forward_pre_hook(TPreHook hook)
                {
                    var key = Guid.NewGuid().ToString();
                    pre_hooks.Add(key, hook);
                    return new HookRemover(this, key);
                }

                private void remove(string key)
                {
                    if (pre_hooks.ContainsKey(key)) pre_hooks.Remove(key);
                    if (post_hooks.ContainsKey(key)) post_hooks.Remove(key);
                }

                protected Dictionary<string, TPreHook> pre_hooks = new Dictionary<string, TPreHook>();
                protected Dictionary<string, TPostHook> post_hooks = new Dictionary<string, TPostHook>();

                /// <summary>
                /// Used to remove a specific hook, following the PyTorch API design.
                /// </summary>
                /// <remarks>The name and namespace of this class is not the same as in PyTorch, but serves the same purpose.</remarks>
                public class HookRemover
                {
                    public HookRemover(HookableScriptModule<TPreHook, TPostHook> module, string key)
                    {
                        this.module = module;
                        this.key = key;
                    }

                    public void remove()
                    {
                        module.remove(key);
                    }

                    private HookableScriptModule<TPreHook, TPostHook> module;
                    private string key;
                }
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
                public TResult call(params Tensor[] input)
                {
                    // TODO: Call pre-hooks, if available.

                    var result = forward(input);

                    // TODO: Call post-hooks, if available.

                    return result;
                }

                public TResult forward(params Tensor[] tensor) => (TResult)base.forward(tensor);
            }

            /// <summary>
            /// A script module taking a single argument.
            /// </summary>
            /// <typeparam name="T">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            public class ScriptModule<T, TResult> : HookableScriptModule<Func<ScriptModule<T, TResult>, T, T>, Func<ScriptModule<T, TResult>, T, TResult, TResult>>, torch.nn.IModule<T, TResult>
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
                public TResult call(T input)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input);
                        if (modified is not null)
                            input = modified;
                    }

                    var result = forward(input);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input, result);
                        if (modified is not null)
                            result = modified;
                    }

                    return result;
                }

                public TResult forward(T tensor) => (TResult)base.forward(tensor);
            }

            /// <summary>
            /// A script module taking two arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            public class ScriptModule<T1, T2, TResult> : HookableScriptModule<Func<ScriptModule<T1, T2, TResult>, T1, T2, (T1, T2)?>, Func<ScriptModule<T1, T2, TResult>, T1, T2, TResult, TResult>>, torch.nn.IModule<T1, T2, TResult>
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
                public TResult call(T1 input1, T2 input2)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input1, input2);
                        if (modified.HasValue) {
                            input1 = modified.Value.Item1;
                            input2 = modified.Value.Item2;
                        }
                    }

                    var result = forward(input1, input2);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input1, input2, result);
                        if (modified is not null)
                            result = modified;
                    }

                    return result;
                }

                public TResult forward(T1 input1, T2 input2) => (TResult)base.forward(input1, input2);
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule load(string filename, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule(_load(filename, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule load(byte[] bytes, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule(_load(bytes, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule load(string filename, Device map_location)
            {
                return new ScriptModule(_load(filename, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule load(byte[] bytes, Device map_location)
            {
                return new ScriptModule(_load(bytes, map_location.type, map_location.index));
            }
            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule load(string filename, string map_location)
            {
                return load(filename, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule load(byte[] bytes, string map_location)
            {
                return load(bytes, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<TResult> load<TResult>(string filename, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule<TResult>(_load(filename, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<TResult> load<TResult>(byte[] bytes, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule<TResult>(_load(bytes, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<TResult> load<TResult>(string filename, Device map_location)
            {
                return new ScriptModule<TResult>(_load(filename, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<TResult> load<TResult>(byte[] bytes, Device map_location)
            {
                return new ScriptModule<TResult>(_load(bytes, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<TResult> load<TResult>(string filename, string map_location)
            {
                return load<TResult>(filename, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<TResult> load<TResult>(byte[] bytes, string map_location)
            {
                return load<TResult>(bytes, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, TResult> load<T1, TResult>(string filename, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule<T1, TResult>(_load(filename, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<T1, TResult> load<T1, TResult>(byte[] bytes, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule<T1, TResult>(_load(bytes, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, TResult> load<T1, TResult>(string filename, Device map_location)
            {
                return new ScriptModule<T1, TResult>(_load(filename, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<T1, TResult> load<T1, TResult>(byte[] bytes, Device map_location)
            {
                return new ScriptModule<T1, TResult>(_load(bytes, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, TResult> load<T1, TResult>(string filename, string map_location)
            {
                return load<T1, TResult>(filename, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<T1, TResult> load<T1, TResult>(byte[] bytes, string map_location)
            {
                return load<T1, TResult>(bytes, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(string filename, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule<T1, T2, TResult>(_load(filename, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="device_type">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="device_index">The optional device index.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(byte[] bytes, DeviceType device_type = DeviceType.CPU, long device_index = -1)
            {
                return new ScriptModule<T1, T2, TResult>(_load(bytes, device_type, device_index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(string filename, Device map_location)
            {
                return new ScriptModule<T1, T2, TResult>(_load(filename, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(byte[] bytes, Device map_location)
            {
                return new ScriptModule<T1, T2, TResult>(_load(bytes, map_location.type, map_location.index));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="filename">The file name of the module.</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            /// <exception cref="System.IO.FileNotFoundException">Raised if the file is not found.</exception>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(string filename, string map_location)
            {
                return load<T1, T2, TResult>(filename, new Device(map_location));
            }

            /// <summary>
            /// Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
            /// </summary>
            /// <typeparam name="T1">The first argument type.</typeparam>
            /// <typeparam name="T2">The second argument type.</typeparam>
            /// <typeparam name="TResult">The return type of the module.</typeparam>
            /// <param name="bytes">A byte array holding a serialized module</param>
            /// <param name="map_location">The device type where the script module should be loaded.</param>
            /// <returns>A ScriptModule instance, whether the script originated as a module or function.</returns>
            /// <remarks>
            /// All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from.If this fails (e.g.because the run time system doesn’t have certain devices), an exception is raised.
            /// </remarks>
            public static ScriptModule<T1, T2, TResult> load<T1, T2, TResult>(byte[] bytes, string map_location)
            {
                return load<T1, T2, TResult>(bytes, new Device(map_location));
            }

            private static IntPtr _load(string filename, DeviceType device_type, long device_index)
            {
                if (!System.IO.File.Exists(filename))
                    throw new System.IO.FileNotFoundException(filename);

                if (device_type != DeviceType.CUDA) device_index = -1;

                if (device_type == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                var result = THSJIT_load(filename, (long)device_type, device_index);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return result;
            }

            private unsafe static IntPtr _load(byte[] bytes, DeviceType device_type, long device_index)
            {
                if (device_type != DeviceType.CUDA) device_index = -1;

                if (device_type == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                fixed (byte* inputs = bytes) {
                    var result = THSJIT_load_byte_array((IntPtr)inputs, bytes.LongLength, (long)device_type, device_index);
                    if (result == IntPtr.Zero)
                        CheckForErrors();
                    return result;
                }
            }

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

            /// <summary>
            /// Save an offline version of a previously loaded script module.
            ///
            /// The saved module serializes all of the methods, submodules, parameters, and attributes of this module.
            /// It can be loaded into the C++ API using torch::jit::load(filename) or into the .NET API with torch.jit.load().
            /// </summary>
            /// <param name="module">The script module to save.</param>
            /// <param name="bytes">A byte array where the serialized module should be stored.</param>
            public unsafe static void save(ScriptModule module, byte[] bytes)
            {
                fixed (byte* inputs = bytes) {
                    THSJIT_save_byte_array(module.handle, (IntPtr)inputs, bytes.LongLength);
                    CheckForErrors();
                }
            }

        }
    }
}
