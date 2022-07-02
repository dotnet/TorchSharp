// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class jit
        {
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
                private static extern void THSJIT_Module_train(HType module);

                public override void train()
                {
                    THSJIT_Module_train(handle);
                    CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Module_eval(HType module);

                public override void eval()
                {
                    THSJIT_Module_eval(handle);
                    CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern bool THSJIT_Module_is_training(HType module);

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
                static extern void THSJIT_Module_to_dtype(HType module, sbyte dtype);

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
                /// <param name="deviceIndex">The optional device index.</param>
                /// <returns></returns>
                public override nn.Module to(DeviceType deviceType, int deviceIndex = -1)
                {
                    if (deviceType != DeviceType.CUDA) deviceIndex = -1;

                    if (deviceType == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                    if (deviceType != _deviceType || deviceIndex != _deviceIndex) {

                        InitializeDeviceType(deviceType);
                        THSJIT_Module_to_device(handle, (int)deviceType, deviceIndex);
                        CheckForErrors();

                        foreach (var (_, sm) in named_children()) sm.to(deviceType, deviceIndex);

                        foreach (var field in GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)) {

                            var fieldName = field.Name;
                            var value = field.GetValue(this);

                            switch (value) {
                            // This test must come before the Tensor test
                            case Modules.Parameter param when deviceType == param.device_type && deviceIndex == param.device_index:
                                continue;

                            case Modules.Parameter param: {
                                    var t = param.to(deviceType, deviceIndex);
                                    t.retain_grad();
                                    var p = new Modules.Parameter(t, param.requires_grad);
                                    field.SetValue(this, p);
                                    ConditionallyRegisterParameter(fieldName, p);
                                    break;
                                }

                            case Tensor tensor when (deviceType != tensor.device_type || deviceIndex != tensor.device_index): {
                                    var t = tensor.to(deviceType, deviceIndex);
                                    field.SetValue(this, t);
                                    ConditionallyRegisterBuffer(fieldName, t);
                                    break;
                                }
                            }
                        }

                        _deviceType = deviceType;
                        _deviceIndex = deviceIndex;
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
                public override nn.Module to(ScalarType dtype)
                {
                    THSJIT_Module_to_dtype(handle, (sbyte)dtype);
                    CheckForErrors();

                    foreach (var (_, sm) in named_children()) sm.to(dtype);
                    foreach (var field in GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)) {

                        var fieldName = field.Name;
                        var value = field.GetValue(this);

                        switch (value) {
                        // This test must come before the Tensor test
                        case Modules.Parameter param when dtype == param.dtype:
                            continue;

                        case Modules.Parameter param: {
                                var t = param.to(dtype);
                                t.retain_grad();
                                var p = new Modules.Parameter(t, param.requires_grad);
                                field.SetValue(this, p);
                                ConditionallyRegisterParameter(fieldName, p);
                                break;
                            }

                        case Tensor tensor when dtype == tensor.dtype:
                            continue;

                        case Tensor tensor: {
                                var t = tensor.to(dtype);
                                field.SetValue(this, t);
                                ConditionallyRegisterBuffer(fieldName, t);
                                break;
                            }
                        }
                    }

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
                private static extern IntPtr THSJIT_Module_forward(HType module, IntPtr tensors, int length);

                /// <summary>
                /// Invoke the 'forward' function of the script with one tensor as its argument
                /// </summary>
                /// <param name="tensor">The input tensor</param>
                /// <returns></returns>
                public unsafe override Tensor forward(Tensor tensor)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorRefs = parray.CreateArray(new[] { tensor.Handle });
                        var res = THSJIT_Module_forward(handle, tensorRefs, 1);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }

                /// <summary>
                /// Invoke the 'forward' function of the script with two tensors as its argument
                /// </summary>
                /// <param name="x">The first input tensor</param>
                /// <param name="y">The second input tensor</param>
                /// <returns></returns>
                public unsafe override Tensor forward(Tensor x, Tensor y)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorRefs = parray.CreateArray(new[] { x.Handle, y.Handle });
                        var res = THSJIT_Module_forward(handle, tensorRefs, 2);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }

                /// <summary>
                /// Invoke the 'forward' function of the script with three tensors as its argument
                /// </summary>
                /// <param name="x">The first input tensor</param>
                /// <param name="y">The second input tensor</param>
                /// <param name="z">The third input tensor</param>
                /// <returns></returns>
                public unsafe override Tensor forward(Tensor x, Tensor y, Tensor z)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorRefs = parray.CreateArray(new[] { x.Handle, y.Handle, z.Handle });
                        var res = THSJIT_Module_forward(handle, tensorRefs, 3);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }

                /// <summary>
                /// Invoke the 'forward' function of the script with four or more tensors as its argument
                /// </summary>
                /// <param name="x">The first input tensor</param>
                /// <param name="y">The second input tensor</param>
                /// <param name="z">The third input tensor</param>
                /// <param name="tensors">The remaining tensors.</param>
                /// <returns></returns>
                public unsafe Tensor forward(Tensor x, Tensor y, Tensor z, params Tensor[] tensors)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorRefs = parray.CreateArray(new[] { x, y, z }.Concat(tensors).Select(t => t.Handle).ToArray());
                        var res = THSJIT_Module_forward(handle, tensorRefs, parray.Array.Length);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSJIT_load(string filename);

            public static ScriptModule load(string filename)
            {
                if (!System.IO.File.Exists(filename))
                    throw new System.IO.FileNotFoundException(filename);

                var result = THSJIT_load(filename);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return new ScriptModule(result);
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSJIT_save(nn.Module.HType handle, string filename);

            public static void save(ScriptModule module, string filename)
            {
                THSJIT_save(module.handle, filename);
                CheckForErrors();
            }

        }
    }
}
