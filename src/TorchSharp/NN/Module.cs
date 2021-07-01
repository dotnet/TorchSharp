// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

using static TorchSharp.Utils.LEB128Codec;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public class Module : IDisposable
            {
                /// <summary>
                ///    Class wrapping PyTorch's module object reference.
                /// </summary>
                internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
                    {
                        SetHandle(preexistingHandle);
                    }

                    public override bool IsInvalid => handle == IntPtr.Zero;

                    // This is just for marshalling
                    internal HType() : base(IntPtr.Zero, true)
                    {
                    }

                    [DllImport("LibTorchSharp")]
                    private static extern void THSNN_Module_dispose(HType handle);

                    protected override bool ReleaseHandle()
                    {
                        if (!IsInvalid) THSNN_Module_dispose(this);
                        SetHandle(IntPtr.Zero);
                        return true;
                    }

                    protected override void Dispose(bool disposing)
                    {
                        if (disposing) {
                            ReleaseHandle();
                        }
                    }
                }

                internal HType handle;

                /// Stores the AnyModule corresponding to this module.
                internal BoxedModule boxedModule;

                internal BoxedModule BoxedModule {
                    get {
                        if (boxedModule == null)
                            throw new InvalidOperationException("A Sequential or Loaded module may not be added to a Sequential");
                        return boxedModule;
                    }
                }

                internal Module(IntPtr handle, IntPtr? boxedHandle, bool ownsHandle = true)
                {
                    this.handle = new HType(handle, ownsHandle);
                    this.boxedModule = boxedHandle.HasValue ? new BoxedModule(boxedHandle.Value) : null;
                    //Debug.Assert (!this.handle.IsInvalid);
                    //Debug.Assert (!this.boxedHandle.IsInvalid);
                }

                ~Module()
                {
                    Dispose(false);
                }

                /// <summary>
                ///   Releases the storage.
                /// </summary>
                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                /// <summary>
                ///   Implements the .NET Dispose pattern.
                /// </summary>
                protected void Dispose(bool disposing)
                {
                    if (disposing) {
                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                        boxedModule?.Dispose();
                    }
                }

                [DllImport("LibTorchSharp")]
                extern static void THSNN_Module_to_device(HType module, long deviceType, long deviceIndex);

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
                /// <param name="deviceIndex">The optional device index.</param>
                /// <returns></returns>
                public Module to(DeviceType deviceType, int deviceIndex = -1)
                {
                    torch.InitializeDeviceType(deviceType);
                    THSNN_Module_to_device(handle, (int)deviceType, deviceIndex);
                    torch.CheckForErrors();
                    return this;
                }

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="device">A string denoting the target device.</param>
                /// <returns></returns>
                public Module to(string device)
                {
                    // Rely on the Device constructor to parse the string.
                    return to(new Device(device));
                }

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="device">The target device</param>
                /// <returns></returns>
                public Module to(Device device)
                {
                    return to(device.Type, device.Index);
                }

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="other">The tensor serving as a template.</param>
                /// <returns></returns>
                public Module to(TorchTensor other)
                {
                    return to(other.device_type, other.device_index);
                }

                /// <summary>
                /// Moves all model parameters and buffers to the CPU.
                /// </summary>
                public Module cpu()
                {
                    return to(DeviceType.CPU);
                }

                /// <summary>
                /// Moves all model parameters and buffers to a GPU.
                ///
                /// This also makes associated parameters and buffers different objects.So it should be called before constructing optimizer if the module will live on GPU while being optimized.
                /// </summary>
                /// <param name="deviceIndex">If specified, all parameters will be copied to that device</param>
                public Module cuda(int deviceIndex = -1)
                {
                    return to(DeviceType.CUDA, deviceIndex);
                }

                [DllImport("LibTorchSharp")]
                extern static IntPtr THSNN_Module_load([MarshalAs(UnmanagedType.LPStr)] string location);

                public static Module Load(String location)
                {
                    var handle = THSNN_Module_load(location);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Module(handle, IntPtr.Zero);
                }

                [DllImport("LibTorchSharp")]
                extern static void THSNN_Module_save(HType handle, [MarshalAs(UnmanagedType.LPStr)] string location);

                public virtual void Save(String modelPath)
                {
                    THSNN_Module_save(handle, modelPath);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_train(HType module);

                public virtual void Train()
                {
                    THSNN_Module_train(handle);
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_eval(HType module);

                public virtual void Eval()
                {
                    THSNN_Module_eval(handle);
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern bool THSNN_Module_is_training(HType module);

                public bool IsTraining()
                {
                    var res = THSNN_Module_is_training(handle);
                    torch.CheckForErrors();
                    return res;
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_zero_grad(HType module);

                public virtual void ZeroGrad()
                {
                    THSNN_Module_zero_grad(handle);
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_parameters(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                public virtual (string name, TorchTensor parameter)[] NamedParameters()
                {
                    IntPtr[] ptrArray;
                    IntPtr[] strArray;

                    using (var pa = new PinnedArray<IntPtr>())
                    using (var sa = new PinnedArray<IntPtr>()) {
                        THSNN_Module_get_named_parameters(handle, pa.CreateArray, sa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                        strArray = sa.Array;
                    }
                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new TorchTensor(x))).ToArray();

                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_buffers(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                public virtual (string name, TorchTensor parameter)[] NamedBuffers()
                {
                    IntPtr[] ptrArray;
                    IntPtr[] strArray;

                    using (var pa = new PinnedArray<IntPtr>())
                    using (var sa = new PinnedArray<IntPtr>()) {
                        THSNN_Module_get_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                        strArray = sa.Array;
                    }
                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new TorchTensor(x))).ToArray();

                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_children(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);
                public virtual (string name, Module parameter)[] NamedChildren()
                {
                    IntPtr[] ptrArray;
                    IntPtr[] strArray;

                    using (var pa = new PinnedArray<IntPtr>())
                    using (var sa = new PinnedArray<IntPtr>()) {
                        THSNN_Module_get_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                        strArray = sa.Array;
                    }
                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Module(x, IntPtr.Zero))).ToArray();

                }


                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_modules(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);
                public virtual (string name, Module parameter)[] NamedModules()
                {
                    IntPtr[] ptrArray;
                    IntPtr[] strArray;

                    using (var pa = new PinnedArray<IntPtr>())
                    using (var sa = new PinnedArray<IntPtr>()) {
                        THSNN_Module_get_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                        strArray = sa.Array;
                    }
                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Module(x, IntPtr.Zero))).ToArray();

                }


                public Dictionary<string, TorchTensor> state_dict()
                {
                    var ptrArray = new List<IntPtr>();
                    var strArray = new List<IntPtr>();

                    using (var pa = new PinnedArray<IntPtr>())
                    using (var sa = new PinnedArray<IntPtr>()) {
                        THSNN_Module_get_named_parameters(handle, pa.CreateArray, sa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray.AddRange(pa.Array);
                        strArray.AddRange(sa.Array);

                        THSNN_Module_get_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray.AddRange(pa.Array);
                        strArray.AddRange(sa.Array);
                    }

                    var result = new Dictionary<string, TorchTensor>();
                    for (var i = 0; i < ptrArray.Count; ++i) {
                        result[Marshal.PtrToStringAnsi(strArray[i])] = new TorchTensor(ptrArray[i]);
                    }
                    return result;
                }


                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_parameters(HType module, AllocatePinnedArray allocator);

                public virtual TorchTensor[] parameters()
                {
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        AllocatePinnedArray allocator = pa.CreateArray;
                        THSNN_Module_get_parameters(handle, allocator);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                    }
                    return ptrArray.Select(x => new TorchTensor(x)).ToArray();
                }

                [DllImport("LibTorchSharp")]
                static extern bool THSNN_Module_has_parameter(HType module, [MarshalAs(UnmanagedType.LPStr)] string name);

                public bool HasParameter(string name)
                {
                    var res = THSNN_Module_has_parameter(handle, name);
                    torch.CheckForErrors();
                    return res;
                }

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_Module_get_parameter(HType module, [MarshalAs(UnmanagedType.LPStr)] string name);

                public TorchTensor GetParameter(string name)
                {
                    var parameter = THSNN_Module_get_parameter(handle, name);
                    torch.CheckForErrors();

                    if (parameter == IntPtr.Zero) {
                        throw new ArgumentNullException("Linear module without bias term.");
                    }

                    return new TorchTensor(parameter);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_register_buffer(HType module, string name, IntPtr tensor);

                public virtual void RegisterBuffer(string name, TorchTensor tensor)
                {
                    THSNN_Module_register_buffer(handle, name, tensor.handle);
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_register_module(HType module, string name, HType submodule);

                public virtual void RegisterModule(string name, Module submodule)
                {
                    THSNN_Module_register_module(handle, name, submodule.handle);
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern long THSNN_Module_children_size(HType module);

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_Module_child(HType module, int index);

                /// Get the sub-modules of this module. The Module objects won't have the correct .NET types
                /// so this is not made public.
                internal virtual Module[] GetModulesInternal()
                {
                    var numModules = THSNN_Module_children_size(handle);
                    torch.CheckForErrors();
                    Module[] result = new Module[numModules];

                    for (int i = 0; i < numModules; i++) {
                        var childHandle = THSNN_Module_child(handle, i);
                        torch.CheckForErrors();
                        result[i] = new Module(childHandle, null, ownsHandle: false);
                    }

                    return result;
                }

                [DllImport("LibTorchSharp")]
                [return: MarshalAs(UnmanagedType.LPStr)]
                private static extern string THSNN_Module_name(HType module);

                public virtual string GetName()
                {
                    var res = THSNN_Module_name(handle);
                    torch.CheckForErrors();
                    return res;
                }

                public virtual TorchTensor forward(TorchTensor t) => throw new NotImplementedException("forward");

                public Module save(string location)
                {
                    cpu();

                    using (var stream = System.IO.File.OpenWrite(location))
                    using (var writer = new System.IO.BinaryWriter(stream))
                        save(writer);
                    return this;
                }

                public Module save(System.IO.BinaryWriter writer)
                {
                    var sd = state_dict();

                    // First, write how many entries.

                    writer.Encode(sd.Count); // 4 bytes

                    foreach (var (k, v) in sd) {
                        writer.Write(k);
                        v.Save(writer);
                    }

                    return this;
                }

                public Module load(string location)
                {
                    using (var stream = System.IO.File.OpenRead(location))
                    using (var reader = new System.IO.BinaryReader(stream))
                        load(reader);
                    return this;
                }

                public Module load(System.IO.BinaryReader reader)
                {
                    var sd = state_dict();

                    // First, figure out how many entries.
                    var streamEntries = reader.Decode();

                    if (streamEntries != sd.Count)
                        throw new ArgumentException("Mismatched number of state entries while loading module.");

                    for (int i = 0; i < streamEntries; ++i) {
                        var key = reader.ReadString();
                        if (!sd.ContainsKey(key)) {
                            throw new ArgumentException("Mismatched module state names.");
                        }

                        sd[key].Load(reader);
                    }

                    return this;
                }
            }

            internal class BoxedModule : IDisposable
            {
                internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
                    {
                        SetHandle(preexistingHandle);
                    }

                    public override bool IsInvalid => handle == IntPtr.Zero;

                    // This is just for marshalling
                    internal HType() : base(IntPtr.Zero, true)
                    {
                    }

                    [DllImport("LibTorchSharp")]
                    private static extern void THSNN_AnyModule_dispose(HType handle);

                    protected override bool ReleaseHandle()
                    {
                        if (!IsInvalid) THSNN_AnyModule_dispose(this);
                        handle = IntPtr.Zero;
                        return true;
                    }

                    protected override void Dispose(bool disposing)
                    {
                        if (disposing) {
                            ReleaseHandle();
                        }
                    }
                }

                internal HType handle;

                internal BoxedModule(IntPtr handle)
                {
                    this.handle = new HType(handle, true);
                }

                ~BoxedModule()
                {
                    Dispose(false);
                }

                /// <summary>
                ///   Releases the storage.
                /// </summary>
                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                /// <summary>
                ///   Implements the .NET Dispose pattern.
                /// </summary>
                protected void Dispose(bool disposing)
                {
                    if (disposing) {
                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                    }
                }
            }

#nullable enable
            public abstract class CustomModule : Module
            {
                private delegate IntPtr ForwardFunctionC(IntPtr tensor);

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_custom_module([MarshalAs(UnmanagedType.LPStr)] string name,
                    IntPtr names, IntPtr parameters, IntPtr require_grad,
                    int length, ForwardFunctionC forward, out IntPtr pBoxedModule);

                protected CustomModule(string name, params parameter.Parameter[] parameters) : base(IntPtr.Zero, IntPtr.Zero)
                {
                    var names = parameters.Select(p => Marshal.StringToHGlobalAnsi(p.Name)).ToArray();
                    var @params = parameters.Select(p => p.Tensor.Handle).ToArray();
                    var withGrads = parameters.Select(p => p.WithGrad).ToArray();

                    var namesPinned = new PinnedArray<IntPtr>();
                    var paramsPinned = new PinnedArray<IntPtr>();
                    var wGradPinned = new PinnedArray<bool>();

                    var nparray = namesPinned.CreateArray(names);
                    var pparray = paramsPinned.CreateArray(@params);
                    var gparray = wGradPinned.CreateArray(withGrads);

                    ForwardFunctionC forwardNative = t => (forward(new TorchTensor(t)).Handle);
                    var res = THSNN_custom_module(name, nparray, pparray, gparray, names.Length, forwardNative, out var boxedHandle);
                    torch.CheckForErrors();
                    this.handle = new HType(res, true);
                    this.forwardNative = forwardNative;
                    this.boxedModule = new BoxedModule(boxedHandle);
                }

                protected void RegisterComponents()
                {
                    foreach (var field in this.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance)) {
                        var value = field.GetValue(this);

                        var module = field.GetValue(this) as Module;
                        TorchTensor? tensor = value as TorchTensor;

                        if (module != null) {
                            RegisterModule(field.Name, module);
                        } else if (!(tensor is null)) {
                            RegisterBuffer(field.Name, tensor);
                        }
                    }
                }

                /// Keeps the callback delegate alive
                private ForwardFunctionC forwardNative;
            }
        }
    }
}
