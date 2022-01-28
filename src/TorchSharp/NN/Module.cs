// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.Utils.LEB128Codec;
using System.Diagnostics;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Base class for all neural network modules.
            /// Your models should subclass this class.
            /// </summary>
            /// <remarks>
            /// Modules can also contain other Modules, allowing to nest them in a tree structure.
            /// You can assign the submodules as regular fields of the derived Module class. Submodules assigned
            /// to fields will be registered, and will have their parameters converted and moved when you call to(),
            /// and saved to disk when calling save().
            /// </remarks>
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

                    if (handle != IntPtr.Zero) {
                        foreach (var np in _named_parameters()) {
                            register_parameter(np.name, np.parameter);
                        }
                    }
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
                protected virtual void Dispose(bool disposing)
                {
                    if (disposing) {
                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                        boxedModule?.Dispose();
                    }
                }

                [DllImport("LibTorchSharp")]
                extern static void THSNN_Module_to_device(HType module, long deviceType, long deviceIndex);

                [DllImport("LibTorchSharp")]
                extern static void THSNN_Module_to_dtype(HType module, sbyte dtype);

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
                /// <param name="deviceIndex">The optional device index.</param>
                /// <returns></returns>
                public virtual Module to(DeviceType deviceType, int deviceIndex = -1)
                {
                    if (deviceType != DeviceType.CUDA) deviceIndex = -1;

                    if (deviceType != _deviceType || deviceIndex != _deviceIndex) {

                        torch.InitializeDeviceType(deviceType);
                        THSNN_Module_to_device(handle, (int)deviceType, deviceIndex);
                        torch.CheckForErrors();

                        foreach (var (_, sm) in named_children()) sm.to(deviceType, deviceIndex);

                        foreach (var field in this.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance)) {

                            var name = field.Name;
                            var value = field.GetValue(this);

                            Tensor tensor = value as Tensor;
                            Modules.Parameter param = value as Modules.Parameter;

                            if (param is not null) {  // This test must come before the Tensor test
                                if (deviceType != param.device_type || deviceIndex != param.device_index) {
                                    var p = new Modules.Parameter(param.to(deviceType, deviceIndex), param.requires_grad);
                                    field.SetValue(this, p);
                                    ConditionallyRegisterParameter(name, p);
                                }
                            } else if (tensor is not null) {
                                if (deviceType != tensor.device_type || deviceIndex != tensor.device_index) {
                                    var t = tensor.to(deviceType, deviceIndex);
                                    field.SetValue(this, t);
                                    ConditionallyRegisterBuffer(name, t);
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
                public Module to(torch.Device device)
                {
                    return to(device.type, device.index);
                }

                /// <summary>
                /// Convert the parameters and buffers.
                /// </summary>
                /// <returns></returns>
                public virtual Module to(ScalarType dtype)
                {
                    THSNN_Module_to_dtype(handle, (sbyte)dtype);
                    torch.CheckForErrors();

                    foreach (var (_, sm) in named_children()) sm.to(dtype);

                    foreach (var field in this.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance)) {

                        var name = field.Name;
                        var value = field.GetValue(this);

                        Tensor tensor = value as Tensor;
                        Modules.Parameter param = value as Modules.Parameter;

                        if (param is not null) {  // This test must come before the Tensor test
                            if (dtype != param.dtype) {
                                var p = new Modules.Parameter(param.to(dtype), param.requires_grad);
                                field.SetValue(this, p);
                                ConditionallyRegisterParameter(name, p);
                            }
                        } else if (tensor is not null) {
                            if (dtype != tensor.dtype) {
                                var t = tensor.to(dtype);
                                field.SetValue(this, t);
                                ConditionallyRegisterBuffer(name, t);
                            }
                        }
                    }

                    return this;
                }

                /// <summary>
                /// Moves and converts the parameters and buffers.
                /// </summary>
                /// <param name="other">The tensor serving as a template.</param>
                /// <returns></returns>
                public Module to(Tensor other)
                {
                    to(other.device_type, other.device_index);
                    return to(other.dtype);
                }

                /// <summary>
                /// Moves all model parameters and buffers to the CPU.
                /// </summary>
                public Module cpu()
                {
                    return to(DeviceType.CPU);
                }

                /// <summary>
                /// Applies a function recursively to every submodule as well as this.
                /// </summary>
                /// <param name="fn">Function to be applied to each submodule</param>
                /// <returns></returns>
                public virtual Module apply(Action<Module> fn)
                {
                    foreach (var m in GetModulesInternal()) m.apply(fn);
                    fn(this);
                    return this;
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
                    foreach (var (_, m) in named_children()) { m.Train(); }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_eval(HType module);

                public virtual void Eval()
                {
                    THSNN_Module_eval(handle);
                    torch.CheckForErrors();
                    foreach (var (_, m) in named_children()) { m.Eval(); }
                }

                [DllImport("LibTorchSharp")]
                private static extern bool THSNN_Module_is_training(HType module);

                public bool training {
                    get {
                        var res = THSNN_Module_is_training(handle);
                        torch.CheckForErrors();
                        return res;
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_zero_grad(HType module);

                public virtual void zero_grad()
                {
                    THSNN_Module_zero_grad(handle);
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_buffers(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                public virtual IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true)
                {
                    foreach (var nsm in _internal_buffers) {
                        yield return nsm;
                    }

                    if (recurse) {
                        foreach (var (name, sm) in _internal_submodules) {
                            foreach (var (n, p) in sm.named_buffers(true)) {
                                yield return ($"{name}.{n}", p);
                            }
                        }
                    }
                }

                public virtual IEnumerable<(string name, Module module)> named_children()
                {
                    return _internal_submodules;
                }

                public virtual IEnumerable<(string name, Module module)> named_modules()
                {
                    foreach (var nsm in _internal_submodules) {
                        yield return nsm;
                    }

                    foreach (var (name, sm) in _internal_submodules) {
                        foreach (var (n, p) in sm.named_modules()) {
                            yield return ($"{name}.{n}", p);
                        }
                    }
                }


                public Dictionary<string, Tensor> state_dict()
                {
                    var res = new Dictionary<string, Tensor>();
                    state_dict(res);
                    return res;
                }

                protected virtual void state_dict(Dictionary<string, Tensor> res)
                {
                    foreach (var p in named_parameters()) {
                        res.Add(p.Item1, p.Item2);
                    }
                    foreach (var p in named_buffers()) {
                        res.Add(p.Item1, p.Item2);
                    }
                    foreach (var p in _internal_submodules.Values) {
                        p.state_dict(res);
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_parameters(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                protected (string name, Modules.Parameter parameter)[] _named_parameters()
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
                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Modules.Parameter(x))).ToArray();

                }

                public virtual IEnumerable<(string name, Modules.Parameter parameter)> named_parameters(bool recurse = true)
                {
                    foreach (var nsm in _internal_params) {
                        yield return nsm;
                    }

                    if (recurse) {
                        foreach (var (name, sm) in _internal_submodules) {
                            foreach (var (n, p) in sm.named_parameters(true)) {
                                yield return ($"{name}.{n}", p);
                            }
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_parameters(HType module, AllocatePinnedArray allocator, bool recurse);

                protected virtual Modules.Parameter[] _parameters(bool recurse = true)
                {
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        AllocatePinnedArray allocator = pa.CreateArray;
                        THSNN_Module_get_parameters(handle, allocator, recurse);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                    }
                    return ptrArray.Select(x => new Modules.Parameter(x)).ToArray();
                }

                public virtual IEnumerable<Modules.Parameter> parameters(bool recurse = true)
                {
                    return named_parameters(recurse).Select(np => np.parameter);
                }

                public bool has_parameter(string name)
                {
                    if (_internal_params.TryGetValue(name, out var parameter)) {
                        return true;
                    }
                    foreach (var child in named_children().Where(nc => name.StartsWith(nc.name))) {
                        var prefix = child.name + ".";
                        var p = child.module.get_parameter(name.Remove(0, prefix.Length));
                        if (p is not null) return true;
                    }
                    return false;
                }

                public Modules.Parameter get_parameter(string name)
                {
                    if (_internal_params.TryGetValue(name, out var parameter)) {
                        return parameter;
                    }
                    foreach (var child in named_children().Where(nc => name.StartsWith(nc.name))) {
                        var prefix = child.name + ".";
                        var p = child.module.get_parameter(name.Remove(0, prefix.Length));
                        if (p is not null) return p;
                    }
                    return null;
                }

                public virtual void register_buffer(string name, Tensor tensor)
                {
                    if (tensor is null || tensor.handle == IntPtr.Zero) throw new ArgumentNullException("A null tensor cannot be registered as a buffer.");
                    if (_internal_buffers.ContainsKey(name)) {
                        throw new InvalidOperationException($"Tensor {name} is already registered.");
                    }
                    _internal_buffers.Add(name, tensor);
                }

                public virtual void register_parameter(string name, Tensor tensor, bool requires_grad = true)
                {
                    if (tensor is null || tensor.handle == IntPtr.Zero) throw new ArgumentNullException("A null tensor cannot be registered as a parameter.");
                    if (_internal_params.ContainsKey(name)) {
                        throw new InvalidOperationException($"Parameter {name} is already registered.");
                    }
                    _internal_params.Add(name, (tensor is Modules.Parameter)
                        ? (Modules.Parameter)tensor
                        : new Modules.Parameter(tensor, requires_grad));
                }

                public virtual void register_module(string name, Module submodule)
                {
                    if (submodule is null || submodule.handle.IsInvalid) {
                        if (_internal_submodules.ContainsKey(name)) {
                            _internal_submodules.Remove(name);
                        }
                    } else {
                        if (_internal_submodules.ContainsKey(name)) {
                            throw new InvalidOperationException($"Sub-module {name} is already registered.");
                        }

                        submodule.RegisterComponents();

                        _internal_submodules.Add(name, submodule);
                    }
                }

                protected void ConditionallyRegisterParameter(string name, Tensor value)
                {
                    if (value is null) {
                        if (_internal_params.ContainsKey(name)) {
                            _internal_params.Remove(name);
                        }
                    } else {
                        var p = value is Modules.Parameter
                            ? (Modules.Parameter)value
                            : new Modules.Parameter(value, requires_grad: true);

                        if (_internal_params.ContainsKey(name)) {
                            _internal_params[name] = p;
                        } else {
                            _internal_params.Add(name, p);
                        }
                    }
                }

                protected void ConditionallyRegisterBuffer(string name, Tensor value)
                {
                    if (value is null) {
                        if (_internal_buffers.ContainsKey(name)) {
                            _internal_buffers.Remove(name);
                        }
                    } else {
                        if (_internal_buffers.ContainsKey(name)) {
                            _internal_buffers[name] = value;
                        } else {
                            _internal_buffers.Add(name, value);
                        }
                    }
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

                public virtual Tensor forward(Tensor t) => throw new NotImplementedException("forward(t)");

                public virtual Tensor forward(Tensor x, Tensor y) => throw new NotImplementedException("forward(x,y)");

                /// <summary>
                /// Save the parameters and buffers of the module to a disk location.
                /// </summary>
                /// <param name="location">The file path.</param>
                /// <returns></returns>
                public Module save(string location)
                {
                    using (var stream = System.IO.File.OpenWrite(location))
                    using (var writer = new System.IO.BinaryWriter(stream))
                        save(writer);

                    return this;
                }

                public Module save(System.IO.BinaryWriter writer)
                {
                    var sd = state_dict();

                    // First, write how many entries.

                    SaveStateDictionary(writer, sd);

                    return this;
                }

                public static void SaveStateDictionary(System.IO.BinaryWriter writer, Dictionary<string, Tensor> sd)
                {
                    writer.Encode(sd.Count); // 4 bytes

                    foreach (var (k, v) in sd) {
                        writer.Write(k);
                        v.Save(writer);
                    }
                }

                /// <summary>
                /// Load the parameters and buffers 
                /// </summary>
                /// <param name="location">The file path.</param>
                /// <param name="strict">
                /// If true, will only load a module if it exactly corresponds to the current module's state.
                /// If false, will load the parameters and buffers that it finds in the saved file,
                /// leaving everything else alone.
                /// </param>
                /// <returns></returns>
                public Module load(string location, bool strict = true)
                {
                    var dt = _deviceType;
                    var di = _deviceIndex;

                    cpu();

                    try {
                        using (var stream = System.IO.File.OpenRead(location))
                        using (var reader = new System.IO.BinaryReader(stream))
                            load(reader, strict);
                    } finally {
                        to(dt, di);
                    }

                    return this;
                }

                public virtual Module load(System.IO.BinaryReader reader, bool strict = true)
                {
                    var sd = state_dict();

                    // First, figure out how many entries.
                    var streamEntries = reader.Decode();

                    if (streamEntries != sd.Count && strict)
                        throw new ArgumentException($"Mismatched state_dict sizes: expected {sd.Count}, but found {streamEntries} entries.");

                    for (int i = 0; i < streamEntries; ++i) {
                        var key = reader.ReadString();
                        var found = sd.ContainsKey(key);
                        if (!found && strict) {
                            throw new ArgumentException($"Mismatched module state names: the target modules does not have a submodule or buffer named '{key}'");
                        }

                        if (found) {
                            sd[key].Load(reader);
                        }
                    }

                    return this;
                }


                /// <summary>
                /// Create a module and load its weights from disk.
                /// </summary>
                /// <typeparam name="T"></typeparam>
                /// <param name="path"></param>
                /// <returns></returns>
                public static Module Create<T>(string path) where T : nn.Module, new()
                {
                    var model = new T();
                    return model.load(path);
                }

                private delegate IntPtr ForwardFunctionC(IntPtr tensor);

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_custom_module([MarshalAs(UnmanagedType.LPStr)] string name,
                    ForwardFunctionC forward, out IntPtr pBoxedModule);

                /// <summary>
                /// Constructor for custom modules, i.e. those defined outside of TorchSharp.
                /// </summary>
                /// <param name="name">The name of the module. Useful for debugging purposes, mostly.</param>
                protected Module(string name) : this(IntPtr.Zero, IntPtr.Zero)
                {
                    this.name = name;

                    ForwardFunctionC forwardNative = t => {
                        var input = new Tensor(t);
                        var output = forward(input);

                        // handles must live on - we don't own them, but
                        // the managed objects should go away.

                        input.DecoupleFromNativeHandle();

                        return output.DecoupleFromNativeHandle();
                    };

                    var res = THSNN_custom_module(name, forwardNative, out var boxedHandle);
                    torch.CheckForErrors();
                    this.handle = new HType(res, true);
                    this.forwardNative = forwardNative;
                    this.boxedModule = new BoxedModule(boxedHandle);

                    // In this case, the parameter registration was not done yet.

                    foreach (var np in _named_parameters()) {
                        register_parameter(np.name, np.parameter);
                    }
                }

                protected virtual void RegisterComponents()
                {
                    if (_registered) return;

                    foreach (var field in this.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance)) {

                        var name = field.Name;
                        var value = field.GetValue(this);

                        var module = value as Module;
                        Tensor tensor = value as Tensor;
                        Modules.Parameter param = value as Modules.Parameter;

                        if (module != null) {
                            register_module(name, module);
                        } else if (param is not null) {  // This test must come before the Tensor test
                            register_parameter(name, tensor);
                        } else if (tensor is not null) {
                            register_buffer(name, tensor);
                        }
                    }
                    _registered = true;
                }

                private bool _registered = false;

                protected Utils.OrderedDict<string, Module> _internal_submodules = new Utils.OrderedDict<string, Module>();
                protected Utils.OrderedDict<string, Tensor> _internal_buffers = new Utils.OrderedDict<string, Tensor>();
                protected Utils.OrderedDict<string, Modules.Parameter> _internal_params = new Utils.OrderedDict<string, Modules.Parameter>();

                /// Keeps the callback delegate alive
                private ForwardFunctionC forwardNative;
                protected string name;
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
        }
    }
}
