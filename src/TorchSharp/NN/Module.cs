// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp.Modules;
using static TorchSharp.Utils.LEB128Codec;

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
                /// Class wrapping PyTorch's module object reference.
                /// </summary>
                internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle)
                        : base(IntPtr.Zero, ownsHandle)
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
                    boxedModule = boxedHandle.HasValue ? new BoxedModule(boxedHandle.Value) : null;

                    if (handle == IntPtr.Zero) return;
                    foreach (var (parameterName, parameter) in _named_parameters()) {
                        register_parameter(parameterName, parameter);
                    }
                    foreach (var (bufferName, buffer) in _named_buffers()) {
                        register_buffer(bufferName, buffer);
                    }
                }

                ~Module() => Dispose(false);

                /// <summary>
                /// Releases the storage.
                /// </summary>
                public void Dispose()
                {
                    GC.SuppressFinalize(this);
                    Dispose(true);
                }

                /// <summary>
                /// Implements the .NET Dispose pattern.
                /// </summary>
                protected virtual void Dispose(bool disposing)
                {
                    if (disposing && !handle.IsInvalid) {

                        foreach (var (_, p) in named_buffers(false)) {
                            p.Dispose();
                        }
                        foreach (var (_, b) in named_parameters(false)) {
                            b.Dispose();
                        }

                        foreach (var (_,m) in named_modules()) {
                            m.Dispose();
                        }

                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                        boxedModule?.Dispose();
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern void THSNN_Module_to_device(HType module, long deviceType, long deviceIndex);

                [DllImport("LibTorchSharp")]
                static extern void THSNN_Module_to_dtype(HType module, sbyte dtype);

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
                /// <param name="deviceIndex">The optional device index.</param>
                /// <returns></returns>
                public virtual Module to(DeviceType deviceType, int deviceIndex = -1)
                {
                    if (deviceType != DeviceType.CUDA) deviceIndex = -1;

                    if (deviceType == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                    if (deviceType != _deviceType || deviceIndex != _deviceIndex) {

                        InitializeDeviceType(deviceType);
                        THSNN_Module_to_device(handle, (int)deviceType, deviceIndex);
                        CheckForErrors();

                        foreach (var (_, sm) in named_children()) sm.to(deviceType, deviceIndex);

                        foreach (var field in GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)) {

                            var fieldName = field.Name;
                            var value = field.GetValue(this);

                            switch (value) {
                            // This test must come before the Tensor test
                            case Parameter param when deviceType == param.device_type && deviceIndex == param.device_index:
                                continue;

                            case Parameter param: {
                                    var t = param.to(deviceType, deviceIndex);
                                    t.retain_grad();
                                    var p = new Parameter(t, param.requires_grad);
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
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="device">A string denoting the target device.</param>
                /// <returns></returns>
                /// <remarks>Relies on the Device constructor to parse the string.</remarks>
                public Module to(string device) => to(new Device(device));

                /// <summary>
                /// Moves the parameters and buffers.
                /// </summary>
                /// <param name="device">The target device</param>
                /// <returns></returns>
                public Module to(Device device) => to(device.type, device.index);

                /// <summary>
                /// Convert the parameters and buffers.
                /// </summary>
                /// <returns></returns>
                public virtual Module to(ScalarType dtype)
                {
                    THSNN_Module_to_dtype(handle, (sbyte)dtype);
                    CheckForErrors();

                    foreach (var (_, sm) in named_children()) sm.to(dtype);
                    foreach (var field in GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)) {

                        var fieldName = field.Name;
                        var value = field.GetValue(this);

                        switch (value) {
                        // This test must come before the Tensor test
                        case Parameter param when dtype == param.dtype:
                            continue;

                        case Parameter param: {
                                var t = param.to(dtype);
                                t.retain_grad();
                                var p = new Parameter(t, param.requires_grad);
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
                public Module cpu() => to(DeviceType.CPU);

                /// <summary>
                /// Applies a function recursively to every submodule as well as this.
                /// </summary>
                /// <param name="fn">Function to be applied to each submodule</param>
                /// <returns></returns>
                public virtual Module apply(Action<Module> fn)
                {
                    foreach (var (_, m) in _internal_submodules) m.apply(fn);
                    fn(this);
                    return this;
                }

                /// <summary>
                /// Moves all model parameters and buffers to a GPU.
                ///
                /// This also makes associated parameters and buffers different objects.So it should be called before constructing optimizer if the module will live on GPU while being optimized.
                /// </summary>
                /// <param name="deviceIndex">If specified, all parameters will be copied to that device</param>
                public Module cuda(int deviceIndex = -1) => to(DeviceType.CUDA, deviceIndex);

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSNN_Module_load([MarshalAs(UnmanagedType.LPStr)] string location);

                public static Module Load(string location)
                {
                    var handle = THSNN_Module_load(location);
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    return new Module(handle, IntPtr.Zero);
                }

                [DllImport("LibTorchSharp")]
                static extern void THSNN_Module_save(
                    HType handle,
                    [MarshalAs(UnmanagedType.LPStr)] string location);

                public virtual void Save(string modelPath)
                    => THSNN_Module_save(handle, modelPath);

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_train(HType module);

                public virtual void train()
                {
                    THSNN_Module_train(handle);
                    CheckForErrors();
                    foreach (var (_, m) in named_children()) { m.train(); }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_eval(HType module);

                public virtual void eval()
                {
                    THSNN_Module_eval(handle);
                    CheckForErrors();
                    foreach (var (_, m) in named_children()) { m.eval(); }
                }

                [DllImport("LibTorchSharp")]
                private static extern bool THSNN_Module_is_training(HType module);

                public bool training {
                    get {
                        var res = THSNN_Module_is_training(handle);
                        CheckForErrors();
                        return res;
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_zero_grad(HType module);

                public virtual void zero_grad()
                {
                    THSNN_Module_zero_grad(handle);
                    CheckForErrors();
                }

                /// <summary>
                /// Returns an enumerable of module buffers, yielding both the name of the buffer as well as the buffer itself.
                /// </summary>
                /// <param name="recurse">If true, then yields buffers of this module and all submodules. Otherwise, yields only buffers that are direct members of this module.</param>
                /// <returns>(string, torch.Tensor) – Tuple containing the name and buffer</returns>
                public virtual IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true)
                {
                    foreach (var nsm in _internal_buffers) {
                        yield return nsm;
                    }

                    if (!recurse) yield break;

                    foreach (var (submoduleName, submodule) in _internal_submodules) {
                        foreach (var (parameterName, parameter) in submodule.named_buffers(true)) {
                            yield return ($"{submoduleName}.{parameterName}", parameter);
                        }
                    }
                }

                /// <summary>
                /// Returns an enumerable of immediate children modules, yielding both the name of the module as well as the module itself.
                /// </summary>
                /// <returns>(string, Module) – Tuple containing a name and child module</returns>
                public virtual IEnumerable<(string name, Module module)> named_children() => _internal_submodules;

                /// <summary>
                /// Returns an enumerable of all modules in the network, yielding both the name of the module as well as the module itself.
                /// </summary>
                /// <returns>(string, Module) – Tuple of name and module</returns>
                public virtual IEnumerable<(string name, Module module)> named_modules()
                {
                    foreach (var nsm in _internal_submodules) {
                        yield return nsm;
                    }

                    foreach (var (submoduleName, sm) in _internal_submodules) {
                        foreach (var (n, p) in sm.named_modules()) {
                            yield return ($"{submoduleName}.{n}", p);
                        }
                    }
                }

                /// <summary>
                /// Returns a dictionary containing a whole state of the module.
                ///
                /// Both parameters and persistent buffers(e.g.running averages) are included.Keys are corresponding parameter and buffer names.
                /// Parameters and buffers set to null are not included.
                /// </summary>
                /// <param name="destination">An optional dictionary where the state should be accumulated.</param>
                /// <param name="prefix">A prefix string to use when entering the name of entries into the dictionary.</param>
                /// <returns></returns>
                public virtual Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor> destination = null, string prefix = null)
                {
                    destination ??= new Dictionary<string, Tensor>();

                    foreach (var p in named_parameters()) {
                        var key = string.IsNullOrEmpty(prefix) ? $"{p.name}" : $"{prefix}.{p.name}";
                        destination.TryAdd(key, p.Item2);
                    }

                    foreach (var p in named_buffers()) {
                        var key = string.IsNullOrEmpty(prefix) ? $"{p.name}" : $"{prefix}.{p.name}";
                        destination.TryAdd(key, p.Item2);
                    }

                    foreach (var (n, p) in _internal_submodules) {
                        var key = string.IsNullOrEmpty(prefix) ? $"{n}" : $"{prefix}.{n}";
                        p.state_dict(destination, key);
                    }

                    return destination;
                }

                /// <summary>
                /// Copies parameters and buffers from state_dict into this module and its descendants.
                ///
                /// If strict is True, then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function.
                /// </summary>
                /// <param name="source">A dict containing parameters and persistent buffers.</param>
                /// <param name="strict">Whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function.</param>
                /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
                /// <returns></returns>
                public virtual (IList<string> missing_keys, IList<string> unexpected_keyes) load_state_dict(Dictionary<string, Tensor> source, bool strict = true, IList<string> skip = null)
                {
                    List<string> missing = new List<string>();
                    List<string> unexpected = new List<string>();
                    skip ??= Array.Empty<string>();

                    var destination = state_dict();

                    foreach (var key in source.Keys) {
                        if (skip.Contains(key)) continue;
                        if (!destination.ContainsKey(key)) {
                            unexpected.Add(key);
                        }
                    }

                    foreach (var key in destination.Keys) {
                        if (skip.Contains(key)) continue;
                        if (!source.ContainsKey(key)) {
                            missing.Add(key);
                        }
                    }

                    if (strict && (missing.Count > 0 || unexpected.Count > 0))
                        throw new InvalidOperationException("The loaded state_dict is not identical to the target dictionary.");

                    foreach (var key in source.Keys) {
                        if (skip.Contains(key)) continue;
                        if (destination.ContainsKey(key)) {
                            destination[key].bytes = source[key].bytes;
                        }
                    }

                    return (missing, unexpected);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_parameters(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                protected (string name, Parameter parameter)[] _named_parameters()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSNN_Module_get_named_parameters(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Parameter(x))).ToArray();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_named_buffers(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

                protected (string name, Tensor buffer)[] _named_buffers()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSNN_Module_get_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new Tensor(x))).ToArray();
                }

                /// <summary>
                /// Returns an enumerable of module parameters, yielding both the name of the parameter as well as the parameter itself.
                /// </summary>
                /// <param name="recurse">If true, then yields parameters of this module and all submodules. Otherwise, yields only parameters that are direct members of this module.</param>
                /// <returns>(string, Parameter) – Tuple containing the name and parameter</returns>
                public virtual IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
                {
                    var seen = new HashSet<IntPtr>();

                    foreach (var nsm in _internal_params) {
                        if (seen.Contains(nsm.Item2.Handle)) continue;
                        seen.Add(nsm.Item2.Handle);
                        yield return nsm;
                    }

                    if (!recurse) yield break;
                    foreach (var (submoduleName, subModule) in _internal_submodules) {
                        foreach (var (parameterName, parameter) in subModule.named_parameters(true)) {
                            if (seen.Contains(parameter.handle)) continue;
                            seen.Add(parameter.handle);
                            yield return ($"{submoduleName}.{parameterName}", parameter);
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Module_get_parameters(HType module, AllocatePinnedArray allocator, bool recurse);

                protected virtual Parameter[] _parameters(bool recurse = true)
                {
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        AllocatePinnedArray allocator = pa.CreateArray;
                        THSNN_Module_get_parameters(handle, allocator, recurse);
                        CheckForErrors();
                        ptrArray = pa.Array;
                    }
                    return ptrArray.Select(x => new Parameter(x)).ToArray();
                }

                /// <summary>
                /// Returns an enumerable of module parameters.
                /// </summary>
                /// <param name="recurse">If true, then yields parameters of this module and all submodules. Otherwise, yields only parameters that are direct members of this module.</param>
                /// <returns></returns>
                public virtual IEnumerable<Parameter> parameters(bool recurse = true)
                    => named_parameters(recurse).Select(np => np.parameter);

                public virtual bool has_buffer(string target)
                {
                    if (_internal_buffers.TryGetValue(target, out var buffer)) {
                        return true;
                    }

                    var splits = target.Split('.');
                    if (splits.Length <= 1) return false;
                    foreach (var child in named_children().Where(nc => nc.name == splits[0])) {
                        if (child.module.has_buffer(target.Remove(0, splits[0].Length + 1)))
                            return true;
                    }
                    return false;
                }

                public virtual bool has_parameter(string target)
                {
                    if (_internal_params.TryGetValue(target, out var parameter)) {
                        return true;
                    }

                    var splits = target.Split('.');
                    if (splits.Length <= 1) return false;
                    foreach (var child in named_children().Where(nc => nc.name == splits[0])) {
                        if (child.module.has_parameter(target.Remove(0, splits[0].Length + 1)))
                            return true;
                    }
                    return false;
                }

                /// <summary>
                /// Returns the buffer given by target if it exists, otherwise throws an error.
                /// </summary>
                /// <param name="target">The fully-qualified string name of the buffer to look for.</param>
                /// <returns>The tensor referenced by target</returns>
                public virtual Tensor get_buffer(string target)
                {
                    if (target is null) throw new ArgumentNullException("target");
                    if (_internal_buffers.TryGetValue(target, out var buffer)) {
                        return buffer;
                    }

                    var splits = target.Split('.');
                    if (splits.Length <= 1) return null;
                    foreach (var child in named_children().Where(nc => nc.name == splits[0])) {
                        var p = child.module.get_buffer(target.Remove(0, splits[0].Length + 1));
                        if (p is not null)
                            return p;
                    }
                    return null;
                }

                /// <summary>
                /// Returns the parameter given by target if it exists, otherwise throws an error.
                /// </summary>
                /// <param name="target">The fully-qualified string name of the Parameter to look for.</param>
                /// <returns>The Parameter referenced by target</returns>
                public virtual Parameter get_parameter(string target)
                {
                    if (_internal_params.TryGetValue(target, out var parameter)) {
                        return parameter;
                    }

                    var splits = target.Split('.');
                    if (splits.Length <= 1) return null;
                    foreach (var child in named_children().Where(nc => nc.name == splits[0])) {
                        var p = child.module.get_parameter(target.Remove(0, splits[0].Length + 1));
                        if (p is not null)
                            return p;
                    }
                    return null;
                }

                /// <summary>
                /// Adds a buffer to the module.
                ///
                /// This is typically used to register a buffer that should not to be considered a model parameter.For example, BatchNorm’s running_mean is not a parameter,
                /// but is part of the module’s state.Buffers, by default, are persistent and will be saved alongside parameters.
                /// </summary>
                /// <param name="name">Name of the buffer. The buffer can be accessed from this module using the given name</param>
                /// <param name="tensor">
                /// Buffer to be registered. If null, then operations that run on buffers, such as cuda(), are ignored.
                /// If null, the buffer is not included in the module’s state_dict.</param>
                /// <exception cref="ArgumentNullException"></exception>
                /// <exception cref="InvalidOperationException"></exception>
                public virtual void register_buffer(string name, Tensor tensor)
                {
                    if (tensor is null || tensor.handle == IntPtr.Zero)
                        throw new ArgumentNullException(nameof(tensor), "A null tensor cannot be registered as a buffer.");

                    if (!_internal_buffers.TryAdd(name, tensor))
                        throw new InvalidOperationException($"Tensor {name} is already registered.");
                }

                /// <summary>
                /// Adds a parameter to the module.
                /// </summary>
                /// <param name="name">Name of the parameter. The parameter can be accessed from this module using the given name</param>
                /// <param name="param">
                /// Buffer to be registered.
                /// If null, then operations that run on buffers, such as cuda(), are ignored.
                /// If null, the buffer is not included in the module’s state_dict.</param>
                /// <exception cref="ArgumentNullException"></exception>
                /// <exception cref="InvalidOperationException"></exception>
                public virtual void register_parameter(string name, Parameter param)
                {
                    if (param is null || param.handle == IntPtr.Zero)
                        throw new ArgumentNullException(nameof(param), "A null tensor cannot be registered as a parameter.");

                    if (!_internal_params.TryAdd(name, param))
                        throw new InvalidOperationException($"Parameter {name} is already registered.");
                }

                /// <summary>
                /// Register a submodule.
                /// </summary>
                /// <param name="name">Name of the submodule.</param>
                /// <param name="submodule">The module to register.</param>
                /// <exception cref="InvalidOperationException"></exception>
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
                        var p = value is Parameter parameter
                            ? parameter
                            : new Parameter(value, requires_grad: true);

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
                [return: MarshalAs(UnmanagedType.LPStr)]
                private static extern string THSNN_Module_name(HType module);

                public virtual string GetName()
                {
                    var res = THSNN_Module_name(handle);
                    CheckForErrors();
                    return res;
                }

                public virtual Tensor forward(Tensor t)
                    => throw new NotImplementedException("forward(t)");

                public virtual Tensor forward(Tensor x, Tensor y)
                    => throw new NotImplementedException("forward(x,y)");

                /// <summary>
                /// Save the parameters and buffers of the module to a disk location.
                /// </summary>
                /// <param name="location">The file path.</param>
                /// <param name="skip">A list of keys not to consider when saving the weights.</param>
                /// <returns></returns>
                public Module save(string location, IList<string> skip = null)
                {
                    using var stream = System.IO.File.OpenWrite(location);
                    using var writer = new System.IO.BinaryWriter(stream);
                    save(writer, skip);

                    return this;
                }

                /// <summary>
                /// Save the parameters and buffers of the module to a disk location.
                /// </summary>
                /// <param name="writer">A binary writer instance.</param>
                /// <param name="skip">A list of keys not to consider when saving the weights.</param>
                /// <returns></returns>
                public Module save(System.IO.BinaryWriter writer, IList<string> skip = null)
                {
                    var sd = state_dict();

                    // First, write how many entries.
                    save_state_dict(writer, sd, skip);

                    return this;
                }

                /// <summary>
                /// 
                /// </summary>
                /// <param name="writer">A binary writer instance.</param>
                /// <param name="skip">A list of keys not to consider when saving the weights.</param>
                /// <param name="sd">A dictionary containing all the buffers and parameters of the module.</param>
                public static void save_state_dict(System.IO.BinaryWriter writer, Dictionary<string, Tensor> sd, IList<string> skip = null)
                {
                    if (skip is not null && skip.Count > 0) {
                        // We need to make a copy, so that the passed-in 'sd' isn't modified.
                        var tmp = new Dictionary<string, Tensor>();
                        foreach (var kv in sd.Where(kv => !skip.Contains(kv.Key))) {
                            tmp.Add(kv.Key, kv.Value);
                        }
                        sd = tmp;
                    }

                    writer.Encode(sd.Count); // 4 bytes

                    foreach (var kvp in sd) {
                        writer.Write(kvp.Key);
                        kvp.Value.Save(writer);
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
                /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>              
                /// <returns>The module, with parameters and buffers loaded.</returns>
                /// <remarks>
                /// Using a skip list only prevents tensors in the target module from being modified, it
                /// does not alter any logic related to checking for matching tensor element types or entries.
                /// It may be necessary to also pass 'strict=false' to avoid exceptions.
                /// </remarks>
                public Module load(string location, bool strict = true, IList<string> skip = null)
                {
                    var dt = _deviceType;
                    var di = _deviceIndex;

                    cpu();

                    try {
                        using var stream = System.IO.File.OpenRead(location);
                        using var reader = new System.IO.BinaryReader(stream);
                        load(reader, strict, skip);
                    } finally {
                        to(dt, di);
                    }

                    return this;
                }

                /// <summary>
                /// Load the parameters and buffers
                /// </summary>
                /// <param name="reader">A binary reader instance.</param>
                /// <param name="strict">
                /// If true, will only load a module if it exactly corresponds to the current module's state.
                /// If false, will load the parameters and buffers that it finds in the saved file,
                /// leaving everything else alone.
                /// </param>
                /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
                /// <returns>The module, with parameters and buffers loaded.</returns>
                /// <remarks>
                /// Using a skip list only prevents tensors in the target module from being modified, it
                /// does not alter any logic related to checking for matching tensor element types or entries.
                /// It may be necessary to also pass 'strict=false' to avoid exceptions.
                /// </remarks>
                public virtual Module load(System.IO.BinaryReader reader, bool strict = true, IList<string> skip = null)
                {
                    skip ??= Array.Empty<string>();

                    var sd = state_dict();

                    // First, figure out how many entries.
                    var streamEntries = reader.Decode();

                    if (streamEntries != sd.Count && strict)
                        throw new ArgumentException($"Mismatched state_dict sizes: expected {sd.Count}, but found {streamEntries} entries.");

                    for (int i = 0; i < streamEntries; ++i) {
                        var key = reader.ReadString();
                        var found = sd.ContainsKey(key);
                        if (!found && strict)
                            throw new ArgumentException($"Mismatched module state names: the target modules does not have a submodule or buffer named '{key}'");

                        if (found) {
                            sd[key].Load(reader, skip: skip.Contains(key));
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
                public static Module Create<T>(string path)
                    where T : Module, new()
                {
                    var model = new T();
                    return model.load(path);
                }

                private delegate IntPtr ForwardFunctionC(IntPtr tensor);

                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_custom_module(
                    [MarshalAs(UnmanagedType.LPStr)] string name,
                    ForwardFunctionC forward,
                    out IntPtr pBoxedModule);

                /// <summary>
                /// Constructor for custom modules, i.e. those defined outside of TorchSharp.
                /// </summary>
                /// <param name="name">The name of the module. Useful for debugging purposes, mostly.</param>
                protected Module(string name) : this(IntPtr.Zero, IntPtr.Zero)
                {
                    this.name = name;

                    IntPtr ForwardNative(IntPtr t)
                    {
                        var input = new Tensor(t);
                        var output = forward(input);

                        // handles must live on - we don't own them, but
                        // the managed objects should go away.
                        input.DecoupleFromNativeHandle();

                        return output.DecoupleFromNativeHandle();
                    }

                    var res = THSNN_custom_module(name, ForwardNative, out var boxedHandle);
                    CheckForErrors();
                    handle = new HType(res, true);
                    this._forwardNative = ForwardNative;
                    boxedModule = new BoxedModule(boxedHandle);

                    // In this case, the parameter registration was not done yet.

                    foreach (var (parameterName, parameter) in _named_parameters()) {
                        register_parameter(parameterName, parameter);
                    }
                }

                protected virtual void RegisterComponents()
                {
                    if (_areComponentsRegistered) return;

                    foreach (var field in GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)) {

                        var fieldName = field.Name;
                        if (_internal_submodules.ContainsKey(fieldName) || _internal_params.ContainsKey(fieldName) || _internal_buffers.ContainsKey(fieldName)) continue;

                        var value = field.GetValue(this);

                        switch (value) {
                        case Module module:
                            register_module(fieldName, module);
                            break;
                        case Parameter param: // This test must come before the Tensor test
                            register_parameter(fieldName, param);
                            break;
                        case Tensor tensor:
                            register_buffer(fieldName, tensor);
                            break;
                        }
                    }

                    _areComponentsRegistered = true;
                }

                private bool _areComponentsRegistered;

                protected Utils.OrderedDict<string, Module> _internal_submodules = new Utils.OrderedDict<string, Module>();
                protected Utils.OrderedDict<string, Tensor> _internal_buffers = new Utils.OrderedDict<string, Tensor>();
                protected Utils.OrderedDict<string, Parameter> _internal_params = new Utils.OrderedDict<string, Parameter>();

                /// Keeps the callback delegate alive
                private ForwardFunctionC _forwardNative;
                protected string name;
            }

            internal class BoxedModule : IDisposable
            {
                internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
                        => SetHandle(preexistingHandle);

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

                ~BoxedModule() => Dispose(false);

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
