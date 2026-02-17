// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp.Modules;

using static System.Linq.Enumerable;
using static TorchSharp.torch;
using static TorchSharp.Utils.LEB128Codec;
using static TorchSharp.PInvoke.NativeMethods;
using TorchSharp.Utils;

#nullable enable
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
                protected internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle, Action<HType>? dispose = null)
                        : base(IntPtr.Zero, ownsHandle)
                    {
                        _dispose = dispose ?? THSNN_Module_dispose;
                        SetHandle(preexistingHandle);
                    }

                    public override bool IsInvalid => handle == IntPtr.Zero;

                    // This is just for marshalling
                    internal HType() : base(IntPtr.Zero, true)
                    {
                    }

                    protected override bool ReleaseHandle()
                    {
                        if (!IsInvalid && _dispose is not null) {
                            _dispose(this);
                        }
                        SetHandle(IntPtr.Zero);
                        return true;
                    }

                    protected override void Dispose(bool disposing)
                    {
                        if (disposing) {
                            ReleaseHandle();
                        }
                    }

                    private Action<HType>? _dispose;
                }

                internal HType handle;

                /// Stores the AnyModule corresponding to this module.
                internal BoxedModule? boxedModule;

                internal BoxedModule BoxedModule {
                    get {
                        if (boxedModule == null)
                            throw new InvalidOperationException("A Sequential or Loaded module may not be added to a Sequential");
                        return boxedModule;
                    }
                }

                internal Module(HType handle, IntPtr? boxedHandle)
                {
                    this.handle = handle;
                    boxedModule = boxedHandle.HasValue ? new BoxedModule(boxedHandle.Value) : null;

                    if (handle.IsInvalid) return;
                    register_p_and_b();
                }

                internal Module(IntPtr handle, IntPtr? boxedHandle, bool ownsHandle = true)
                {
                    this.handle = new HType(handle, ownsHandle);
                    boxedModule = boxedHandle.HasValue ? new BoxedModule(boxedHandle.Value) : null;

                    if (handle == IntPtr.Zero) return;
                    register_p_and_b();
                }

                private void register_p_and_b()
                {
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
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                /// <summary>
                /// Implements the .NET Dispose pattern.
                /// </summary>
                protected virtual void Dispose(bool disposing)
                {
                    if (disposing && !handle.IsInvalid) {

                        foreach (var (_, p) in named_buffers(false)) {
                            p.DetachFromDisposeScope().Dispose();
                        }
                        foreach (var (_, b) in named_parameters(false)) {
                            b.DetachFromDisposeScope().Dispose();
                        }

                        foreach (var (_, m) in named_modules()) {
                            m.Dispose();
                        }

                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                        boxedModule?.Dispose();
                    }
                }

                /// <summary>
                /// Moves and converts the parameters and buffers.
                /// </summary>
                /// <param name="device">The target device.</param>
                /// <param name="dtype">The target element type.</param>
                /// <param name="non_blocking">
                /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
                /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
                /// </param>
                protected internal virtual Module _to(Device device, ScalarType dtype, bool non_blocking)
                {
                    if (!dtype.IsFloatingPoint() && !dtype.IsComplex())
                        throw new ArgumentException($"nn.Module.to only accepts floating point or complex types, but got desired dtype={dtype.ToString()}");

                    if (device.type != DeviceType.CUDA) { device = new Device(device.type, -1); };

                    if (device.type == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                    InitializeDeviceType(device.type);
                    THSNN_Module_to_device_dtype(handle, (sbyte)dtype, (int)device.type, device.index, non_blocking);
                    CheckForErrors();

                    _toEpilog(device, dtype, non_blocking);

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
                protected internal virtual Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
                {
                    if (deviceType != DeviceType.CUDA) deviceIndex = -1;

                    if (deviceType == DeviceType.CUDA && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA is not available.");

                    if (deviceType != _deviceType || deviceIndex != _deviceIndex) {

                        InitializeDeviceType(deviceType);
                        THSNN_Module_to_device(handle, (int)deviceType, deviceIndex, non_blocking);
                        CheckForErrors();

                        _toEpilog(deviceType, deviceIndex, non_blocking);
                    }

                    Debug.Assert(_deviceType == DeviceType.CUDA || _deviceIndex == -1);

                    return this;
                }

                protected DeviceType _deviceType = DeviceType.CPU;
                protected int _deviceIndex = -1;

                /// <summary>
                /// Convert the parameters and buffers.
                /// </summary>
                /// <returns></returns>
                protected internal virtual Module _to(ScalarType dtype, bool non_blocking)
                {
                    if (!dtype.IsFloatingPoint() && !dtype.IsComplex())
                        throw new ArgumentException($"nn.Module.to only accepts floating point or complex types, but got desired dtype={dtype.ToString()}");

                    THSNN_Module_to_dtype(handle, (sbyte)dtype, non_blocking);
                    CheckForErrors();

                    _toEpilog(dtype, non_blocking);

                    return this;
                }

                protected void _toEpilog(ScalarType dtype, bool non_blocking)
                {
                    _toEpilog(dtype, null, non_blocking);
                }

                protected void _toEpilog(Device device, ScalarType dtype, bool non_blocking)
                {
                    _toEpilog(dtype, device, non_blocking);
                }

                protected void _toEpilog(DeviceType deviceType, int deviceIndex, bool non_blocking)
                {
                    _toEpilog(null, new Device(deviceType, deviceIndex), non_blocking);
                }

                protected virtual void _toEpilog(ScalarType? dtype, Device? device, bool non_blocking)
                {
                    if (dtype is null && device is null) throw new ArgumentNullException($"{nameof(dtype)} and {nameof(device)} are both null,");

                    foreach (var (_, sm) in named_children()) {
                        if (device is null) sm._to(dtype!.Value, non_blocking);
                        else if (dtype is null) sm._to(device.type, device.index, non_blocking);
                        else sm._to(device, dtype.Value, non_blocking);
                    }

                    var fieldsByComponentName =
                        GetFieldsRecursive(GetType(), BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance).ToDictionary(field => field.ComponentName());

                    var props = GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance);

                    // var propsByName = new Dictionary<string, PropertyInfo>();
                    // foreach (var p in props) {
                    //     // There may be duplicates, and this just overwrites it.
                    //     propsByName[p.Name] = p;
                    // }

                    var propsByName = props.ToDictionary(prop => prop.Name);

                    foreach (var (name, param) in named_parameters(false).ToList()) {

                        if (!ReplaceParameter(dtype, device, param, out var p)) continue;

                        if (propsByName.TryGetValue(name, out var property)) {
                            property.SetValue(this, p);
                        } else {
                            param?.Dispose();

                            ConditionallyRegisterParameter(name, p);

                            // If this parameter is a field, set it
                            if (fieldsByComponentName.TryGetValue(name, out var field))
                                field.SetValue(this, p);
                        }

                    }

                    foreach (var (name, buffer) in named_buffers(false).ToList()) {

                        if (!ReplaceBuffer(dtype, device, buffer, out var t)) continue;

                        if (propsByName.TryGetValue(name, out var property)) {
                            property.SetValue(this, t);
                        } else {
                            ConditionallyRegisterBuffer(name, t);
                            if (fieldsByComponentName.TryGetValue(name, out var field))
                                field.SetValue(this, t);
                        }
                    }

                    if (device is not null) {
                        _deviceType = device.type;
                        _deviceIndex = device.index;
                    }
                }

                protected static bool  ReplaceBuffer(ScalarType? dtype, Device? device, Tensor buffer, out Tensor? result)
                {
                    result = null;

                    if (!buffer.toWillCopy(dtype ?? buffer.dtype, device ?? buffer.device)) return false;

                    ScalarType bufferType =
                        dtype != null && (buffer.dtype.IsFloatingPoint() || buffer.dtype.IsComplex()) ? dtype.Value : buffer.dtype;

                    // Buffers don't get grads so we don't need to detach them afterwards
                    result = buffer.to(bufferType, device ?? buffer.device, disposeAfter: true).DetachFromDisposeScope();
                    return true;
                }

                protected static bool ReplaceParameter(ScalarType? dtype, Device? device, Parameter param, out Parameter? p)
                {
                    Tensor? grad = param.grad;
                    p = null;

                    if (!param.toWillCopy(dtype ?? param.dtype, device ?? param.device) &&
                        (grad is null || !grad.toWillCopy(dtype ?? param.dtype, device ?? param.device)))
                        return false;

                    ScalarType paramType =
                        dtype != null && (param.dtype.IsFloatingPoint() || param.dtype.IsComplex()) ? dtype.Value : param.dtype;

                    // When moving the parameter, we don't want the autograd to track this movement on the graph.
                    // In addition, we need the new tensor to be a leaf to accumulate gradients, so if we didn't
                    // disable grad we would need to call .detach() on the moved tensor.
                    using (var d = torch.no_grad()) {
                        p = new Parameter(
                            data: param.to(paramType, device ?? param.device),
                            requires_grad: param.requires_grad);
                        _ = p.DetachFromDisposeScope();

                        // Copy the gradient over as well, if it exists
                        if (grad is not null) {
                            using var newGrad = grad.to(paramType, device ?? param.device)
                                .with_requires_grad(grad.requires_grad);
                            p.grad = newGrad;
                        }
                    }
                    return true;
                }

                private static IEnumerable<FieldInfo> GetFieldsRecursive(Type type, BindingFlags bindingFlags) {

                    Type? currentType = type;
                    var seenFields = new HashSet<string>();  // Track field names we've seen

                    while (currentType != null && currentType != typeof(object)) {
                        var fields = currentType.GetFields(bindingFlags);

                        foreach (var field in fields) {
                            if (seenFields.Add(field.Name))
                                yield return field;
                        }

                        currentType = currentType.BaseType;
                    }
                }

                /// <summary>
                /// Moves and converts the parameters and buffers.
                /// </summary>
                /// <param name="other">The tensor serving as a template.</param>
                /// <param name="non_blocking">
                /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
                /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
                /// </param>
                /// <returns></returns>
                public Module _to(Tensor other, bool non_blocking)
                {
                    _to(other.dtype, non_blocking);
                    return _to(other.device_type, other.device_index, non_blocking);
                }



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
                /// Sets the module in training mode.
                /// </summary>
                /// <remarks>
                /// This has any effect only on certain modules.See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.Dropout, BatchNorm, etc.
                /// </remarks>
                public virtual void train(bool train = true)
                {
                    THSNN_Module_train(handle, train);
                    CheckForErrors();
                    foreach (var (_, m) in named_children()) { m.train(train); }
                }

                /// <summary>
                /// Sets the module in evaluation mode.
                /// </summary>
                /// <remarks>
                /// This has any effect only on certain modules.See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.Dropout, BatchNorm, etc.
                /// </remarks>
                public virtual void eval()
                {
                    train(false);
                }

                /// <summary>
                /// Check whether the module is set to training or evaluation mode.
                /// </summary>
                public virtual bool training {
                    get {
                        var res = THSNN_Module_is_training(handle);
                        CheckForErrors();
                        return res;
                    }
                }

                public virtual void zero_grad(bool set_to_none = true)
                {
                    THSNN_Module_zero_grad(handle, set_to_none);
                    CheckForErrors();

                    foreach (var (_, p) in named_parameters()) {
                        using var grad = p.grad;
                        if (grad is not null) {
                            if (set_to_none) {
                                p.grad = null;
                            } else {
                                grad.zero_();
                            }
                        }
                    }
                }

                /// <summary>
                /// Returns an enumerable of module buffers, yielding both the name of the buffer as well as the buffer itself.
                /// </summary>
                /// <param name="recurse">If true, then yields buffers of this module and all submodules. Otherwise, yields only buffers that are direct members of this module.</param>
                /// <param name="include_nonpersistent">Include buffers that are not persistent.</param>
                /// <returns>(string, torch.Tensor) – Tuple containing the name and buffer</returns>
                public virtual IEnumerable<(string name, Tensor buffer)> named_buffers(bool recurse = true, bool include_nonpersistent = true)
                {
                    var seen = new HashSet<IntPtr>();
                    seen.Add(IntPtr.Zero);              // Ignore invalid buffers.

                    foreach (var nsm in _internal_buffers) {
                        if (seen.Contains(nsm.Item2.Item1.handle) || !nsm.Item2.Item2) continue;
                        seen.Add(nsm.Item2.Item1.handle);
                        yield return (nsm.Item1, nsm.Item2.Item1);
                    }

                    if (!recurse) yield break;

                    foreach (var (submoduleName, subModule) in _internal_submodules) {
                        foreach (var (bufferName, buffer) in subModule.named_buffers(true, include_nonpersistent)) {
                            if (seen.Contains(buffer.handle)) continue;
                            seen.Add(buffer.handle);
                            yield return ($"{submoduleName}.{bufferName}", buffer);
                        }
                    }

                }

                /// <summary>
                /// Returns an enumerable of buffers.
                /// </summary>
                /// <param name="recurse">If true, then yields buffers of this module and all submodules. Otherwise, yields only buffers that are direct members of this module.</param>
                /// <param name="include_nonpersistent">Include buffers that are not persistent.</param>
                public virtual IEnumerable<Tensor> buffers(bool recurse = true, bool include_nonpersistent = true) => named_buffers(recurse, include_nonpersistent).Select(np => np.buffer);

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
                /// Returns an enumerable of modules.
                /// </summary>
                public virtual IEnumerable<Module> modules() => named_modules().Select(np => np.module);

                /// <summary>
                /// Returns an enumerable of immediate modules.
                /// </summary>
                public virtual IEnumerable<Module> children() => named_children().Select(np => np.module);

                /// <summary>
                /// Returns a dictionary containing a whole state of the module.
                ///
                /// Both parameters and persistent buffers(e.g.running averages) are included.Keys are corresponding parameter and buffer names.
                /// Parameters and buffers set to null are not included.
                /// </summary>
                /// <param name="destination">An optional dictionary where the state should be accumulated.</param>
                /// <param name="prefix">A prefix string to use when entering the name of entries into the dictionary.</param>
                /// <returns></returns>
                public virtual Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null, string? prefix = null)
                {
                    destination ??= new Dictionary<string, Tensor>();

                    foreach (var p in named_parameters()) {
                        var key = string.IsNullOrEmpty(prefix) ? $"{p.name}" : $"{prefix}.{p.name}";
                        destination.TryAdd(key, p.Item2);
                    }

                    foreach (var p in named_buffers(include_nonpersistent: false) ) {
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
                public virtual (IList<string> missing_keys, IList<string> unexpected_keyes) load_state_dict(Dictionary<string, Tensor> source, bool strict = true, IList<string>? skip = null)
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

                    // The copy_ operation is an in-place operation which can't be performed on a leaf variable which
                    // requires_grad. Therefore, we will perform the copy in a no_grad context.
                    using var d = torch.no_grad();

                    foreach (var key in source.Keys) {
                        if (skip.Contains(key)) continue;
                        if (destination.ContainsKey(key)) {
                            destination[key].copy_(source[key]);
                        }
                    }

                    return (missing, unexpected);
                }

                protected virtual (string name, Parameter parameter)[] _named_parameters()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSNN_Module_get_named_parameters(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i])!, new Parameter(x))).ToArray();
                }

                protected virtual (string name, Tensor buffer)[] _named_buffers()
                {
                    using var pa = new PinnedArray<IntPtr>();
                    using var sa = new PinnedArray<IntPtr>();
                    THSNN_Module_get_named_buffers(handle, pa.CreateArray, sa.CreateArray);
                    CheckForErrors();
                    var ptrArray = pa.Array;
                    var strArray = sa.Array;

                    return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i])!, new Tensor(x))).ToArray();
                }

                /// <summary>
                /// Returns an enumerable of module parameters, yielding both the name of the parameter as well as the parameter itself.
                /// </summary>
                /// <param name="recurse">If true, then yields parameters of this module and all submodules. Otherwise, yields only parameters that are direct members of this module.</param>
                /// <returns>(string, Parameter) – Tuple containing the name and parameter</returns>
                public virtual IEnumerable<(string name, Parameter parameter)> named_parameters(bool recurse = true)
                {
                    var seen = new HashSet<IntPtr>();
                    seen.Add(IntPtr.Zero);              // Ignore invalid parameters.

                    foreach (var nsm in _internal_params) {
                        if (seen.Contains(nsm.Item2.handle)) continue;
                        seen.Add(nsm.Item2.handle);
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
                public virtual Tensor? get_buffer(string target)
                {
                    if (_internal_buffers.TryGetValue(target, out var buffer)) {
                        return buffer.Item1;
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
                public virtual Parameter? get_parameter(string target)
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
                /// If null, the buffer is not included in the module’s state_dict.
                /// </param>
                /// <param name="persistent">Whether the buffer is part of this module’s state_dict.</param>
                /// <exception cref="ArgumentNullException"></exception>
                /// <exception cref="InvalidOperationException"></exception>
                public virtual void register_buffer(string name, Tensor tensor, bool persistent = true)
                {
                    if (tensor is null || tensor.handle == IntPtr.Zero)
                        throw new ArgumentNullException(nameof(tensor), "A null tensor cannot be registered as a buffer.");

                    if (!_internal_buffers.TryAdd(name, (tensor, persistent)))
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
                /// Alias for register_module()
                /// </summary>
                /// <param name="name">
                /// name of the child module.
                /// The child module can be accessed from this module using the given name
                /// </param>
                /// <param name="module">child module to be added to the module.</param>
                /// <exception cref="ArgumentException"></exception>
                /// <exception cref="InvalidOperationException"></exception>
                public void add_module(string name, Module module)
                    => register_module(name, module);

                /// <summary>
                /// Register a submodule.
                /// </summary>
                /// <param name="name">Name of the submodule.</param>
                /// <param name="submodule">The module to register.</param>
                /// <exception cref="ArgumentException"></exception>
                /// <exception cref="InvalidOperationException"></exception>
                public virtual void register_module(string name, Module submodule)
                {
                    if (submodule is null || submodule.handle.IsInvalid) {
                        if (_internal_submodules.ContainsKey(name)) {
                            _internal_submodules.Remove(name);
                        }
                    } else {
                        if (name.Contains(".")) {
                            throw new ArgumentException($"module name can't contain \".\", got: {name}");
                        }
                        if (string.IsNullOrEmpty(name)) {
                            throw new ArgumentException("module name can't be empty string \"\"");
                        }
                        if (_internal_submodules.ContainsKey(name)) {
                            throw new InvalidOperationException($"Sub-module {name} is already registered.");
                        }

                        submodule.RegisterComponents();

                        _internal_submodules.Add(name, submodule);
                    }
                }

                protected void ConditionallyRegisterParameter(string name, Tensor value)
                {
                    ConditionallyRegisterParameter(name, value as Parameter);
                }

                protected void ConditionallyRegisterParameter(string name, Parameter? value)
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

                protected void ConditionallyRegisterBuffer(string name, Tensor? value, bool persistent = true)
                {
                    if (value is null) {
                        if (_internal_buffers.ContainsKey(name)) {
                            _internal_buffers.Remove(name);
                        }
                    } else {
                        if (_internal_buffers.ContainsKey(name)) {
                            _internal_buffers[name] = (value, persistent);
                        } else {
                            _internal_buffers.Add(name, (value, persistent));
                        }
                    }
                }

                public virtual string GetName()
                {
                    if (!string.IsNullOrEmpty(this.name)) return this.name!;

                    var res = THSNN_Module_name(handle);
                    CheckForErrors();
                    return res;
                }

                /// <summary>
                /// Save the parameters and buffers of the module to a disk location.
                /// </summary>
                /// <param name="location">The file path.</param>
                /// <param name="skip">A list of keys not to consider when saving the weights.</param>
                /// <returns></returns>
                public Module save(string location, IList<string>? skip = null)
                {
                    using var stream = System.IO.File.Create(location);
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
                public Module save(System.IO.BinaryWriter writer, IList<string>? skip = null)
                {
                    var sd = state_dict();

                    // First, write how many entries.
                    save_state_dict(writer, sd, skip);

                    return this;
                }

                /// <summary>
                /// Save the parameters and buffers of the module to a disk location.
                /// </summary>
                /// <param name="stream">A writable stream instance.</param>
                /// <param name="skip">A list of keys not to consider when saving the weights.</param>
                /// <returns></returns>
                public Module save(System.IO.Stream stream, IList<string> ?skip = null)
                {
                    using var writer = new System.IO.BinaryWriter(stream);
                    return save(writer, skip);
                }

                /// <summary>
                ///
                /// </summary>
                /// <param name="writer">A binary writer instance.</param>
                /// <param name="skip">A list of keys not to consider when saving the weights.</param>
                /// <param name="sd">A dictionary containing all the buffers and parameters of the module.</param>
                public static void save_state_dict(System.IO.BinaryWriter writer, Dictionary<string, Tensor> sd, IList<string>? skip = null)
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
                /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
                /// <returns>The module, with parameters and buffers loaded.</returns>
                /// <remarks>
                /// Using a skip list only prevents tensors in the target module from being modified, it
                /// does not alter any logic related to checking for matching tensor element types or entries.
                /// It may be necessary to also pass 'strict=false' to avoid exceptions.
                /// </remarks>
                public virtual Module load(string location, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
                {
                    if (!System.IO.File.Exists(location))
                        throw new System.IO.FileNotFoundException(location);

                    using var stream = System.IO.File.OpenRead(location);
                    using var reader = new System.IO.BinaryReader(stream);
                    load(reader, strict, skip, loadedParameters);

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
                /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
                /// <returns>The module, with parameters and buffers loaded.</returns>
                /// <remarks>
                /// Using a skip list only prevents tensors in the target module from being modified, it
                /// does not alter any logic related to checking for matching tensor element types or entries.
                /// It may be necessary to also pass 'strict=false' to avoid exceptions.
                /// </remarks>
                public virtual Module load(System.IO.BinaryReader reader, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
                {
                    skip ??= Array.Empty<string>();

                    var dt = _deviceType;
                    var di = _deviceIndex;

                    if (dt != DeviceType.CPU) this.cpu();

                    try {
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

                            if (found && !skip.Contains(key)) {
                                sd[key].Load(reader);
                            }
                            else {
                                // Even if we are skipping this tensor, we need to load it in so that
                                // the BinaryReader seeks forward in the input stream.
                                TensorExtensionMethods.Load(reader, skip: true);
                            }

                            loadedParameters?.Add(key, found);
                        }
                    } finally {
                        if (dt != DeviceType.CPU) _to(dt, di, false);
                    }

                    return this;
                }

                /// <summary>
                /// Load the parameters and buffers
                /// </summary>
                /// <param name="stream">A readable stream instance.</param>
                /// <param name="strict">
                /// If true, will only load a module if it exactly corresponds to the current module's state.
                /// If false, will load the parameters and buffers that it finds in the saved file,
                /// leaving everything else alone.
                /// </param>
                /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
                /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
                /// <returns>The module, with parameters and buffers loaded.</returns>
                /// <remarks>
                /// Using a skip list only prevents tensors in the target module from being modified, it
                /// does not alter any logic related to checking for matching tensor element types or entries.
                /// It may be necessary to also pass 'strict=false' to avoid exceptions.
                /// </remarks>
                public Module load(System.IO.Stream stream, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
                {
                    using var reader = new System.IO.BinaryReader(stream);
                    return load(reader, strict, skip, loadedParameters);
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
                        var output = ((nn.Module<Tensor, Tensor>)this).call(input);

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

                    _init_parameters();
                }

                private void _init_parameters()
                {
                    // In this case, the parameter registration was not done yet.
                    foreach (var (parameterName, parameter) in _named_parameters()) {
                        register_parameter(parameterName, parameter);
                    }
                }

                protected virtual void RegisterComponents()
                {
                    if (_areComponentsRegistered) return;

                    foreach (var field in GetFieldsRecursive(GetType(), BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)) {

                        var fieldName = field.ComponentName();
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

                protected static (Device device, ScalarType dtype) GetDefaultDeviceAndType(Device? device = null, ScalarType? dtype = null)
                {
                    if (!dtype.HasValue)
                        dtype = get_default_dtype();

                    if (device is null)
                    {
                        device = get_default_device();
                    }

                    return (device, dtype.Value);
                }

                internal T MoveModule<T>(Device? device, ScalarType? dtype) where T : Module
                {
                    T module = (T)this;
                    var (targetDevice, targetDtype) = GetDefaultDeviceAndType(device, dtype);
                    return (T)module._to(targetDevice, targetDtype, false);
                }

                protected void ClearModules() { _internal_submodules.clear(); }

                private bool _areComponentsRegistered;

                protected Utils.OrderedDict<string, Module> _internal_submodules = new Utils.OrderedDict<string, Module>();
                protected Utils.OrderedDict<string, (Tensor, bool)> _internal_buffers = new Utils.OrderedDict<string, (Tensor, bool)>();
                protected Utils.OrderedDict<string, Parameter> _internal_params = new Utils.OrderedDict<string, Parameter>();

                /// Keeps the callback delegate alive
                private ForwardFunctionC? _forwardNative;
                protected string? name;
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

            /// <summary>
            /// Interface for concrete modules with a forward() that takes a single argument.
            /// </summary>
            /// <typeparam name="T">The argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public interface IModule<T, TResult>
            {
                public TResult call(T input1);
            }

            /// <summary>
            /// Interface for concrete modules with a forward() that takes two arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public interface IModule<T1, T2, TResult>
            {
                public abstract TResult call(T1 input1, T2 input2);
            }

            /// <summary>
            /// Interface for concrete modules with a forward() that takes three arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public interface IModule<T1, T2, T3, TResult>
            {
                public abstract TResult call(T1 input1, T2 input2, T3 input3);
            }

            /// <summary>
            /// Interface for concrete modules with a forward() that takes four arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T4">The fourth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public interface IModule<T1, T2, T3, T4, TResult>
            {
                public abstract TResult call(T1 input1, T2 input2, T3 input3, T4 input4);
            }

            /// <summary>
            /// Interface for concrete modules with a forward() that takes five arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T4">The fourth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T5">The fifth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public interface IModule<T1, T2, T3, T4, T5, TResult>
            {
                public abstract TResult call(T1 input1, T2 input2, T3 input3, T4 input4, T5 input5);
            }

            /// <summary>
            /// Interface for concrete modules with a forward() that takes six arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T4">The fourth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T5">The fifth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T6">The sixth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public interface IModule<T1, T2, T3, T4, T5, T6, TResult>
            {
                public abstract TResult call(T1 input1, T2 input2, T3 input3, T4 input4, T5 input5, T6 input6);
            }

            /// <summary>
            /// Represents a module that accepts 'hook' to the module logic.
            /// </summary>
            public class HookableModule<TPreHook,TPostHook> : Module
            {
                protected HookableModule(string name) : base(name) { }

                protected HookableModule(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal HookableModule(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

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

                public HookRemover register_forward_hook(Action<Module> hook)
                {
                    var key = Guid.NewGuid().ToString();
                    module_post_hooks.Add(key, hook);
                    return new HookRemover(this, key);
                }

                public HookRemover register_forward_pre_hook(Action<Module> hook)
                {
                    var key = Guid.NewGuid().ToString();
                    module_pre_hooks.Add(key, hook);
                    return new HookRemover(this, key);
                }

                private void remove(string key)
                {
                    if (pre_hooks.ContainsKey(key)) pre_hooks.Remove(key);
                    if (post_hooks.ContainsKey(key)) post_hooks.Remove(key);
                    if (module_pre_hooks.ContainsKey(key)) module_pre_hooks.Remove(key);
                    if (module_post_hooks.ContainsKey(key)) module_post_hooks.Remove(key);
                }

                protected Dictionary<string, TPreHook> pre_hooks = new Dictionary<string, TPreHook>();
                protected Dictionary<string, TPostHook> post_hooks = new Dictionary<string, TPostHook>();

                protected Dictionary<string, Action<Module>> module_pre_hooks = new Dictionary<string, Action<Module>>();
                protected Dictionary<string, Action<Module>> module_post_hooks = new Dictionary<string, Action<Module>>();

                /// <summary>
                /// Used to remove a specific hook, following the PyTorch API design.
                /// </summary>
                /// <remarks>The name and namespace of this class is not the same as in PyTorch, but serves the same purpose.</remarks>
                public class HookRemover
                {
                    public HookRemover(HookableModule<TPreHook, TPostHook> module, string key)
                    {
                        this.module = module;
                        this.key = key;
                    }

                    public void remove()
                    {
                        module.remove(key);
                    }

                    private HookableModule<TPreHook, TPostHook> module;
                    private string key;
                }
            }

            /// <summary>
            /// Base class for concrete modules with a forward() that takes a single argument.
            /// </summary>
            /// <typeparam name="T">The argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public abstract class Module<T, TResult> : HookableModule<Func<Module<T,TResult>, T, T>, Func<Module<T, TResult>, T, TResult, TResult>>, IModule<T, TResult>
            {
                protected Module(string name) : base(name) { }
                protected Module(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal Module(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`forward` will not invoke any registered hooks for the module.</remarks>
                public abstract TResult forward(T input);

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`call` will invoke any registered hooks for the module.</remarks>
                public TResult call(T input)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in module_pre_hooks.Values) {
                        hook(this);
                    }

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

                    foreach (var hook in module_post_hooks.Values) {
                        hook(this);
                    }

                    return result;
                }
            }

            /// <summary>
            /// Base class for concrete modules with a forward() that takes two arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public abstract class Module<T1, T2, TResult> : HookableModule<Func<Module<T1, T2, TResult>, T1, T2, (T1, T2)?>, Func<Module<T1, T2, TResult>, T1, T2, TResult, TResult>>, IModule<T1, T2, TResult>
            {
                protected Module(string name) : base(name) { }
                protected Module(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal Module(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`forward` will not invoke any registered hooks for the module.</remarks>
                public abstract TResult forward(T1 input1, T2 input2);

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`call` will invoke any registered hooks for the module.</remarks>
                public TResult call(T1 input1, T2 input2)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in module_pre_hooks.Values) {
                        hook(this);
                    }

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input1, input2);
                        if (modified.HasValue) {
                            input1 = modified.Value.Item1;
                            input2 = modified.Value.Item2;
                        }
                    }

                    var result = forward(input1,  input2);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input1, input2, result);
                        if (modified is not null)
                            result = modified;
                    }

                    foreach (var hook in module_post_hooks.Values) {
                        hook(this);
                    }

                    return result;
                }
            }

            /// <summary>
            /// Base class for concrete modules with a forward() that takes three arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public abstract class Module<T1, T2, T3, TResult> : HookableModule<Func<Module<T1, T2, T3, TResult>, T1, T2, T3, (T1, T2, T3)?>, Func<Module<T1, T2, T3, TResult>, T1, T2, T3, TResult, TResult>>, IModule<T1, T2, T3, TResult>
            {
                protected Module(string name) : base(name) { }
                protected Module(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal Module(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`forward` will not invoke any registered hooks for the module.</remarks>
                public abstract TResult forward(T1 input1, T2 input2, T3 input3);

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`call` will invoke any registered hooks for the module.</remarks>
                public TResult call(T1 input1, T2 input2, T3 input3)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in module_pre_hooks.Values) {
                        hook(this);
                    }

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input1, input2, input3);
                        if (modified.HasValue) {
                            input1 = modified.Value.Item1;
                            input2 = modified.Value.Item2;
                            input3 = modified.Value.Item3;
                        }
                    }

                    var result = forward(input1, input2, input3);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, result);
                        if (modified is not null)
                            result = modified;
                    }

                    foreach (var hook in module_post_hooks.Values) {
                        hook(this);
                    }

                    return result;
                }
            }

            /// <summary>
            /// Base class for concrete modules with a forward() that takes four arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T4">The fourth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public abstract class Module<T1, T2, T3, T4, TResult> : HookableModule<Func<Module<T1, T2, T3, T4, TResult>, T1, T2, T3, T4, (T1, T2, T3, T4)?>, Func<Module<T1, T2, T3, T4, TResult>, T1, T2, T3, T4, TResult, TResult>>, IModule<T1, T2, T3, T4, TResult>
            {
                protected Module(string name) : base(name) { }
                protected Module(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal Module(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`forward` will not invoke any registered hooks for the module.</remarks>
                public abstract TResult forward(T1 input1, T2 input2, T3 input3, T4 input4);

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`call` will invoke any registered hooks for the module.</remarks>
                public TResult call(T1 input1, T2 input2, T3 input3, T4 input4)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in module_pre_hooks.Values) {
                        hook(this);
                    }

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, input4);
                        if (modified.HasValue) {
                            input1 = modified.Value.Item1;
                            input2 = modified.Value.Item2;
                            input3 = modified.Value.Item3;
                            input4 = modified.Value.Item4;
                        }
                    }

                    var result = forward(input1, input2, input3, input4);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, input4, result);
                        if (modified is not null)
                            result = modified;
                    }

                    foreach (var hook in module_post_hooks.Values) {
                        hook(this);
                    }

                    return result;
                }
            }

            /// <summary>
            /// Base class for concrete modules with a forward() that takes five arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T4">The fourth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T5">The fifth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public abstract class Module<T1, T2, T3, T4, T5,TResult> : HookableModule<Func<Module<T1, T2, T3, T4, T5, TResult>, T1, T2, T3, T4, T5, (T1, T2, T3, T4, T5)?>, Func<Module<T1, T2, T3, T4, T5, TResult>, T1, T2, T3, T4, T5, TResult, TResult>>, IModule<T1, T2, T3, T4, T5, TResult>
            {
                protected Module(string name) : base(name) { }
                protected Module(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal Module(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`forward` will not invoke any registered hooks for the module.</remarks>
                public abstract TResult forward(T1 input1, T2 input2, T3 input3, T4 input4, T5 input5);

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`call` will invoke any registered hooks for the module.</remarks>
                public TResult call(T1 input1, T2 input2, T3 input3, T4 input4, T5 input5)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in module_pre_hooks.Values) {
                        hook(this);
                    }

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, input4, input5);
                        if (modified.HasValue) {
                            input1 = modified.Value.Item1;
                            input2 = modified.Value.Item2;
                            input3 = modified.Value.Item3;
                            input4 = modified.Value.Item4;
                            input5 = modified.Value.Item5;
                        }
                    }

                    var result = forward(input1, input2, input3, input4, input5);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, input4, input5, result);
                        if (modified is not null)
                            result = modified;
                    }

                    foreach (var hook in module_post_hooks.Values) {
                        hook(this);
                    }

                    return result;
                }
            }

            /// <summary>
            /// Base class for concrete modules with a forward() that takes six arguments.
            /// </summary>
            /// <typeparam name="T1">The first argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T2">The second argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T3">The third argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T4">The fourth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T5">The fifth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="T6">The sixth argument type of the module's forward() function.</typeparam>
            /// <typeparam name="TResult">The return type of the module's forward() function.</typeparam>
            public abstract class Module<T1, T2, T3, T4, T5, T6, TResult> : HookableModule<Func<Module<T1, T2, T3, T4, T5, T6, TResult>, T1, T2, T3, T4, T5, T6, (T1, T2, T3, T4, T5, T6)?>, Func<Module<T1, T2, T3, T4, T5, T6, TResult>, T1, T2, T3, T4, T5, T6, TResult, TResult>>, IModule<T1, T2, T3, T4, T5, T6, TResult>
            {
                protected Module(string name) : base(name) { }
                protected Module(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
                internal Module(HType handle, IntPtr? boxedHandle) : base(handle, boxedHandle) { }

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`forward` will not invoke any registered hooks for the module.</remarks>
                public abstract TResult forward(T1 input1, T2 input2, T3 input3, T4 input4, T5 input5, T6 input6);

                /// <summary>
                /// Invoke the logic of the module.
                /// </summary>
                /// <remarks>`call` will invoke any registered hooks for the module.</remarks>
                public TResult call(T1 input1, T2 input2, T3 input3, T4 input4, T5 input5, T6 input6)
                {
                    // Call pre-hooks, if available.

                    foreach (var hook in module_pre_hooks.Values) {
                        hook(this);
                    }

                    foreach (var hook in pre_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, input4, input5, input6);
                        if (modified.HasValue) {
                            input1 = modified.Value.Item1;
                            input2 = modified.Value.Item2;
                            input3 = modified.Value.Item3;
                            input4 = modified.Value.Item4;
                            input5 = modified.Value.Item5;
                            input6 = modified.Value.Item6;
                        }
                    }

                    var result = forward(input1, input2, input3, input4, input5, input6);

                    // Call post-hooks, if available.

                    foreach (var hook in post_hooks.Values) {
                        var modified = hook(this, input1, input2, input3, input4, input5, input6, result);
                        if (modified is not null)
                            result = modified;
                    }

                    foreach (var hook in module_post_hooks.Values) {
                        hook(this);
                    }

                    return result;
                }
            }
        }
    }

    public static class ModuleExtensionMethods
    {
        /// <summary>
        /// Converts the parameters and buffers.
        /// </summary>
        /// <param name="module">The module to move</param>
        /// <param name="type">The target element type.</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T to<T>(this T module, torch.ScalarType type, bool non_blocking = false) where T : torch.nn.Module
        {
            return (T)module._to(type, non_blocking);
        }

        /// <summary>
        /// Moves and converts the parameters and buffers.
        /// </summary>
        /// <param name="module">The module to move</param>
        /// <param name="device">The target device.</param>
        /// <param name="type">The target element type.</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T to<T>(this T module, torch.Device device, torch.ScalarType type, bool non_blocking = false) where T : torch.nn.Module
        {
            return (T)module._to(device, type, non_blocking);
        }

        /// <summary>
        /// Moves the parameters and buffers.
        /// </summary>
        /// <param name="module">The module to move</param>
        /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
        /// <param name="deviceIndex">The optional device index.</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T to<T>(this T module, DeviceType deviceType, int deviceIndex = -1, bool non_blocking = false) where T : torch.nn.Module
        {
            return (T)module._to(deviceType, deviceIndex, non_blocking);
        }

        /// <summary>
        /// Moves the parameters and buffers.
        /// </summary>
        /// <param name="module">The module to move</param>"
        /// <param name="device">The target device</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        /// <returns></returns>
        public static T to<T>(this T module, Device device, bool non_blocking = false) where T : torch.nn.Module => (T)module._to(device.type, device.index, non_blocking);

        /// <summary>
        /// Moves the parameters and buffers.
        /// </summary>
        /// <param name="module">The module to move</param>
        /// <param name="device">A string denoting the target device.</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        /// <returns></returns>
        /// <remarks>Relies on the Device constructor to parse the string.</remarks>
        public static T to<T>(this T module, string device, bool non_blocking = false) where T : torch.nn.Module
        {
            var dev = new Device(device);
            return (T)module._to(dev.type, dev.index, non_blocking);
        }

        /// <summary>
        /// Moves and converts the parameters and buffers.
        /// </summary>
        /// <param name="module">The module to move</param>
        /// <param name="other">The tensor serving as a template.</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        /// <returns></returns>
        public static T to<T>(this T module, Tensor other, bool non_blocking = false) where T : torch.nn.Module
        {
            return (T)module._to(other.device, other.dtype, non_blocking);
        }

        /// <summary>
        /// Moves all model parameters and buffers to the CPU.
        /// </summary>
        public static T cpu<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => (T)module._to(DeviceType.CPU, -1, non_blocking);

        /// <summary>
        /// Moves all model parameters and buffers to a GPU.
        ///
        /// This also makes associated parameters and buffers different objects.So it should be called before constructing optimizer if the module will live on GPU while being optimized.
        /// </summary>
        /// <param name="module">The module to move</param>
        /// <param name="deviceIndex">If specified, all parameters will be copied to that device</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T cuda<T>(this T module, int deviceIndex = -1, bool non_blocking = false) where T : torch.nn.Module => (T)module._to(DeviceType.CUDA, deviceIndex, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Bool`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @bool<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Bool);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Byte`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @byte<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Byte, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Int8`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @char<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Int8, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Int16`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @short<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Int16, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Int32`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @int<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Int32, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Int64`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @long<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Int64, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Float16`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T half<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Float16, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.BFloat16`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T bfloat16<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.BFloat16, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Float32`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @float<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Float32, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.Float64`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T @double<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.Float64, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.ComplexFloat32`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T cfloat<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.ComplexFloat32, non_blocking);

        /// <summary>
        /// Converts all model parameters and buffers to `ScalarType.ComplexFloat64`.
        /// </summary>
        /// <param name="module">The module to convert</param>
        /// <param name="non_blocking">
        /// When non_blocking is set, it tries to convert/move asynchronously with respect to the host if possible,
        /// e.g., moving CPU Tensors with pinned memory to CUDA devices.
        /// </param>
        public static T cdouble<T>(this T module, bool non_blocking = false) where T : torch.nn.Module => module.to(ScalarType.ComplexFloat64, non_blocking);
    }

    public static class FieldInfoExtensionMethods
    {
        /// <summary>
        /// Retrieves the custom component name defined by the ComponentNameAttribute for a given field,
        /// or defaults to the field's own name if the attribute is not present.
        /// </summary>
        /// <param name="field">The field for which to retrieve the component name.</param>
        /// <returns>The custom component name if specified, otherwise the field's name.</returns>
        public static string ComponentName(this FieldInfo field) => field.GetCustomAttribute<ComponentNameAttribute>()?.Name ?? field.Name;
    }

    internal delegate IntPtr ForwardFunctionC(IntPtr tensor);
}
