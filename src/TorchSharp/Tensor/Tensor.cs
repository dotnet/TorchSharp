// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Diagnostics.Contracts;
using Google.Protobuf;
using static Tensorboard.TensorShapeProto.Types;

using TorchSharp.PInvoke;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Represents a TorchSharp tensor.
        /// </summary>
        public partial class Tensor : IDisposable
        {
            /// <summary>
            /// A handle to the underlying native tensor.
            /// This field should only be used in rare circumstances. Instead, use the 'Handle' property, which
            /// validates that the handle is not zero.
            /// </summary>
            internal IntPtr handle;

            static long _totalCount = 0;
            static long _peakCount = 0;

            internal DisposeScope? OwningDisposeScope { get; set; }

            internal Tensor(IntPtr handle)
            {
                this.handle = handle;
                System.Threading.Interlocked.Increment(ref _totalCount);
                _peakCount = Math.Max(_totalCount, _peakCount);
                OwningDisposeScope = DisposeScopeManager.ThreadSingleton.RegisterOnCurrentDisposeScope(this);
            }

            /// <summary>
            /// Allows external packages to create tensors from the same native pointers that TorchSharp uses.
            /// </summary>
            /// <param name="handle">A pointer to a native at::Tensor.</param>
            /// <returns>A Tensor reference</returns>
            public static Tensor UnsafeCreateTensor(IntPtr handle) => new Tensor(handle);

            /// <summary>
            /// TODO
            /// </summary>
            /// <param name="obj"></param>
            public override bool Equals(object? obj)
            {
                return (obj is Tensor) && this.Equals((obj as Tensor)!);
            }

            /// <summary>
            /// TODO
            /// </summary>
            public override int GetHashCode() => base.GetHashCode();

            /// <summary>
            /// A friendly name for the tensor. This is useful for debugging purposes.
            /// </summary>
            public string? name { get; set; }

            /// <summary>
            /// Finalize the tensor. Releases the tensor and its associated data.
            /// </summary>
            ~Tensor() => Dispose(false);

            public void Dispose()
            {
                OwningDisposeScope?.MarkAsDisposed(this);
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            /// <summary>
            /// Implements the .NET Dispose pattern.
            /// </summary>
            void Dispose(bool disposing)
            {
                if (handle != IntPtr.Zero) {
                    System.Threading.Interlocked.Decrement(ref _totalCount);
                    LibTorchSharp.THSTensor_dispose(handle);
                    handle = IntPtr.Zero;
                }
            }

            /// <summary>
            /// Is true if the tensor has been disposed, false otherwise.
            /// </summary>
            public bool IsInvalid => handle == IntPtr.Zero;

            /// <summary>
            /// Moves tensor to the outer DisposeScope. If there is no outer DisposeScope, it's detached from the
            /// DisposeScope system.
            /// </summary>
            /// <returns>The same tensor that the method was called on</returns>
            public torch.Tensor MoveToOuterDisposeScope()
            {
                OwningDisposeScope?.MoveToOuter(this);
                return this;
            }

            /// <summary>
            /// Detaches the tensor completely from the DisposeScope system.
            /// </summary>
            /// <returns>The same tensor that the method was called on</returns>
            public torch.Tensor DetachFromDisposeScope()
            {
                OwningDisposeScope?.Detach(this);
                return this;
            }

            /// <summary>
            /// Detaches the tensor completely from the DisposeScope system.
            /// </summary>
            /// <returns>The same tensor that the method was called on</returns>
            /// <remarks>
            /// This was a misspelling of 'Detach'. Keeping it to avoid making a
            /// breaking change, but it is deprecated and will be removed in a future
            /// release.
            /// </remarks>
            [Obsolete("The method name misspells 'Detach.' Use 'DetachFromDisposeScope' instead.", false)]
            public torch.Tensor DetatchFromDisposeScope() => DetachFromDisposeScope();

            /// <summary>
            /// Decouple the managed tensor from its underlying native tensor.
            ///
            /// This is primarily useful when returning a tensor to native code in a callback,
            /// or when having created a managed tensor from a passed-in native handle.
            ///
            /// See the torch.nn.Module.Module(string name) constructor for an example of its use.
            /// </summary>
            public IntPtr DecoupleFromNativeHandle()
            {
                GC.SuppressFinalize(this);

                if (handle == IntPtr.Zero)
                    throw new InvalidOperationException("Tensor invalid -- empty handle.");

                System.Threading.Interlocked.Decrement(ref _totalCount);
                var h = handle;
                handle = IntPtr.Zero;
                return h;
            }

            /// <summary>
            /// The total number of allocated tensors.
            /// </summary>
            /// <remarks>
            /// Only tensors that are realized in managed code will be counted, so tensors
            /// resulting from computations that remain in native code will not be counted
            /// in this property.
            ///
            /// Further, two tensors may alias each other, pointing at the same underlying data.
            ///
            /// Therefore, this property is mostly useful for diagnostic purposes, to
            /// make sure that there is no drift in tensor count from epoch to epoch,
            /// for example.
            /// </remarks>
            public static long TotalCount => _totalCount;

            /// <summary>
            /// The peak number of allocated tensors.
            /// </summary>
            /// <remarks>
            /// Only tensors that are realized in managed code will be counted, so tensors
            /// resulting from computations that remain in native code will not be counted
            /// in this property.
            ///
            /// Further, two tensors may alias each other, pointing at the same underlying data.
            ///
            /// Therefore, this property is mostly useful for diagnostic purposes.
            /// </remarks>
            public static long PeakCount => _peakCount;

            /// <summary>
            /// Get the handle for the tensor, validating that it's not null.
            /// </summary>
            /// <remarks>
            /// This property validates the handle. If you **aboslutely** need to get the handle without validation,
            /// use the 'handle' field.
            /// </remarks>
            public IntPtr Handle {
                get {
                    if (handle == IntPtr.Zero)
                        throw new InvalidOperationException("Tensor invalid -- empty handle.");
                    return handle;
                }
            }

            internal IntPtr MoveHandle()
            {
                var h = handle;
                handle = IntPtr.Zero;
                return h;
            }

            /// <summary>
            /// Returns the number of dimensions for this tensor
            /// </summary>
            public long Dimensions => LibTorchSharp.THSTensor_ndimension(Handle);

            /// <summary>
            /// Returns the number of dimensions for this tensor
            /// </summary>
            public long dim() => Dimensions;

            /// <summary>
            /// Returns the number of dimensions for this tensor
            /// </summary>
            public long ndim => Dimensions;

            /// <summary>
            /// Get the number of elements in the tensor.
            /// </summary>
            public long NumberOfElements => LibTorchSharp.THSTensor_numel(Handle);

            /// <summary>
            /// Get the number of elements in the tensor.
            /// </summary>
            public long numel() => NumberOfElements;

            /// <summary>
            /// Get the size of each element in the tensor.
            /// </summary>
            public long ElementSize => LibTorchSharp.THSTensor_element_size(Handle);

            public long element_size() => LibTorchSharp.THSTensor_element_size(Handle);

            public bool is_integral() => torch.is_integral(dtype);

            /// <summary>
            /// Returns True if the data type of input is a floating point data type.
            /// </summary>
            public bool is_floating_point() => torch.is_floating_point(dtype);

            /// <summary>
            /// Returns True if the data type of input is a complex data type i.e., one of torch.complex64, and torch.complex128.
            /// </summary>
            public bool is_complex() => torch.is_complex(dtype);

            /// <summary>
            /// Returns True if the input is a single element tensor which is not equal to zero after type conversions,
            /// i.e. not equal to torch.tensor([0.]) or torch.tensor([0]) or torch.tensor([False]).
            /// Throws an InvalidOperationException if torch.numel() != 1.
            /// </summary>
            public bool is_nonzero()
            {
                if (numel() != 1)
                    throw new InvalidOperationException("is_nonzero() called on non-singleton tensor");
                var res = LibTorchSharp.THSTensor_is_nonzero(Handle);
                CheckForErrors();
                return res != 0;
            }

            public bool is_cuda => device.type == DeviceType.CUDA;

            public bool is_meta => device.type == DeviceType.META;

            /// <summary>
            /// All Tensors that have requires_grad which is true will be leaf Tensors by convention.
            /// For Tensors that have requires_grad which is true, they will be leaf Tensors if they were created by the user.This means that they are not the result of an operation and so grad_fn is None.
            /// Only leaf Tensors will have their grad populated during a call to backward(). To get grad populated for non-leaf Tensors, you can use retain_grad().
            /// </summary>
            public bool is_leaf { get => LibTorchSharp.THSTensor_is_leaf(Handle) != 0; }


            /// <summary>
            /// Create a new reference to the same underlying native tensor.
            /// </summary>
            /// <returns>A fresh reference to the underlying native tensor.</returns>
            /// <remkars>
            /// This is useful for function implementations where a caller may expect the input and output to be
            /// distinct; in such situations, there's a risk that the tensor is disposed twice, with bad consequences.
            /// With 'alias(),' the reference count to the underlying native tensor is increased, meaning that the
            /// input and output can (and should) be disposed or finalized independently of each other.
            /// </remkars>
            public Tensor alias()
            {
                var res = LibTorchSharp.THSTensor_alias(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the underlying storage.
            /// </summary>
            /// <returns></returns>
            public Storage<T> storage<T>() where T : unmanaged
            {
                return Storage.Create<T>(this);
            }

            /// <summary>
            /// Returns the tensor’s offset in the underlying storage in terms of number of storage elements (not bytes).
            /// </summary>
            /// <returns></returns>
            public long storage_offset()
            {
                var res = LibTorchSharp.THSTensor_storage_offset(Handle);
                CheckForErrors();
                return res;
            }

            /// <summary>
            /// Returns a pointer to the unmanaged data managed by this tensor.
            /// </summary>
            public Utils.TensorAccessor<T> data<T>() where T : unmanaged
            {
                ValidateType(typeof(T));

                return device_type != DeviceType.CPU
                    ? new Utils.TensorAccessor<T>(cpu())
                    : new Utils.TensorAccessor<T>(this);
            }

            /// <summary>
            /// Returns the singleton value of a scalar tensor.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <returns>The scalar held in the tensor</returns>
            public T item<T>() where T : unmanaged
            {
                if (NumberOfElements != 1) throw new ArgumentException("Number of elements in the tensor must be 1");

                return data<T>()[0];
            }

            internal void ValidateType(Type dotnetType)
            {
                switch (dtype) {
                case ScalarType.Byte:
                    if (dotnetType != typeof(byte))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.Int8:
                    if (dotnetType != typeof(sbyte))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.Int16:
                    if (dotnetType != typeof(short))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.Int32:
                    if (dotnetType != typeof(int))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.Int64:
                    if (dotnetType != typeof(long))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.Bool:
                    if (dotnetType != typeof(bool))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.BFloat16:
                case ScalarType.Float16:
                case ScalarType.Float32:
                    if (dotnetType != typeof(float))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.Float64:
                    if (dotnetType != typeof(double))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.ComplexFloat32:
                    if (dotnetType != typeof((float, float)))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                case ScalarType.ComplexFloat64:
                    if (dotnetType != typeof(System.Numerics.Complex))
                        throw new ArgumentException($"{dotnetType.Name} is not compatible with {dtype.ToString()}");
                    break;
                }
            }

            /// <summary>
            /// Get or set the contents of a tensor as raw bytes.
            /// </summary>
            public Span<byte> bytes {
                get {
                    if (!is_contiguous()) throw new NotImplementedException("Bytes() called on non-contiguous tensor.");

                    long totalSize = NumberOfElements * ElementSize;

                    if (totalSize > int.MaxValue) {
                        throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                    }
                    if (device_type != DeviceType.CPU) {
                        throw new InvalidOperationException("Reading data from non-CPU memory is not supported. Move or copy the tensor to the cpu before reading.");
                    }

                    unsafe {
                        var res = LibTorchSharp.THSTensor_data(handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        // NOTE: there is no safety here.
                        return new Span<byte>((void*)res, (int)totalSize);
                    }
                }

                set {
                    if (!is_contiguous()) throw new NotImplementedException("SetBytes() called on non-contiguous tensor.");

                    long totalSize = NumberOfElements * ElementSize;
                    if (totalSize != value.Length) {
                        throw new ArgumentException("Mismatched data sizes in SetBytes().");
                    }

                    unsafe {
                        var res = LibTorchSharp.THSTensor_data(handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        // NOTE: there is no safety here.
                        var data = new Span<byte>((void*)res, value.Length);
                        value.CopyTo(data);
                    }
                }
            }

            public Tensor real {
                get {
                    var res = LibTorchSharp.THSTensor_real(Handle);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);

                }
            }

            public Tensor imag {
                get {
                    var res = LibTorchSharp.THSTensor_imag(Handle);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Read the double-precision value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public double ReadCpuDouble(long i) => Utils.TensorAccessor<double>.ReadItemAt(this, i);

            /// <summary>
            /// Read the single-precision float value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public float ReadCpuSingle(long i) => Utils.TensorAccessor<float>.ReadItemAt(this, i);

            /// <summary>
            /// Read the 32-bit integer float value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public int ReadCpuInt32(long i) => Utils.TensorAccessor<int>.ReadItemAt(this, i);

            /// <summary>
            /// Read the 64-bit integer value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public long ReadCpuInt64(long i) => Utils.TensorAccessor<long>.ReadItemAt(this, i);

            /// <summary>
            /// Read the byte value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public byte ReadCpuByte(long i) => Utils.TensorAccessor<byte>.ReadItemAt(this, i);

            /// <summary>
            /// Read the short value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public sbyte ReadCpuSByte(long i) => Utils.TensorAccessor<sbyte>.ReadItemAt(this, i);

            /// <summary>
            /// Read the int16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public short ReadCpuInt16(long i) => Utils.TensorAccessor<short>.ReadItemAt(this, i);

            /// <summary>
            /// Read the Boolean value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public bool ReadCpuBool(long i) => Utils.TensorAccessor<bool>.ReadItemAt(this, i);

            /// <summary>
            /// Read the value at the given index.
            /// </summary>
            /// <typeparam name="T">The type of the element to read.</typeparam>
            /// <param name="i">The index.</param>
            public T ReadCpuValue<T>(long i) where T : unmanaged => Utils.TensorAccessor<T>.ReadItemAt(this, i);

            /// <summary>
            /// Read the Float16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public float ReadCpuFloat16(long i)
            {
                if (i >= NumberOfElements) {
                    throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
                }
                return LibTorchSharp.THSTensor_data_idx_float16(handle, i);
            }

            /// <summary>
            /// Read the BFloat16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public float ReadCpuBFloat16(long i)
            {
                if (i >= NumberOfElements) {
                    throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
                }
                return LibTorchSharp.THSTensor_data_idx_bfloat16(handle, i);
            }

            /// <summary>
            /// Convert to a scalar.
            /// </summary>
            public Scalar ToScalar()
            {
                var res = LibTorchSharp.THSTensor_item(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Scalar(res);
            }

            /// <summary>
            /// Fill the tensor with the provided scalar value.
            /// </summary>
            /// <param name="value">A scalar value</param>
            public Tensor fill_(Scalar value)
            {
                var res = LibTorchSharp.THSTensor_fill_(handle, value is null ? IntPtr.Zero : value.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Gets the type of the tensor elements.
            /// </summary>
            public ScalarType dtype => (ScalarType)LibTorchSharp.THSTensor_type(Handle);

            /// <summary>
            /// Gets a string representing the device where the tensor is stored.
            /// </summary>
            public torch.Device device {
                get {
                    var dev_type = device_type;
                    if (dev_type == DeviceType.CPU) {
                        return new torch.Device(DeviceType.CPU);
                    } else {
                        return new torch.Device(dev_type, device_index);
                    }
                }
            }

            /// <summary>
            /// Gets a index of the device where the tensor is stored.
            /// </summary>
            public int device_index {
                get {
                    var res = LibTorchSharp.THSTensor_device_index(Handle);
                    CheckForErrors();
                    return res;
                }
            }

            /// <summary>
            /// Gets the type ('CPU', 'CUDA', etc.) of the device where the tensor is stored.
            /// </summary>
            public DeviceType device_type {
                get {
                    var res = LibTorchSharp.THSTensor_device_type(Handle);
                    CheckForErrors();
                    return (DeviceType)res;
                }
            }

            /// <summary>
            /// Is the tensor a sparse tensor?
            /// </summary>
            public bool is_sparse {
                get {
                    var res = LibTorchSharp.THSTensor_is_sparse(Handle);
                    CheckForErrors();
                    return res;
                }
            }

            public void backward(IList<Tensor>? grad_tensors = null, bool create_graph = false, bool retain_graph = false, IList<Tensor>? inputs = null) =>
                torch.autograd.backward(new[] { this }, grad_tensors, create_graph, retain_graph, inputs);

            /// <summary>
            /// Creates a tensor by loading it from a file.
            /// </summary>
            /// <param name="location">The file path where tensor values are stored.</param>
            public static Tensor load(string location)
            {
                var res = LibTorchSharp.THSTensor_load(location);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Save the contents of a tensor to a file.
            /// </summary>
            /// <param name="location">The file path where tensor values are to be stored.</param>
            public void save(string location)
            {
                LibTorchSharp.THSTensor_save(Handle, location);
                CheckForErrors();
            }

            /// <summary>
            /// Is the tensor tracking gradients?
            /// </summary>
            /// <remarks>Typically, gradients are tracked when the tensor is used as parameters of a module.</remarks>
            public bool requires_grad {
                get { return LibTorchSharp.THSTensor_requires_grad(Handle); }
                set {
                    var res = LibTorchSharp.THSTensor_set_requires_grad(Handle, value);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                }
            }

            public Tensor requires_grad_(bool requires_grad = true)
            {
                this.requires_grad = requires_grad;
                return this;
            }

            /// <summary>
            /// Enables this Tensor to have their grad populated during backward(). This is a no-op for leaf tensors.
            /// </summary>
            public void retain_grad()
            {
                LibTorchSharp.THSTensor_retain_grad(Handle);
                CheckForErrors();
            }

            /// <summary>
            /// Adds gradient tracking.
            /// </summary>
            public Tensor with_requires_grad(bool requires_grad = true)
            {
                this.requires_grad = requires_grad;
                return this;
            }

            /// <summary>
            /// Returns true if the tensor is on the CPU
            /// </summary>
            public bool is_cpu()
            {
                var res = LibTorchSharp.THSTensor_is_cpu(Handle);
                torch.CheckForErrors();
                return res;
            }

            /// <summary>
            /// Moves the tensor data to the CPU device
            /// </summary>
            public Tensor cpu()
            {
                var res = LibTorchSharp.THSTensor_cpu(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a copy of this object in CUDA memory.
            /// If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.
            /// </summary>
            public Tensor cuda(Device? device = null)
            {
                if (device is not null && device.type != DeviceType.CUDA) {
                    throw new ArgumentException("Not a CUDA device.", "device");
                }

                torch.InitializeDeviceType(DeviceType.CUDA);

                var res = device is null
                    ? LibTorchSharp.THSTensor_cuda(Handle)
                    : LibTorchSharp.THSTensor_to_device(Handle, (int)DeviceType.CUDA, device_index, false);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Cast the tensor to the given element type.
            /// </summary>
            /// <param name="type">The target type</param>
            /// <param name="copy">When copy is set, a new Tensor is created even when the Tensor already matches the desired conversion.</param>
            public Tensor to_type(ScalarType type, bool copy = false)
            {
                var res = LibTorchSharp.THSTensor_to_type(Handle, (sbyte)type, copy);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns this tensor cast to the type of the given tensor.
            /// </summary>
            public Tensor type_as(Tensor tensor) => to_type(tensor.dtype);

            /// <summary>
            /// Overwrite an existing tensor with the contents of another tensor.
            /// </summary>
            /// <param name="source">The source tensor</param>
            public Tensor set_(Tensor source)
            {
                var res = LibTorchSharp.THSTensor_set_(Handle, source.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Moves the tensor data to a specific device.
            /// </summary>
            /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="deviceIndex">The optional device index.</param>
            /// <param name="copy">When copy is set, a new Tensor is created even when the Tensor already matches the desired conversion.</param>
            public Tensor to(DeviceType deviceType, int deviceIndex = -1, bool copy = false)
            {
                torch.InitializeDeviceType(deviceType);
                var res = LibTorchSharp.THSTensor_to_device(Handle, (int)deviceType, deviceIndex, copy);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Moves the tensor data and casts it to the given element type.
            /// </summary>
            /// <param name="type">The target type</param>
            /// <param name="device">The target device</param>
            /// <param name="copy">When copy is set, a new Tensor is created even when the Tensor already matches the desired conversion.</param>
            public Tensor to(ScalarType type, torch.Device device, bool copy = false)
            {
                torch.InitializeDevice(device);
                var res = LibTorchSharp.THSTensor_to_type_and_device(Handle, (sbyte)type, (int)device.type, device.index, copy);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Cast the tensor to the given element type.
            /// </summary>
            /// <remarks>Alias for to_type</remarks>
            public Tensor to(ScalarType type) => to_type(type);

            /// <summary>
            /// Moves the tensor data.
            /// </summary>
            /// <param name="device">A string denoting the target device.</param>
            public Tensor to(string device) => to(new torch.Device(device));

            /// <summary>
            /// Moves the tensor data.
            /// </summary>
            /// <param name="device">The target device</param>
            public Tensor to(torch.Device device) => to(device.type, device.index);

            /// <summary>
            /// Moves the tensor data.
            /// </summary>
            /// <param name="other">The tensor serving as a template.</param>
            public Tensor to(Tensor other) => to(other.dtype, other.device);

            public Tensor type(Func<Tensor, Tensor> typeFunc) => typeFunc(this);

            public Tensor type(ScalarType dtype) => this.to(dtype);

            /// <summary>
            /// Retrieves the size of the specified dimension in the tensor.
            /// </summary>
            /// <param name="dim">The dimension for which to retrieve the size.</param>
            public long size(int dim)
            {
                var res = LibTorchSharp.THSTensor_size(Handle, dim);
                CheckForErrors();
                return res;
            }

            /// <summary>
            /// Retrieves the sizes of all dimensions of the tensor.
            /// </summary>
            public long[] size()
            {
                long[] ptrArray;

                using (var pa = new PinnedArray<long>()) {
                    LibTorchSharp.THSTensor_sizes(Handle, pa.CreateArray);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray;
            }

            /// <summary>
            /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor,
            /// and each element is the size of the dimension
            /// </summary>
            /// <remarks>
            /// An array of size 0 is used for constants, an array of size 1 is used
            /// for single-dimension arrays, where the dimension is the value of the
            /// first element. And so on.
            /// </remarks>
            public long[] shape {
                get {
                    return size();
                }
            }

            public bool has_names()
            {
                var res = LibTorchSharp.THSTensor_has_names(Handle);
                CheckForErrors();
                return res;
            }

            /// <summary>
            /// Stores names for each of this tensor’s dimensions.
            ///
            /// names[idx] corresponds to the name of tensor dimension idx.Names are either a string if the dimension is named or None if the dimension is unnamed.
            /// Dimension names may contain characters or underscore.Furthermore, a dimension name must be a valid Python variable name(i.e., does not start with underscore).
            /// Tensors may not have two named dimensions with the same name.
            /// </summary>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public string[] names {

                get {
                    // It should be safe to cache the names, since only rename_() can change them in place.
                    if (_names != null) return _names!;

                    if (!LibTorchSharp.THSTensor_has_names(Handle)) {
                        _names = new string[ndim];
                        return _names!;
                    }

                    using var sa = new PinnedArray<IntPtr>();
                    LibTorchSharp.THSTensor_names(Handle, sa.CreateArray);
                    CheckForErrors();
                    var strArray = sa.Array;

                    if (strArray == null) {
                        _names = new string[ndim];
                        return _names!;
                    }

                    _names = strArray.Select(str => { var s = Marshal.PtrToStringAnsi(str)!; return s == "*" ? null : s; }).ToArray();
                    return _names!;
                }
            }

            private string?[]? _names;

            private static IntPtr MarshalDimensionString(string? s)
            {
                return s == null ? Marshal.StringToHGlobalAnsi("*") : Marshal.StringToHGlobalAnsi(s);
            }

            /// <summary>
            /// Renames dimension names of the input tensor.
            /// </summary>
            /// <param name="names">A list of names, one for each dimension in the tensor. Passing 'null' clears out all the names.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor rename(IEnumerable<string?>? names)
            {
                if (names != null && names.Count() != ndim && !names.Contains("..."))
                    throw new ArgumentException($"The number of dimension names ({names.Count()}) is different from the number of dimensions ({ndim}).");

                IntPtr res = IntPtr.Zero;

                if (names != null && names.Count() > 0) {

                    var dimNamesArray = ExpandEllipsis(names);

                    IntPtr namesRef = IntPtr.Zero;

                    using PinnedArray<IntPtr> pinnedArray = new PinnedArray<IntPtr>();
                    namesRef = pinnedArray.CreateArray(dimNamesArray);

                    res = LibTorchSharp.THSTensor_rename(Handle, namesRef, names is null ? 0 : dimNamesArray.Length);
                } else {

                    res = LibTorchSharp.THSTensor_rename(Handle, IntPtr.Zero, 0);
                }

                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Renames dimension names of the input tensor, in place.
            /// </summary>
            /// <param name="names">A list of names, one for each dimension in the tensor. Passing 'null' clears out all the names.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor rename_(IEnumerable<string?>? names)
            {
                if (names != null && names.Count() != ndim && !names.Contains("..."))
                    throw new ArgumentException($"The number of dimension names ({names.Count()}) is different from the number of dimensions ({ndim}).");

                IntPtr res = IntPtr.Zero;

                if (names != null && names.Count() > 0) {

                    var dimNamesArray = ExpandEllipsis(names);

                    IntPtr namesRef = IntPtr.Zero;

                    using PinnedArray<IntPtr> pinnedArray = new PinnedArray<IntPtr>();
                    namesRef = pinnedArray.CreateArray(dimNamesArray);

                    res = LibTorchSharp.THSTensor_rename_(Handle, namesRef, names is null ? 0 : dimNamesArray.Length);
                } else {

                    res = LibTorchSharp.THSTensor_rename_(Handle, IntPtr.Zero, 0);
                }

                if (res == IntPtr.Zero) { CheckForErrors(); }
                // This is the only situation in which the names change in place.
                _names = null;
                return new Tensor(res);
            }

            /// <summary>
            /// Refines the dimension names of the input tensor according to names.
            ///
            /// Refining is a special case of renaming that “lifts” unnamed dimensions.A None dim can be refined to have any name; a named dim can only be refined to have the same name.
            /// Because named tensors can coexist with unnamed tensors, refining names gives a nice way to write named-tensor-aware code that works with both named and unnamed tensors.
            /// names may contain up to one ellipsis argument, passed as "...". The ellipsis is expanded greedily; it is expanded in-place to fill names to the same length as
            /// this.shape using names from the corresponding indices of this.names.
            /// </summary>
            /// <param name="names">A list of names, one for each dimension in the tensor.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor refine_names(IEnumerable<string> names)
            {
                var dimNamesArray = ExpandEllipsis(names);

                using PinnedArray<IntPtr> pinnedArray = new PinnedArray<IntPtr>();
                IntPtr namesRef = pinnedArray.CreateArray(dimNamesArray);

                IntPtr res = LibTorchSharp.THSTensor_refine_names(Handle, namesRef, dimNamesArray.Length);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            private IntPtr[] ExpandEllipsis(IEnumerable<string?> names)
            {
                var namesArrayLength = names.Count();

                IntPtr[] dimNamesArray = new IntPtr[ndim];

                if (names.Contains("...")) {
                    int idx = -1;
                    foreach (var name in names) {
                        idx += 1;
                        if (name == "...") break;
                        dimNamesArray[idx] = MarshalDimensionString(name);
                    }

                    if (idx == namesArrayLength - 1) { // '...' appears last.

                        int i = idx;

                        if (has_names()) {

                            var n = this.names;

                            for (; i < dimNamesArray.Length; i++)
                                dimNamesArray[i] = MarshalDimensionString(n[i]);

                        } else {
                            for (; i < dimNamesArray.Length; i++)
                                dimNamesArray[i] = IntPtr.Zero;
                        }

                    } else { // Names before and after.

                        int dIdx = idx;
                        int missing_dims = (int)(ndim - namesArrayLength + 1);

                        if (has_names()) {

                            var n = this.names;

                            for (int i = 0; i < missing_dims; i++, dIdx++)
                                dimNamesArray[dIdx] = MarshalDimensionString(n[dIdx]);

                        } else {
                            for (int i = 0; i < missing_dims; i++, dIdx++)
                                dimNamesArray[dIdx] = IntPtr.Zero;
                        }

                        foreach (var name in names.Skip(idx+1)) {
                            dimNamesArray[dIdx] = MarshalDimensionString(name);
                            dIdx++;
                        }
                    }

                } else {
                    int i = 0;
                    foreach (var name in names) {
                        dimNamesArray[i] = MarshalDimensionString(name);
                        i++;
                    }
                }

                return dimNamesArray;
            }

            /// <summary>
            /// Refines the dimension names of the input tensor according to names.
            ///
            /// Refining is a special case of renaming that “lifts” unnamed dimensions.A None dim can be refined to have any name; a named dim can only be refined to have the same name.
            /// Because named tensors can coexist with unnamed tensors, refining names gives a nice way to write named-tensor-aware code that works with both named and unnamed tensors.
            /// names may contain up to one ellipsis argument, passed as "...". The ellipsis is expanded greedily; it is expanded in-place to fill names to the same length as
            /// this.shape using names from the corresponding indices of this.names.
            /// </summary>
            /// <param name="names">A list of names, one for each dimension in the tensor.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor refine_names(params string[] names) => refine_names((IEnumerable<string>)names);

            /// <summary>
            /// Return the indices tensor of a sparse COO tensor.
            /// </summary>
            public Tensor SparseIndices {
                get {
                    var res = LibTorchSharp.THSTensor_indices(Handle);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Return the values tensor of a sparse COO tensor.
            /// </summary>
            public Tensor SparseValues {
                get {
                    var res = LibTorchSharp.THSTensor_values(Handle);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Generates a Vandermonde matrix.
            /// </summary>
            /// <param name="N">Number of columns in the output. If N is not specified, a square array is returned (N = len(x)).</param>
            /// <param name="increasing">
            /// Order of the powers of the columns.
            /// If true, the powers increase from left to right, if false (the default) they are reversed.
            /// </param>
            public Tensor vander(long N = -1, bool increasing = false)
            {
                if (this.Dimensions != 1) throw new InvalidOperationException("Input argument for 'vander()' must be 1-D.");

                var res = LibTorchSharp.THSTensor_vander(Handle, (N == -1) ? this.size(0) : N, increasing);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Retrieves the stride of all dimensions of the tensor.
            /// </summary>
            public long[] stride()
            {
                long[] ptrArray;

                using (var pa = new PinnedArray<long>()) {
                    LibTorchSharp.THSTensor_strides(Handle, pa.CreateArray);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray;
            }

            /// <summary>
            /// Retrieves the stride of the specified dimension in the tensor.
            /// </summary>
            public long stride(int dim)
            {
                var res = LibTorchSharp.THSTensor_stride(Handle, dim);
                CheckForErrors();
                return res;
            }

            /// <summary>
            /// Create a view of an existing torch.Tensor input with specified size, stride and storage offset.
            /// </summary>
            public Tensor as_strided(long[] size, long[] strides, long storageOffset = 0L)
            {
                unsafe {
                    fixed (long* psizes = size, pstrides = strides) {
                        var result = LibTorchSharp.THSTensor_as_strided(Handle, (IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, storageOffset);
                        if (result == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(result);
                    }
                }
            }

            /// <summary>
            /// Computes the gradient of current tensor w.r.t. graph leaves.
            /// </summary>
            public void backward()
            {
                LibTorchSharp.THSTensor_backward(Handle);
                CheckForErrors();
            }

            /// <summary>
            /// Creates a strided copy of the input tensor.
            /// </summary>
            public Tensor to_dense()
            {
                var res = LibTorchSharp.THSTensor_to_dense(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a copy of the tensor input.
            /// </summary>
            public Tensor clone()
            {
                var res = LibTorchSharp.THSTensor_clone(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Copies the elements from source into the tensor and returns it.
            /// </summary>
            /// <remarks>The src tensor must be broadcastable with the target 'this' tensor. It may be of a different data type or reside on a different device.</remarks>
            public Tensor copy_(Tensor source, bool nonBlocking = false)
            {
                var res = LibTorchSharp.THSTensor_copy_(Handle, source.Handle, nonBlocking);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns true if the tensor is contiguous.
            /// </summary>
            public bool is_contiguous()
            {
                var res = LibTorchSharp.THSTensor_is_contiguous(Handle);
                CheckForErrors();
                return res != 0;
            }

            /// <summary>
            /// Returns a contiguous in memory tensor containing the same data as the input tensor.
            /// If tensor is already in the specified memory format, this function returns the original tensor.
            /// </summary>
            public Tensor contiguous()
            {
                var res = LibTorchSharp.THSTensor_contiguous(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns true if the tensor is contiguous.
            /// </summary>
            public bool is_pinned()
            {
                var res = LibTorchSharp.THSTensor_is_pinned(Handle);
                CheckForErrors();
                return res != 0;
            }

            /// <summary>
            /// Returns a contiguous in memory tensor containing the same data as the input tensor.
            /// If tensor is already in the specified memory format, this function returns the original tensor.
            /// </summary>
            public Tensor pin_memory()
            {
                var res = LibTorchSharp.THSTensor_pin_memory(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// This attribute is null by default and becomes a Tensor the first time a call to backward() computes gradients for the tensor.
            /// The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.
            /// </summary>
            public Tensor? grad()
            {
                var res = LibTorchSharp.THSTensor_grad(Handle);
                CheckForErrors();

                if (res == IntPtr.Zero)
                    return null;

                return new Tensor(res);
            }

            internal void EncodeIndices(TensorIndex[] indices,
                out long[] arrKindAndStarts,
                out long[]? arrStops,
                out long[]? arrSteps,
                out IntPtr[]? arrTensors)
            {
                bool hasSliceEnd = false;
                bool hasSliceStep = false;
                bool hasTensor = false;
                var n = indices.Length;
                for (int i = 0; i < indices.Length; i++) {
                    var idx = indices[i];
                    if (idx.kind == TensorIndex.Kind.Slice && idx.stopIndex.HasValue)
                        hasSliceEnd = true;
                    if (idx.kind == TensorIndex.Kind.Slice && idx.step.HasValue)
                        hasSliceStep = true;
                    if (idx.kind == TensorIndex.Kind.Tensor && (object?)idx.tensor != null)
                        hasTensor = true;
                }
                arrStops = hasSliceEnd ? new long[n] : null;
                arrSteps = hasSliceStep ? new long[n] : null;
                arrTensors = hasTensor ? new IntPtr[n] : null;
                arrKindAndStarts = new long[n];
                for (int i = 0; i < indices.Length; i++) {
                    var idx = indices[i];
                    arrKindAndStarts[i] =
                        (idx.kind == TensorIndex.Kind.Null) ? long.MinValue :
                        (idx.kind == TensorIndex.Kind.Bool && idx.startIndexOrBoolOrSingle == 0) ? long.MinValue + 1 :
                        (idx.kind == TensorIndex.Kind.Bool && idx.startIndexOrBoolOrSingle != 0) ? long.MinValue + 2 :
                        (idx.kind == TensorIndex.Kind.Ellipsis) ? long.MinValue + 3 :
                        (idx.kind == TensorIndex.Kind.None) ? long.MinValue + 4 :
                        (idx.kind == TensorIndex.Kind.Tensor) ? long.MinValue + 5 :
                        (idx.kind == TensorIndex.Kind.Slice && !idx.startIndexOrBoolOrSingle.HasValue) ? long.MinValue + 6 :
                        (idx.kind == TensorIndex.Kind.Single) ? idx.startIndexOrBoolOrSingle.GetValueOrDefault() :
                        idx.startIndexOrBoolOrSingle.GetValueOrDefault() + long.MinValue / 2;
                    if (arrStops != null && idx.kind == TensorIndex.Kind.Slice)
                        arrStops[i] = (idx.stopIndex.HasValue ? idx.stopIndex.Value : long.MinValue);
                    if (arrSteps != null && idx.kind == TensorIndex.Kind.Slice)
                        arrSteps[i] = (idx.step.HasValue ? idx.step.Value : long.MinValue);
                    if (arrTensors != null && idx.kind == TensorIndex.Kind.Tensor)
                        arrTensors[i] = ((object?)idx.tensor == null ? IntPtr.Zero : idx.tensor.Handle);
                }

            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions.
            /// </summary>
            [IndexerName("TensorItems")]
            public Tensor this[params TensorIndex[] indices] {
                get { return index(indices); }
                set { index_put_(value, indices); }
            }

            [IndexerName("TensorItems")]
            public Tensor this[params Tensor[] indices] {
                get { return index(indices); }
                set { index_put_(value, indices); }
            }

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            [IndexerName("TensorItems")]
            public Tensor this[long i1] {
                get {
                    var res = LibTorchSharp.THSTensor_get1(Handle, i1);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    LibTorchSharp.THSTensor_set1(Handle, i1, value.Handle);
                    CheckForErrors();
                }
            }

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2] {
                get {
                    var res = LibTorchSharp.THSTensor_get2(Handle, i1, i2);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    LibTorchSharp.THSTensor_set2(Handle, i1, i2, value.Handle);
                    CheckForErrors();
                }
            }

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3] {
                get {
                    var res = LibTorchSharp.THSTensor_get3(Handle, i1, i2, i3);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    LibTorchSharp.THSTensor_set3(Handle, i1, i2, i3, value.Handle);
                    CheckForErrors();
                }
            }

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <param name="i4">The fourth-dimension index</param>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3, long i4] {
                get {
                    var res = LibTorchSharp.THSTensor_get4(Handle, i1, i2, i3, i4);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    LibTorchSharp.THSTensor_set4(Handle, i1, i2, i3, i4, value.Handle);
                    CheckForErrors();
                }
            }

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <param name="i4">The fourth-dimension index</param>
            /// <param name="i5">The fifth-dimension index</param>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3, long i4, long i5] {
                get {
                    var res = LibTorchSharp.THSTensor_get5(Handle, i1, i2, i3, i4, i5);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    LibTorchSharp.THSTensor_set5(Handle, i1, i2, i3, i4, i5, value.Handle);
                    CheckForErrors();
                }
            }


            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <param name="i4">The fourth-dimension index</param>
            /// <param name="i5">The fifth-dimension index</param>
            /// <param name="i6">The sixth-dimension index</param>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3, long i4, long i5, long i6] {
                get {
                    var res = LibTorchSharp.THSTensor_get6(Handle, i1, i2, i3, i4, i5, i6);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    LibTorchSharp.THSTensor_set6(Handle, i1, i2, i3, i4, i5, i6, value.Handle);
                    CheckForErrors();
                }
            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions.
            /// </summary>
            public Tensor index(params TensorIndex[] indices)
            {
                EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
                unsafe {
                    fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                        fixed (IntPtr* ptrTensors = arrTensors) {
                            var res = LibTorchSharp.THSTensor_index(Handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length);
                            if (res == IntPtr.Zero)
                                CheckForErrors();
                            GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                            return new Tensor(res);
                        }
                    }
                }

            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions.
            /// </summary>
            public Tensor index(params Tensor[] indices)
            {
                return index(indices.Select(t => TensorIndex.Tensor(t)).ToArray());
            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions and place a tensor at the index.
            /// </summary>
            public Tensor index_put_(Tensor value, params TensorIndex[] indices)
            {
                EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
                unsafe {
                    fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                        fixed (IntPtr* ptrTensors = arrTensors) {
                            var res = LibTorchSharp.THSTensor_index_put_(Handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
                            if (res == IntPtr.Zero)
                                CheckForErrors();
                            GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                            GC.KeepAlive(value);
                            return new Tensor(res);
                        }
                    }
                }
            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions and place a tensor at the index.
            /// </summary>
            public Tensor index_put_(Tensor value, params Tensor[] indices)
            {
                return index_put_(value, indices.Select(t => TensorIndex.Tensor(t)).ToArray());
            }


            /// <summary>
            /// Index into the tensor using Python-like indexing expressions and place a scalar tensor at the index.
            /// </summary>
            public Tensor index_put_(Scalar value, params TensorIndex[] indices)
            {
                EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
                unsafe {
                    fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                        fixed (IntPtr* ptrTensors = arrTensors) {
                            var res = LibTorchSharp.THSTensor_index_put_scalar_(Handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
                            if (res == IntPtr.Zero)
                                CheckForErrors();
                            GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                            GC.KeepAlive(value);
                            return new Tensor(res);
                        }
                    }
                }
            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions and place a scalar tensor at the index.
            /// </summary>
            public Tensor index_put_(Scalar value, params Tensor[] indices)
            {
                return index_put_(value, indices.Select(t => TensorIndex.Tensor(t)).ToArray());
            }

            /// <summary>
            /// Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
            /// </summary>
            /// <param name="dim">The dimension in which we index</param>
            /// <param name="index">The 1-D tensor containing the indices to index</param>
            public Tensor index_select(long dim, Tensor index)
            {
                var res = LibTorchSharp.THSTensor_index_select(Handle, dim, index.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Slices the input tensor along the selected dimension at the given index.
            /// This function returns a view of the original tensor with the given dimension removed.
            /// </summary>
            /// <param name="dim">The dimension to slice</param>
            /// <param name="index">The index to select with</param>
            public Tensor select(long dim, long index)
            {
                var res = LibTorchSharp.THSTensor_select(Handle, dim, index);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the elements of input at the given indices. The input tensor is treated as if it were viewed as a 1-D tensor.
            /// The result takes the same shape as the indices.
            /// </summary>
            /// <param name="index">The indices into tensor, an Int64 tensor.</param>
            public Tensor take(Tensor index)
            {
                var res = LibTorchSharp.THSTensor_take(Handle, index.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor containing the indices of all non-zero elements of input.
            /// Each row in the result contains the indices of a non-zero element in input.
            /// The result is sorted lexicographically, with the last index changing the fastest (C-style).
            /// If input has n dimensions, then the resulting indices tensor out is of size (z×n), where
            /// z is the total number of non-zero elements in the input tensor.
            /// </summary>
            public Tensor argwhere()
            {
                var res = LibTorchSharp.THSTensor_argwhere(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(Tensor indices)
            {
                var res = LibTorchSharp.THSTensor_take_along_dim_dflt(Handle, indices.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(IEnumerable<long> indices) => take_along_dim(torch.tensor(indices.ToArray()));

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <param name="dim">Dimension to select along.</param>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(Tensor indices, long dim)
            {
                var res = LibTorchSharp.THSTensor_take_along_dim(Handle, indices.Handle, dim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <param name="dim">Dimension to select along.</param>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(IEnumerable<long> indices, long dim) => take_along_dim(torch.tensor(indices.ToArray()), dim);

            /// <summary>
            /// Accumulate the elements of alpha times source into the input tensor by adding to the indices in the order given in index.
            ///
            /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
            /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match the input tensor, or an error will be raised.
            /// </summary>
            /// <param name="dim">Dimension along which to index</param>
            /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
            /// <param name="source">The tensor containing values to add</param>
            /// <param name="alpha">The scalar multiplier for source</param>
            /// <returns></returns>
            public Tensor index_add(long dim, Tensor index, Tensor source, Scalar alpha)
            {
                if (index.dtype != ScalarType.Int64)
                    throw new ArgumentException("Element type of 'index' must be 'Int64'");
                var res = LibTorchSharp.THSTensor_index_add(Handle, dim, index.Handle, source.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Accumulate, in place, the elements of alpha times source into the input tensor by adding to the indices in the order given in index.
            ///
            /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
            /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match the input tensor, or an error will be raised.
            /// </summary>
            /// <param name="dim">Dimension along which to index</param>
            /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
            /// <param name="source">The tensor containing values to add</param>
            /// <param name="alpha">The scalar multiplier for source</param>
            /// <returns></returns>
            public Tensor index_add_(long dim, Tensor index, Tensor source, Scalar alpha)
            {
                if (index.dtype != ScalarType.Int64)
                    throw new ArgumentException("Element type of 'index' must be 'Int64'");
                var res = LibTorchSharp.THSTensor_index_add_(Handle, dim, index.Handle, source.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Copies the elements of the source tensor into the input tensor by selecting the indices in the order given in index.
            ///
            /// For example, if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of the input tensor.
            /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match the input tensor, or an error will be raised.
            /// </summary>
            /// <param name="dim">Dimension along which to index</param>
            /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
            /// <param name="source">The tensor containing values to copy</param>
            /// <returns></returns>
            public Tensor index_copy(long dim, Tensor index, Tensor source)
            {
                if (index.dtype != ScalarType.Int64)
                    throw new ArgumentException("Element type of 'index' must be 'Int64'");
                var res = LibTorchSharp.THSTensor_index_copy(Handle, dim, index.Handle, source.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Copies, in place, the elements of the source tensor into the input tensor by selecting the indices in the order given in index.
            ///
            /// For example, if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of the input tensor.
            /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match the input tensor, or an error will be raised.
            /// </summary>
            /// <param name="dim">Dimension along which to index</param>
            /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
            /// <param name="source">The tensor containing values to copy</param>
            /// <returns></returns>
            public Tensor index_copy_(long dim, Tensor index, Tensor source)
            {
                if (index.dtype != ScalarType.Int64)
                    throw new ArgumentException("Element type of 'index' must be 'Int64'");
                var res = LibTorchSharp.THSTensor_index_copy_(Handle, dim, index.Handle, source.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the elements of the input tensor with value value by selecting the indices in the order given in index.
            ///
            /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
            /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match the input tensor, or an error will be raised.
            /// </summary>
            /// <param name="dim">Dimension along which to index</param>
            /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
            /// <param name="value">The scalar multiplier for source</param>
            /// <returns></returns>
            public Tensor index_fill(long dim, Tensor index, Scalar value)
            {
                if (index.dtype != ScalarType.Int64)
                    throw new ArgumentException("Element type of 'index' must be 'Int64'");
                var res = LibTorchSharp.THSTensor_index_fill(Handle, dim, index.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Fills, in place, the elements of the input tensor with value value by selecting the indices in the order given in index.
            ///
            /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
            /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match the input tensor, or an error will be raised.
            /// </summary>
            /// <param name="dim">Dimension along which to index</param>
            /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
            /// <param name="value">The scalar multiplier for source</param>
            /// <returns></returns>
            public Tensor index_fill_(long dim, Tensor index, Scalar value)
            {
                if (index.dtype != ScalarType.Int64)
                    throw new ArgumentException("Element type of 'index' must be 'Int64'");
                var res = LibTorchSharp.THSTensor_index_fill_(Handle, dim, index.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor with the same data and number of elements as the input tensor but with the specified shape.
            /// </summary>
            /// <param name="shape">The new tensor shape.</param>
            public Tensor reshape(params long[] shape)
            {
                unsafe {
                    fixed (long* pshape = shape) {
                        var res = LibTorchSharp.THSTensor_reshape(Handle, (IntPtr)pshape, shape.Length);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Flattens input by reshaping it into a one-dimensional tensor.
            /// </summary>
            /// <param name="start_dim">The first dim to flatten</param>
            /// <param name="end_dim">The last dim to flatten.</param>
            /// <remarks>Flattening a zero-dimensional tensor will return a one-dimensional view.</remarks>
            public Tensor flatten(long start_dim = 0, long end_dim = -1)
            {
                var res = LibTorchSharp.THSTensor_flatten(Handle, start_dim, end_dim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Flattens dims into a single dimension with name out_dim.
            /// </summary>
            /// <param name="dims">The named dimensions to flatten.</param>
            /// <param name="out_dim">The name of the output dimension.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor flatten(IList<string?> dims, string out_dim)
            {
                if (dims.Count == 0)
                    throw new ArgumentException($"The number of input dimension names ({dims.Count}) is zero.");
                if (dims.Count > ndim)
                    throw new ArgumentException($"The number of input dimension names ({dims.Count}) is larger than the number of dimensions ({ndim}).");

                using PinnedArray<IntPtr> pinnedArray = new PinnedArray<IntPtr>();
                var iPtrArray = dims.Select(s => Marshal.StringToHGlobalAnsi(s)).ToList();
                iPtrArray.Add(Marshal.StringToHGlobalAnsi(out_dim));

                IntPtr namesRef = pinnedArray.CreateArray(iPtrArray.ToArray());

                IntPtr res = LibTorchSharp.THSTensor_flatten_names(Handle, namesRef, iPtrArray.Count);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Expands the dimension dim of the input tensor over multiple dimensions of sizes given by sizes.
            /// </summary>
            /// <param name="dim">Dimension to unflatten.</param>
            /// <param name="sizes">New shape of the unflattened dimension.</param>
            public Tensor unflatten(long dim, params long[] sizes)
            {
                if (dim < 0 || dim >= ndim)
                    throw new ArgumentException($"Invalid 'dim' argument: {dim}.");
                if (sizes.Length == 0)
                    throw new ArgumentException($"The number of output sizes is zero.");

                unsafe {
                    fixed (long* pshape = sizes) {
                        var res = LibTorchSharp.THSTensor_unflatten(Handle, dim, (IntPtr)pshape, sizes.Length);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Expands the dimension dim of the input tensor over multiple dimensions of sizes given by sizes.
            /// </summary>
            /// <param name="dim">Dimension to unflatten.</param>
            /// <param name="sizes">New names and sizes of the unflattened dimension.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor unflatten(string dim, params (string, long)[] sizes)
            {
                if (sizes.Length == 0)
                    throw new ArgumentException($"The number of output sizes is zero.");

                var names = sizes.Select(s => s.Item1).ToList();
                var szs = sizes.Select(s => s.Item2).ToArray();

                names.Insert(0, dim);

                using PinnedArray<IntPtr> pinnedArray = new PinnedArray<IntPtr>();

                IntPtr namesRef = pinnedArray.CreateArray(names.Select(s => Marshal.StringToHGlobalAnsi(s)).ToArray());

                unsafe {
                    fixed (long* pshape = szs) {
                        var res = LibTorchSharp.THSTensor_unflatten_names(Handle, namesRef, (IntPtr)pshape, names.Count);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Permutes the dimensions of the input tensor to match the order specified in names, adding size-one dims for any new names.
            ///
            /// All of the dims of the input tensor must be named in order to use this method.The resulting tensor is a view on the original tensor.
            /// All dimension names of the input tensor must be present in names.names may contain additional names that are not in this.names; the output tensor has a size-one dimension for each of those new names.
            /// names may contain up to one ellipsis "...". The ellipsis is expanded to be equal to all dimension names of the input tensor that are not mentioned in names, in the order that they appear in the input tensor.
            /// </summary>
            /// <param name="names">The desired dimension ordering of the output tensor. May contain up to one ellipsis that is expanded to all unmentioned dim names of the input tensor.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor align_to(IEnumerable<string> names)
            {
                using PinnedArray<IntPtr> pinnedArray = new PinnedArray<IntPtr>();
                IntPtr namesRef = pinnedArray.CreateArray(names.Select(s => Marshal.StringToHGlobalAnsi(s)).ToArray());

                IntPtr res = LibTorchSharp.THSTensor_align_to(Handle, namesRef, names.Count());
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Permutes the dimensions of the input tensor to match the order specified in names, adding size-one dims for any new names.
            ///
            /// All of the dims of the input tensor must be named in order to use this method.The resulting tensor is a view on the original tensor.
            /// All dimension names of the input tensor must be present in names.names may contain additional names that are not in this.names; the output tensor has a size-one dimension for each of those new names.
            /// names may contain up to one ellipsis "...". The ellipsis is expanded to be equal to all dimension names of the input tensor that are not mentioned in names, in the order that they appear in the input tensor.
            /// </summary>
            /// <param name="names">The desired dimension ordering of the output tensor. May contain up to one ellipsis that is expanded to all unmentioned dim names of the input tensor.</param>
            /// <remarks>The named tensor API is experimental and subject to change.</remarks>
            public Tensor align_to(params string[] names) => align_to((IEnumerable<string>)names);

            /// <summary>
            /// Permutes the dimensions of the input tensor to match the dimension order in the other tensor, adding size-one dims for any new names.
            ///
            /// This operation is useful for explicit broadcasting by names.
            /// All of the dims of the input tensor must be named in order to use this method.The resulting tensor is a view on the original tensor.
            /// All dimension names of the input tensor must be present in other.names.other may contain named dimensions that are not in this.names; the output tensor has a size-one dimension for each of those new names.
            /// To align a tensor to a specific order, use align_to().
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor align_as(Tensor other) => align_to(other.names!);

            /// <summary>
            /// Returns the unique elements of the input tensor.
            /// </summary>
            /// <param name="sorted">Whether to sort the unique elements in ascending order before returning as output.</param>
            /// <param name="return_inverse">Whether to also return the indices for where elements in the original input ended up in the returned unique list.</param>
            /// <param name="return_counts">Whether to also return the counts for each unique element.</param>
            /// <param name="dim">The dimension to apply unique. If null, the unique of the flattened input is returned.</param>
            /// <returns></returns>
            /// <remarks>This function is different from torch.unique_consecutive() in the sense that this function also eliminates non-consecutive duplicate values.</remarks>
            public (Tensor output, Tensor? inverse_indices, Tensor? counts) unique(bool sorted = true, bool return_inverse = false, bool return_counts = false, int? dim = null)
            {
                IntPtr res = IntPtr.Zero;
                IntPtr inverse_indices, counts;

                if (dim is null) {
                    res = LibTorchSharp.THSTensor_unique(Handle, sorted, return_inverse, return_counts, out inverse_indices, out counts);
                } else {
                    res = LibTorchSharp.THSTensor_unique_dim(Handle, dim.Value, sorted, return_inverse, return_counts, out inverse_indices, out counts);
                }

                if (res == IntPtr.Zero)
                    CheckForErrors();
                return (new Tensor(res), inverse_indices != IntPtr.Zero ? new Tensor(inverse_indices) : null, counts != IntPtr.Zero ? new Tensor(counts) : null);
            }

            /// <summary>
            /// Returns the unique elements of the input tensor.
            /// </summary>
            /// <param name="return_inverse">Whether to also return the indices for where elements in the original input ended up in the returned unique list.</param>
            /// <param name="return_counts">Whether to also return the counts for each unique element.</param>
            /// <param name="dim">The dimension to apply unique. If null, the unique of the flattened input is returned.</param>
            /// <returns>A tuple with the output, the indices, and counts. The latter two may be 'null'</returns>
            /// <remarks>This function is different from torch.unique_consecutive() in the sense that this function also eliminates non-consecutive duplicate values.</remarks>
            public (Tensor output, Tensor? inverse_indices, Tensor? counts) unique_consecutive(bool return_inverse = false, bool return_counts = false, int? dim = null)
            {
                IntPtr inverse_indices, counts;

                IntPtr res = (dim is null)
                    ? LibTorchSharp.THSTensor_unique_consecutive(Handle, return_inverse, return_counts, out inverse_indices, out counts)
                    : LibTorchSharp.THSTensor_unique_dim_consecutive(Handle, dim.Value, return_inverse, return_counts, out inverse_indices, out counts);

                if (res == IntPtr.Zero)
                    CheckForErrors();
                return (new Tensor(res), inverse_indices != IntPtr.Zero ? new Tensor(inverse_indices) : null, counts != IntPtr.Zero ? new Tensor(counts) : null);
            }

            /// <summary>
            /// Expands the dimension dim of the input tensor over multiple dimensions of sizes given by sizes.
            /// </summary>
            /// <param name="dim">Dimension to unflatten.</param>
            /// <param name="sizes">New shape of the unflattened dimension.</param>
            public Tensor unflatten(long dim, torch.Size sizes)
            {
                return unflatten(dim, sizes.Shape);
            }

            /// <summary>
            /// Returns a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
            /// </summary>
            /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
            public Tensor squeeze(long? dim = null)
            {
                var res = dim.HasValue ? LibTorchSharp.THSTensor_squeeze(Handle, dim.Value) : LibTorchSharp.THSTensor_squeeze_no_dim(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Modify (in-palce) a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
            /// </summary>
            /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
            public Tensor squeeze_(long? dim = null)
            {
                var res = dim.HasValue ? LibTorchSharp.THSTensor_squeeze_(Handle, dim.Value) : LibTorchSharp.THSTensor_squeeze_no_dim_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Expects input to be 1- or 2-D tensor and transposes dimensions 0 and 1.
            /// </summary>
            public Tensor t()
            {
                var res = LibTorchSharp.THSTensor_t(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Is this Tensor with its dimensions reversed.
            /// </summary>
            /// <remarks>
            /// Starting with Pytorch 1.11, 'T' should not be used for tensors that do not represents matrices:
            /// https://github.com/pytorch/pytorch/pull/64180
            /// </remarks>
            public Tensor T {
                get {
                    return this.permute(Enumerable.Range(0, (int)ndim).Reverse().Select(i => (long)i).ToArray());
                }
            }

            /// <summary>
            /// Is this Tensor with its dimensions reversed.
            /// </summary>
            public Tensor H {
                get {
                    return is_complex() ? transpose(0, 1).conj() : transpose(0, 1);
                }
            }

            /// <summary>
            /// Returns a view of this tensor with the last two dimensions transposed.
            /// </summary>
            public Tensor mT {
                get {
                    var res = LibTorchSharp.THSTensor_mT(Handle);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Accessing this property is equivalent to calling adjoint().
            /// </summary>
            public Tensor mH {
                get {
                    var res = LibTorchSharp.THSTensor_mH(Handle);
                    if (res == IntPtr.Zero)
                        CheckForErrors();
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// </summary>
            /// <param name="dim0"></param>
            /// <param name="dim1"></param>
            public Tensor transpose(long dim0, long dim1)
            {
                var res = LibTorchSharp.THSTensor_transpose(Handle, dim0, dim1);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a view of the tensor conjugated and with the last two dimensions transposed.
            /// </summary>
            public Tensor adjoint()
            {
                var res = LibTorchSharp.THSTensor_adjoint(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
            /// The lower triangular part of the matrix is defined as the elements on and below the diagonal.
            /// </summary>
            /// <param name="diagonal">The diagonal to consider</param>
            public Tensor tril(long diagonal = 0)
            {
                var res = LibTorchSharp.THSTensor_tril(Handle, diagonal);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
            /// The upper triangular part of the matrix is defined as the elements on and above the diagonal.
            /// </summary>
            /// <param name="diagonal">The diagonal to consider</param>
            public Tensor triu(long diagonal = 0)
            {
                var res = LibTorchSharp.THSTensor_triu(Handle, diagonal);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// </summary>
            public Tensor swapdims(long dim0, long dim1) => transpose(dim0, dim1);

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// </summary>
            public Tensor swapaxes(long dim0, long dim1) => transpose(dim0, dim1);

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// Inplace version of transpose()
            /// </summary>
            /// <param name="dim0"></param>
            /// <param name="dim1"></param>
            public Tensor transpose_(long dim0, long dim1)
            {
                var res = LibTorchSharp.THSTensor_transpose_(Handle, dim0, dim1);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the same data as the input tensor but of a different shape.
            /// </summary>
            /// <param name="shape">The shape of the view</param>
            public Tensor view(params long[] shape)
            {
                unsafe {
                    fixed (long* pshape = shape) {
                        var res = LibTorchSharp.THSTensor_view(Handle, (IntPtr)pshape, shape.Length);
                        if (res == IntPtr.Zero)
                            CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// View this tensor as the same size as other.
            /// </summary>
            /// <param name="other">The result tensor has the same size as other.</param>
            /// <remarks>
            /// this.view_as(other) is equivalent to this.view(other.size()).
            /// Please see view() for more information about view.
            /// </remarks>
            public Tensor view_as(Tensor other)
            {
                return view(other.shape);
            }

            /// <summary>
            /// Returns a view of input as a complex tensor.
            /// </summary>
            public Tensor view_as_complex()
            {
                var result = LibTorchSharp.THSTensor_view_as_complex(Handle);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(result);
            }

            /// <summary>
            /// Returns a view of input as a real tensor.
            /// </summary>
            public Tensor view_as_real()
            {
                var result = LibTorchSharp.THSTensor_view_as_real(Handle);
                if (result == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(result);
            }

            /// <summary>
            /// Tests if all elements in input evaluate to true.
            /// </summary>
            public Tensor all()
            {
                var res = LibTorchSharp.THSTensor_all(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Tests if all elements in input evaluate to true.
            /// </summary>
            /// <param name="dim">The dimension to reduce</param>
            /// <param name="keepdim">Keep the dimension to reduce</param>
            public Tensor all(long dim, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_all_along_dimension(Handle, dim, keepdim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            /// <param name="out">The output tensor -- optional.</param>
            public Tensor amax(long[] dims, bool keepdim = false, Tensor? @out = null) => amax((ReadOnlySpan<long>)dims, keepdim, @out);

            /// <summary>
            /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            public Tensor amax(params long[] dims) => amax((ReadOnlySpan<long>)dims, false, null);

            /// <summary>
            /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            /// <param name="out">The output tensor -- optional.</param>
            public Tensor amax(ReadOnlySpan<long> dims, bool keepdim = false, Tensor? @out = null)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = @out is null ?
                            LibTorchSharp.THSTensor_amax(Handle, (IntPtr)pdims, dims.Length, keepdim) :
                            LibTorchSharp.THSTensor_amax_out(Handle, (IntPtr)pdims, dims.Length, keepdim, @out.Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            /// <param name="out">The output tensor -- optional.</param>
            public Tensor amin(ReadOnlySpan<long> dims, bool keepdim = false, Tensor? @out = null)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = @out is null ?
                            LibTorchSharp.THSTensor_amin(Handle, (IntPtr)pdims, dims.Length, keepdim) :
                            LibTorchSharp.THSTensor_amin_out(Handle, (IntPtr)pdims, dims.Length, keepdim, @out.Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            /// <param name="out">The output tensor -- optional.</param>
            public Tensor amin(long[] dims, bool keepdim = false, Tensor? @out = null) => amin((ReadOnlySpan<long>)dims, keepdim, @out);

            /// <summary>
            /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            public Tensor amin(params long[] dims) => amin((ReadOnlySpan<long>)dims, false, null);

            /// <summary>
            /// Computes the minimum and maximum values of the input tensor.
            /// </summary>
            /// <param name="dim">The dimension along which to compute the values. If null, computes the values over the entire input tensor</param>
            /// <param name="keepdim"> If true, the reduced dimensions will be kept in the output tensor as dimensions with size 1 for broadcasting.</param>
            public (Tensor min, Tensor max) aminmax(long? dim = null, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_aminmax(Handle, (dim is null) ? -1 : dim.Value, keepdim, out IntPtr maxHandle);
                if (res == IntPtr.Zero || maxHandle == IntPtr.Zero) { CheckForErrors(); }
                return (new Tensor(res), new Tensor(maxHandle));
            }

            /// <summary>
            /// Tests if any element in input evaluate to true.
            /// </summary>
            public Tensor any()
            {
                var res = LibTorchSharp.THSTensor_any(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Tests if any element in input evaluate to true.
            /// </summary>
            /// <param name="dim">The dimension to reduce</param>
            /// <param name="keepdim">Keep the dimension to reduce</param>
            public Tensor any(long dim, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_any_along_dimension(Handle, dim, keepdim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the indices of the maximum value of all elements in the input tensor.
            /// </summary>
            public Tensor argmax()
            {
                var res = LibTorchSharp.THSTensor_argmax(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the indices of the maximum value of all elements in the input tensor.
            /// </summary>
            /// <param name="dim"></param>
            /// <param name="keepdim"></param>
            public Tensor argmax(long dim, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_argmax_along_dimension(Handle, dim, keepdim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the indices of the minimum value of all elements in the input tensor.
            /// </summary>
            public Tensor argmin()
            {
                var res = LibTorchSharp.THSTensor_argmin(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the indices of the minimum value of all elements in the input tensor.
            /// </summary>
            /// <param name="dim"></param>
            /// <param name="keepdim"></param>
            public Tensor argmin(long dim, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_argmin_along_dimension(Handle, dim, keepdim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the indices that sort a tensor along a given dimension in ascending order by value.
            /// </summary>
            /// <param name="dim">The dimension to sort along</param>
            /// <param name="descending">Controls the sorting order (ascending or descending)</param>
            public Tensor argsort(long dim = -1, bool descending = false)
            {
                var res = LibTorchSharp.THSTensor_argsort(Handle, dim, descending);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Convert each element from degrees to radians.
            /// </summary>
            public Tensor deg2rad()
            {
                var res = LibTorchSharp.THSTensor_deg2rad(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Convert each element from radians to degrees.
            /// </summary>
            public Tensor rad2deg()
            {
                var res = LibTorchSharp.THSTensor_rad2deg(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
            /// Supports broadcasting to a common shape, and integer and float inputs.
            /// </summary>
            /// <param name="other">contains value(s) whose signbit(s) are applied to the magnitudes in input.</param>
            /// <returns>the output tensor</returns>
            public Tensor copysign(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_copysign(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor count_nonzero(long[]? dims = null)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = LibTorchSharp.THSTensor_count_nonzero(Handle, (IntPtr)pdims, dims is null ? 0 : dims.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Estimates the covariance matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.
            /// </summary>
            /// <param name="correction">
            /// Difference between the sample size and sample degrees of freedom.
            /// Defaults to Bessel’s correction, correction = 1 which returns the unbiased estimate,
            /// even if both fweights and aweights are specified.
            /// Correction = 0 will return the simple average.
            /// </param>
            /// <param name="fweights">
            /// A Scalar or 1D tensor of observation vector frequencies representing the number of times each observation should be repeated.
            /// Its numel must equal the number of columns of input.
            /// Must have integral dtype.</param>
            /// <param name="aweights">A Scalar or 1D array of observation vector weights.
            /// These relative weights are typically large for observations considered “important” and smaller for
            /// observations considered less “important”.
            /// Its numel must equal the number of columns of input.
            /// Must have floating point dtype.</param>
            public Tensor cov(long correction = 1, Tensor? fweights = null, Tensor? aweights = null)
            {
                var fwHandle = fweights is null ? IntPtr.Zero : fweights.Handle;
                var awHandle = aweights is null ? IntPtr.Zero : aweights.Handle;
                var res = LibTorchSharp.THSTensor_cov(Handle, correction, fwHandle, awHandle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.
            /// </summary>
            /// <remarks>
            /// Due to floating point rounding, the resulting array may not be Hermitian and its diagonal elements may not be 1.
            /// The real and imaginary values are clipped to the interval [-1, 1] in an attempt to improve this situation.
            /// </remarks>
            public Tensor corrcoef()
            {
                var res = LibTorchSharp.THSTensor_corrcoef(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Constructs a tensor by repeating the elements of input. The reps argument specifies the number of repetitions in each dimension.
            /// </summary>
            /// <param name="reps">The number of repetitions per dimension.</param>

            public Tensor tile(long[] reps)
            {
                unsafe {
                    fixed (long* pdims = reps) {
                        var res = LibTorchSharp.THSTensor_tile(Handle, (IntPtr)pdims, reps.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input.
            /// </summary>

            public Tensor digamma()
            {
                var res = LibTorchSharp.THSTensor_digamma(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input, in place.
            /// </summary>

            public Tensor digamma_()
            {
                var res = LibTorchSharp.THSTensor_digamma_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the logarithm of the gamma function on input.
            /// </summary>

            public Tensor lgamma()
            {
                var res = LibTorchSharp.THSTensor_lgamma(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the logarithm of the gamma function on input, in place.
            /// </summary>

            public Tensor lgamma_()
            {
                var res = LibTorchSharp.THSTensor_lgamma_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the multivariate log-gamma function) with dimension pp element-wise
            /// </summary>
            /// <param name="p">The number of dimensions</param>

            public Tensor mvlgamma(long p)
            {
                var res = LibTorchSharp.THSTensor_mvlgamma(Handle, p);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the multivariate log-gamma function) with dimension pp element-wise, in place.
            /// </summary>
            /// <param name="p">The number of dimensions</param>

            public Tensor mvlgamma_(long p)
            {
                var res = LibTorchSharp.THSTensor_mvlgamma_(Handle, p);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor polygamma(long p)
            {
                var res = LibTorchSharp.THSTensor_polygamma(Handle, p);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor polygamma_(long p)
            {
                var res = LibTorchSharp.THSTensor_polygamma_(Handle, p);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns input. Throws a runtime error if input is a bool tensor.
            /// </summary>

            public Tensor positive()
            {
                if (this.dtype == ScalarType.Bool) throw new ArgumentException("Boolean tensor");
                var res = LibTorchSharp.THSTensor_positive(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the softmax function for the input tensor.
            /// </summary>
            /// <param name="dim">A dimension along which softmax will be computed.</param>
            /// <param name="dtype">The desired data type of returned tensor.</param>
            public Tensor softmax(long dim, ScalarType? dtype = null) =>
                torch.special.softmax(this, dim, dtype);

            public Tensor softplus()
            {
                var res = LibTorchSharp.THSTensor_softplus(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor ravel()
            {
                var res = LibTorchSharp.THSTensor_ravel(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor relu()
            {
                var res = LibTorchSharp.THSTensor_relu(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor relu_()
            {
                var res = LibTorchSharp.THSTensor_relu_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor relu6()
            {
                var res = LibTorchSharp.THSTensor_relu6(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor relu6_()
            {
                var res = LibTorchSharp.THSTensor_relu6_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor celu()
            {
                var res = LibTorchSharp.THSTensor_celu(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor celu_()
            {
                var res = LibTorchSharp.THSTensor_celu_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor elu(Scalar alpha, Scalar scale, Scalar input_scale)
            {
                var res = LibTorchSharp.THSTensor_elu(Handle, alpha.Handle, scale.Handle, input_scale.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor elu_(Scalar alpha, Scalar scale, Scalar input_scale)
            {
                var res = LibTorchSharp.THSTensor_elu_(Handle, alpha.Handle, scale.Handle, input_scale.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor gelu()
            {
                var res = LibTorchSharp.THSTensor_gelu(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor hardsigmoid()
            {
                var res = LibTorchSharp.THSTensor_hardsigmoid(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor hardsigmoid_()
            {
                var res = LibTorchSharp.THSTensor_hardsigmoid_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor hardswish()
            {
                var res = LibTorchSharp.THSTensor_hardswish(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor hardswish_()
            {
                var res = LibTorchSharp.THSTensor_hardswish_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor hardtanh(Scalar min, Scalar max)
            {
                var res = LibTorchSharp.THSTensor_hardtanh(Handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor hardtanh_(Scalar min, Scalar max)
            {
                var res = LibTorchSharp.THSTensor_hardtanh_(Handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor heaviside(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_heaviside(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the regularized lower incomplete gamma function
            /// </summary>
            /// <param name="other">The second non-negative input tensor</param>

            public Tensor igamma(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_igamma(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the regularized upper incomplete gamma function.
            /// </summary>
            /// <param name="other">The second non-negative input tensor</param>

            public Tensor igammac(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_igammac(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the zeroth order modified Bessel function of the first kind for each element of input.
            /// </summary>

            public Tensor i0()
            {
                var res = LibTorchSharp.THSTensor_i0(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with boolean elements representing if each element of input is “close” to the corresponding element of other.
            /// </summary>
            /// <param name="other">Second tensor to compare</param>
            /// <param name="rtol">Relative tolerance</param>
            /// <param name="atol">Absolute tolerance</param>
            /// <param name="nanEqual">If true, then two NaN s will be considered equal</param>
            public Tensor isclose(Tensor other, double rtol = 1e-05, double atol = 1e-08, bool nanEqual = false)
            {
                var res = LibTorchSharp.THSTensor_isclose(Handle, other.Handle, rtol, atol, nanEqual);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Tests if each element of elements is in test_elements.
            /// Returns a boolean tensor of the same shape as elements that is true for elements in test_elements and false otherwise.
            /// </summary>
            /// <param name="test_elements">Values against which to test for each input element</param>
            /// <param name="assumeUnique">If true, assumes both elements and test_elements contain unique elements, which can speed up the calculation.</param>
            /// <param name="invert">If true, inverts the boolean return tensor, resulting in true values for elements not in test_elements.</param>
            public Tensor isin(Tensor test_elements, bool assumeUnique = false, bool invert = false)
            {
                var res = LibTorchSharp.THSTensor_isin(Handle, test_elements.Handle, assumeUnique, invert);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor isinf()
            {
                var res = LibTorchSharp.THSTensor_isinf(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor isfinite()
            {
                var res = LibTorchSharp.THSTensor_isfinite(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor isposinf()
            {
                var res = LibTorchSharp.THSTensor_isposinf(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor isneginf()
            {
                var res = LibTorchSharp.THSTensor_isneginf(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with boolean elements representing if each element of input is <value>NaN</value> or not.
            /// Complex values are considered <value>NaN</value> when either their real and/or imaginary part is <value>NaN</value>.
            /// </summary>
            /// <returns>A boolean tensor that is <value>True</value> where tensor is <value>NaN</value> and <value>False</value> elsewhere</returns>
            [Pure]
            public Tensor isnan()
            {
                var res = LibTorchSharp.THSTensor_isnan(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor isreal()
            {
                var res = LibTorchSharp.THSTensor_isreal(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor leaky_relu(Scalar negative_slope)
            {
                var res = LibTorchSharp.THSTensor_leaky_relu(Handle, negative_slope.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor leaky_relu_(Scalar negative_slope)
            {
                var res = LibTorchSharp.THSTensor_leaky_relu_(Handle, negative_slope.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor selu()
            {
                var res = LibTorchSharp.THSTensor_selu(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor selu_()
            {
                var res = LibTorchSharp.THSTensor_selu_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }


            public Tensor silu()
            {
                var res = LibTorchSharp.THSTensor_silu(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor silu_()
            {
                var res = LibTorchSharp.THSTensor_silu_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor log_sigmoid()
            {
                var res = LibTorchSharp.THSTensor_log_sigmoid(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor lerp(Tensor end, Tensor weight)
            {
                var res = LibTorchSharp.THSTensor_lerp(Handle, end.Handle, weight.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor lerp_(Tensor end, Tensor weight)
            {
                var res = LibTorchSharp.THSTensor_lerp_(Handle, end.Handle, weight.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }


            /// <summary>
            /// Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
            /// batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
            /// </summary>
            /// <param name="batch1">The first batch of matrices to be multiplied</param>
            /// <param name="batch2">The second batch of matrices to be multiplied</param>
            /// <param name="beta">A multiplier for input</param>
            /// <param name="alpha">A multiplier for batch1 @ batch2</param>
            public Tensor baddbmm(Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            {
                var res = LibTorchSharp.THSTensor_baddbmm(Handle, batch1.Handle, batch2.Handle, beta, alpha);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Performs a batch matrix-matrix product of matrices stored in input and mat2.
            /// </summary>
            /// <param name="batch2">the second batch of matrices to be multiplied</param>
            /// <returns></returns>
            public Tensor bmm(Tensor batch2)
            {
                var res = LibTorchSharp.THSTensor_bmm(Handle, batch2.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the indices of the buckets to which each value in the input belongs, where the boundaries of the buckets are set by boundaries.
            /// Return a new tensor with the same size as input. If right is false (default), then the left boundary is closed.
            /// </summary>
            /// <param name="boundaries">1-D tensor, must contain a monotonically increasing sequence.</param>
            /// <param name="outInt32">indicate the output data type. torch.int32 if True, torch.int64 otherwise.
            /// Default value is False, i.e. default output data type is torch.int64.</param>
            /// <param name="right">if false, return the first suitable location that is found. If rrue, return the last such index.
            /// If no suitable index found, return 0 for non-numerical value (eg. nan, inf) or the size of boundaries (one pass the last index).
            /// In other words, if false, gets the lower bound index for each value in input from boundaries.
            /// If true, gets the upper bound index instead. Default value is False.</param>

            public Tensor bucketize(Tensor boundaries, bool outInt32 = false, bool right = false)
            {
                var res = LibTorchSharp.THSTensor_bucketize(Handle, boundaries.Handle, outInt32, right);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Count the frequency of each value in an array of non-negative ints.
            /// </summary>
            public Tensor bincount(Tensor? weights, long minlength = 0)
            {
                var weightsHandle = (weights is null ? IntPtr.Zero : weights.Handle);
                var res = LibTorchSharp.THSTensor_bincount(Handle, weightsHandle, minlength);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            public Tensor @bool() => this.to_type(ScalarType.Bool);

            public Tensor @byte() => this.to_type(ScalarType.Byte);

            public Tensor @char() => this.to_type(ScalarType.Int8);

            public Tensor @int() => this.to_type(ScalarType.Int32);

            public Tensor @long() => this.to_type(ScalarType.Int64);

            public Tensor @float() => this.to_type(ScalarType.Float32);

            public Tensor @double() => this.to_type(ScalarType.Float64);


            /// <summary>
            /// Divide the channels in a tensor into g groups and rearrange them.
            ///
            /// See: https://pytorch.org/docs/1.10/generated/torch.nn.ChannelShuffle.html#channelshuffle
            /// </summary>
            /// <param name="groups">The number of groups to divide channels in.</param>
            public Tensor channel_shuffle(long groups)
            {
                var res = LibTorchSharp.THSTensor_channel_shuffle(Handle, groups);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Clamps all elements in input into the range [ min, max ].
            /// </summary>
            /// <param name="min">The minimum value</param>
            /// <param name="max">The maximum value</param>
            public Tensor clamp(Scalar? min = null, Scalar? max = null)
            {
                var res = LibTorchSharp.THSTensor_clamp(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Clamps all elements in input into the range [ min, max ].
            /// </summary>
            /// <param name="min">The minimum value</param>
            /// <param name="max">The maximum value</param>
            public Tensor clamp(Tensor? min = null, Tensor? max = null)
            {
                var res = LibTorchSharp.THSTensor_clamp_tensor(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Alias for 'clamp'
            /// </summary>
            /// <param name="min">The minimum value</param>
            /// <param name="max">The maximum value</param>
            /// <returns></returns>
            public Tensor clip(Scalar? min = null, Scalar? max = null) => clamp(min, max);


            /// <summary>
            /// Clamps all elements in input into the range [ min, max ] in place.
            /// </summary>
            /// <param name="min">The minimum value</param>
            /// <param name="max">The maximum value</param>
            public Tensor clamp_(Scalar? min = null, Scalar? max = null)
            {
                var res = LibTorchSharp.THSTensor_clamp_(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Clamps all elements in input into the range [ min, max ] in place.
            /// </summary>
            /// <param name="min">The minimum value</param>
            /// <param name="max">The maximum value</param>
            public Tensor clamp_(Tensor? min = null, Tensor? max = null)
            {
                var res = LibTorchSharp.THSTensor_clamp_tensor_(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clamp_max(Scalar max)
            {
                var res = LibTorchSharp.THSTensor_clamp_max(Handle, max.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clamp_max_(Scalar max)
            {
                var res = LibTorchSharp.THSTensor_clamp_max_(Handle, max.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clamp_min(Scalar min)
            {
                var res = LibTorchSharp.THSTensor_clamp_min(Handle, min.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clamp_min_(Scalar min)
            {
                var res = LibTorchSharp.THSTensor_clamp_min_(Handle, min.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the n-th forward difference along the given dimension.
            /// </summary>
            /// <param name="n">The number of times to recursively compute the difference</param>
            /// <param name="dim">The dimension to compute the difference along. Default is the last dimension.</param>
            /// <param name="prepend">
            /// Values to prepend or append to input along dim before computing the difference.
            /// Their dimensions must be equivalent to that of input, and their shapes must match input’s shape except on dim.
            /// </param>
            /// <param name="append">
            /// Values to prepend or append to input along dim before computing the difference.
            /// Their dimensions must be equivalent to that of input, and their shapes must match input’s shape except on dim.
            /// </param>
            public Tensor diff(long n = 1, long dim = -1, Tensor? prepend = null, Tensor? append = null)
            {
                if (n != 1) throw new NotImplementedException("Tensor.diff with n != 1");
                var res = LibTorchSharp.THSTensor_diff(Handle, n, dim, (prepend is Tensor) ? (IntPtr)prepend.Handle : IntPtr.Zero, (append is Tensor) ? (IntPtr)append.Handle : IntPtr.Zero);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
            /// If input is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of input.
            /// </summary>
            /// <param name="diagonal">
            /// The argument diagonal controls which diagonal to consider:
            /// If diagonal is 0, it is the main diagonal.
            /// If diagonal is greater than 0, it is above the main diagonal.
            /// If diagonal is less than 0, it is below the main diagonal.
            /// </param>
            public Tensor diag(long diagonal = 0)
            {
                var res = LibTorchSharp.THSTensor_diag(Handle, diagonal);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the sum of the elements of the diagonal of the input 2-D matrix.
            /// </summary>
            /// <returns></returns>
            public Tensor trace()
            {
                if (ndim != 2)
                    throw new ArgumentException($"Expected a matrix, but got tensor with ndim == {ndim}");
                var res = LibTorchSharp.THSTensor_trace(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input.
            /// To facilitate creating batched diagonal matrices, the 2D planes formed by the last two dimensions of the returned tensor are chosen by default.
            /// 
            /// The argument offset controls which diagonal to consider:
            ///   If offset is equal to 0, it is the main diagonal.
            ///   If offset is greater than 0, it is above the main diagonal.
            ///   If offset is less than 0, it is below the main diagonal.
            ///   
            /// The size of the new matrix will be calculated to make the specified diagonal of the size of the last input dimension.Note that for offset other than 0,
            /// 
            /// the order of dim1 and dim2 matters.Exchanging them is equivalent to changing the sign of offset.
            /// </summary>
            /// <param name="offset">Which diagonal to consider.</param>
            /// <param name="dim1">First dimension with respect to which to take diagonal. </param>
            /// <param name="dim2">Second dimension with respect to which to take diagonal</param>
            public Tensor diag_embed(long offset = 0L, long dim1 = -2L, long dim2 = -1L)
            {
                var res = LibTorchSharp.THSTensor_diag_embed(Handle, offset, dim1, dim2);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
            /// If input is a matrix (2-D tensor), then returns a 2-D tensor with diagonal elements equal to a flattened input.
            /// </summary>
            /// <param name="offset">
            /// The argument diagonal controls which diagonal to consider:
            /// If diagonal is 0, it is the main diagonal.
            /// If diagonal is greater than 0, it is above the main diagonal.
            /// If diagonal is less than 0, it is below the main diagonal.
            /// </param>
            public Tensor diagflat(long offset = 0)
            {
                var res = LibTorchSharp.THSTensor_diagflat(Handle, offset);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.
            /// The argument offset controls which diagonal to consider:
            ///
            /// If offset = 0, it is the main diagonal.
            /// If offset &gt; 0, it is above the main diagonal.
            /// If offset &lt; 0, it is below the main diagonal.
            /// </summary>
            /// <param name="offset">Which diagonal to consider. Default: 0 (main diagonal).</param>
            /// <param name="dim1">First dimension with respect to which to take diagonal. Default: 0.</param>
            /// <param name="dim2">Second dimension with respect to which to take diagonal. Default: 1.</param>
            /// <remarks>
            /// Applying torch.diag_embed() to the output of this function with the same arguments yields a diagonal matrix with the diagonal entries of the input.
            /// However, torch.diag_embed() has different default dimensions, so those need to be explicitly specified.
            /// </remarks>
            public Tensor diagonal(long offset = 0, long dim1 = 0, long dim2 = 0)
            {
                var res = LibTorchSharp.THSTensor_diagonal(Handle, offset, dim1, dim2);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            /// <summary>
            /// Computes the error function of the input.
            /// </summary>
            /// <returns></returns>
            public Tensor erf()
            {
                var res = LibTorchSharp.THSTensor_erf(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the error function of the input in place.
            /// </summary>
            public Tensor erf_()
            {
                var res = LibTorchSharp.THSTensor_erf_(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the complementary error function of input.
            /// </summary>
            /// <returns></returns>
            public Tensor erfc()
            {
                var res = LibTorchSharp.THSTensor_erfc(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the complementary error function of input in place
            /// </summary>
            /// <returns></returns>
            public Tensor erfc_()
            {
                var res = LibTorchSharp.THSTensor_erfc_(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse error function of input.
            /// </summary>
            /// <returns></returns>
            public Tensor erfinv()
            {
                var res = LibTorchSharp.THSTensor_erfinv(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse error function of input in place.
            /// </summary>
            /// <returns></returns>
            public Tensor erfinv_()
            {
                var res = LibTorchSharp.THSTensor_erfinv_(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor eq(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_eq(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor equal(Tensor target) => eq(target);

            public Tensor eq_(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_eq_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor eq(Scalar target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_eq_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor eq_(Scalar target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_eq_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public bool Equals(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_equal(Handle, target.Handle);
                CheckForErrors();
                return res;
            }

            /// <summary>
            /// This function checks if all input and other lie within a certain distance from each other
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rtol">Relative tolerance</param>
            /// <param name="atol">Absolute tolerance</param>
            /// <param name="equal_nan">If true, then two NaN s will be considered equal</param>
            public bool allclose(Tensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_allclose(Handle, target.Handle, rtol, atol, equal_nan);
                CheckForErrors();
                return res;
            }

            public Tensor ge(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_ge(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor greater_equal(Tensor target) => ge(target);

            public Tensor ge_(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_ge_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor ge(Scalar target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_ge_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor ge_(Scalar target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_ge_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor gt(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_gt(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor greater(Tensor target) => gt(target);

            public Tensor gt_(Tensor target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_gt_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor gt(Scalar target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_gt_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor gt_(Scalar target)
            {
                if (target is null) return false;
                var res = LibTorchSharp.THSTensor_gt_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Kronecker product of input and other.
            /// </summary>
            /// <param name="other">The second tensor</param>
            /// <returns></returns>
            public Tensor kron(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_kron(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise least common multiple (LCM) of input and other.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <remarks>Both input and other must have integer types.</remarks>
            public Tensor lcm(Tensor other)
            {
                if (!torch.is_integral(this.dtype) || !torch.is_integral(other.dtype))
                    throw new ArgumentException("Arguments to 'lcm' must have integer element types.");
                var res = LibTorchSharp.THSTensor_lcm(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise least common multiple (LCM) of input and other, in place.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <remarks>Both input and other must have integer types.</remarks>
            public Tensor lcm_(Tensor other)
            {
                if (!torch.is_integral(this.dtype) || !torch.is_integral(other.dtype))
                    throw new ArgumentException("Arguments to 'lcm' must have integer element types.");
                var res = LibTorchSharp.THSTensor_lcm_(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Multiplies input by pow(2,other).
            /// </summary>
            /// <param name="other">A tensor of exponents, typically integers</param>

            /// <remarks>Typically this function is used to construct floating point numbers by multiplying mantissas in input with integral powers of two created from the exponents in other.</remarks>
            public Tensor ldexp(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_ldexp(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Multiplies input by pow(2,other) in place.
            /// </summary>
            /// <param name="other">A tensor of exponents, typically integers</param>

            /// <remarks>Typically this function is used to construct floating point numbers by multiplying mantissas in input with integral powers of two created from the exponents in other.</remarks>
            public Tensor ldexp_(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_ldexp_(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor le(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_le(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less_equal(Tensor target) => le(target);

            public Tensor le_(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_le_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less_equal_(Tensor target) => le_(target);

            public Tensor le(Scalar target)
            {
                var res = LibTorchSharp.THSTensor_le_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor le_(Scalar target)
            {
                var res = LibTorchSharp.THSTensor_le_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor lt(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_lt(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less(Tensor target) => lt(target);

            public Tensor lt_(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_lt_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor lt(Scalar target)
            {
                var res = LibTorchSharp.THSTensor_lt_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor lt_(Scalar target)
            {
                var res = LibTorchSharp.THSTensor_lt_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor masked_fill(Tensor mask, Scalar value)
            {
                var res = LibTorchSharp.THSTensor_masked_fill(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor masked_fill_(Tensor mask, Scalar value)
            {
                var res = LibTorchSharp.THSTensor_masked_fill_(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor masked_scatter(Tensor mask, Tensor value)
            {
                var res = LibTorchSharp.THSTensor_masked_scatter(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            public Tensor masked_scatter_(Tensor mask, Tensor value)
            {
                var res = LibTorchSharp.THSTensor_masked_scatter_(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor masked_select(Tensor mask)
            {
                if (mask.dtype != ScalarType.Bool) throw new ArgumentException("The mask tensor must be Boolean.");
                var res = LibTorchSharp.THSTensor_masked_select(Handle, mask.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public (Tensor values, Tensor indexes) topk(int k, int dim = -1, bool largest = true, bool sorted = true)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_topk(Handle, pa.CreateArray, k, dim, largest, sorted);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }


            /// <summary>
            /// Removes a tensor dimension.
            /// </summary>
            /// <param name="dimension">The dimension to remove.</param>
            /// <returns>An array of all slices along a given dimension, already without it.</returns>
            public Tensor[] unbind(long dimension = 0L)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_unbind(Handle, pa.CreateArray, dimension);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Returns a view of the original tensor which contains all slices of size 'size' from the tensor in the given dimension.
            /// </summary>
            /// <param name="dimension">Dimension in which unfolding happens</param>
            /// <param name="size">The size of each slice that is unfolded</param>
            /// <param name="step">The step between each slice</param>
            public Tensor unfold(long dimension, long size, long step)
            {
                var res = LibTorchSharp.THSTensor_unfold(Handle, dimension, size, step);
                if (res == IntPtr.Zero) CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="size">The size of a single chunk</param>
            /// <param name="dim">The dimension along which to split the tensor.</param>

            public Tensor[] split(long size, int dim = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_split_with_size(Handle, pa.CreateArray, size, dim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="sizes">A list of sizes for each chunk</param>
            /// <param name="dim">The dimension along which to split the tensor.</param>

            public Tensor[] split(ReadOnlySpan<long> sizes, long dim = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            LibTorchSharp.THSTensor_split_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dim);
                            CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="sizes">A list of sizes for each chunk</param>
            /// <param name="dim">The dimension along which to split the tensor.</param>

            public Tensor[] split(long[] sizes, long dim = 0)
            {
                return split((ReadOnlySpan<long>)sizes, dim);
            }

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="sizes">A list of sizes for each chunk</param>

            public Tensor[] split(params long[] sizes)
            {
                return split((ReadOnlySpan<long>)sizes, 0);
            }

            /// <summary>
            /// Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.
            /// </summary>
            /// <param name="size"></param>
            /// <param name="dim"></param>
            /// <returns></returns>
            public Tensor[] tensor_split(long size, long dim = 0L)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_tensor_split_with_size(Handle, pa.CreateArray, size, dim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.
            /// </summary>
            /// <param name="sizes"></param>
            /// <param name="dim"></param>
            /// <returns></returns>
            public Tensor[] tensor_split(long[] sizes, long dim = 0L)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            LibTorchSharp.THSTensor_tensor_split_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dim);
                            CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            public Tensor[] tensor_split(Tensor indices, long dim = 0L)
            {
                if (indices.dtype != ScalarType.Int64) throw new ArgumentException("Tensor indices should be Int64 in 'tensor_split");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_tensor_split_with_tensor_sizes(Handle, pa.CreateArray, indices.Handle, dim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to sizes.
            /// </summary>
            /// <param name="size">The size of each chunk</param>

            public Tensor[] vsplit(long size)
            {
                if (this.shape[0] % size != 0) throw new ArgumentException("The first dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_vsplit_with_size(Handle, pa.CreateArray, size);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to sizes.
            /// </summary>
            /// <param name="sizes">A list of split points</param>

            public Tensor[] vsplit(params long[] sizes)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            LibTorchSharp.THSTensor_vsplit_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
                            CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to indices.
            /// </summary>
            /// <param name="indices">A list of split points</param>

            public Tensor[] vsplit(Tensor indices) => tensor_split(indices, 0);


            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to sizes.
            /// </summary>
            /// <param name="size">The size of each chunk</param>

            public Tensor[] hsplit(long size)
            {
                if (this.shape[1] % size != 0) throw new ArgumentException("The second dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_hsplit_with_size(Handle, pa.CreateArray, size);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to sizes.
            /// </summary>
            /// <param name="sizes">A list of split points</param>

            public Tensor[] hsplit(params long[] sizes)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            LibTorchSharp.THSTensor_hsplit_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
                            CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices.
            /// </summary>
            /// <param name="indices">A list of split points</param>

            public Tensor[] hsplit(Tensor indices) => tensor_split(indices, 1);

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="size">The size of each chunk</param>

            public Tensor[] dsplit(long size)
            {
                if (this.shape[2] % size != 0) throw new ArgumentException("The third dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_dsplit_with_size(Handle, pa.CreateArray, size);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="indices_or_sections">A list of split points</param>
            public Tensor[] dsplit((long, long) indices_or_sections)
                => dsplit(new long[] { indices_or_sections.Item1, indices_or_sections.Item2 });

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="indices_or_sections">A list of split points</param>
            public Tensor[] dsplit((long, long, long) indices_or_sections)
                => dsplit(new long[]{ indices_or_sections.Item1, indices_or_sections.Item2, indices_or_sections.Item3 });

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="indices_or_sections">A list of split points</param>
            public Tensor[] dsplit((long, long, long, long) indices_or_sections)
                => dsplit(new long[]{ indices_or_sections.Item1, indices_or_sections.Item2, indices_or_sections.Item3, indices_or_sections.Item4 });

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="sizes">A list of split points</param>
            public Tensor[] dsplit(params long[] sizes)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            LibTorchSharp.THSTensor_dsplit_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
                            CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="indices">A list of split points</param>
            public Tensor[] dsplit(Tensor indices) => tensor_split(indices, 2);

            /// <summary>
            /// Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
            /// </summary>
            /// <param name="chunks">The number of chunks to return</param>
            /// <param name="dim">Dimension along which to split the tensor</param>

            /// <remarks>The last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.</remarks>
            public Tensor[] chunk(long chunks, long dim = 0L)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_chunk(Handle, pa.CreateArray, chunks, dim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            /// <summary>
            /// Returns a named tuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim. And indices is the index location of each element found.
            /// If dim is not given, the last dimension of the input is chosen.
            /// </summary>
            /// <param name="k">k for the k-th smallest element</param>
            /// <param name="dim">The dimension to find the kth value along</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>

            public (Tensor values, Tensor indices) kthvalue(long k, long? dim, bool keepdim = false)
            {
                var values = LibTorchSharp.THSTensor_kthvalue(Handle, k, dim.HasValue ? dim.Value : -1, keepdim, out var indices);
                if (values == IntPtr.Zero || indices == IntPtr.Zero)
                    CheckForErrors();
                return (new Tensor(values), new Tensor(indices));
            }

            /// <summary>
            /// Returns a named tuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim. And indices is the index location of each element found.
            /// If dim is not given, the last dimension of the input is chosen.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="k">k for the k-th smallest element</param>
            /// <param name="dim">The dimension to find the kth value along</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            [Obsolete("use torch.kthvalue", false)]
            public static (Tensor values, Tensor indices) kthvalue(Tensor input, long k, long? dim, bool keepdim = false)
                => input.kthvalue(k, dim, keepdim);

            /// <summary>
            /// Returns the maximum value of all elements in the input tensor.
            /// </summary>
            public Tensor max()
            {
                var res = LibTorchSharp.THSTensor_max(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            /// <summary>
            /// Computes the element-wise maximum of input and other.
            /// </summary>
            /// <param name="other">The second input tensor</param>
            /// <returns></returns>
            public Tensor maximum(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_max_elementwise(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a named tuple (values, indexes) where values is the maximum value of each row of the input tensor in the given dimension dim.
            /// And indices is the index location of each maximum value found (argmax).
            /// </summary>
            /// <param name="dim">the dimension to reduce.</param>
            /// <param name="keepdim">whether the output tensor has dim retained or not. Default: false.</param>
            /// <remarks>If keepdim is true, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
            /// Otherwise, dim is squeezed(see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.</remarks>
            public (Tensor values, Tensor indexes) max(long dim, bool keepdim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_max_along_dimension(Handle, pa.CreateArray, dim, keepdim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            /// <summary>
            /// Returns the mean value of all elements in the input tensor.
            /// </summary>
            /// <returns></returns>
            public Tensor mean()
            {
                var res = LibTorchSharp.THSTensor_mean(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the q-th quantiles of all elements in the input tensor, doing a linear interpolation when the q-th quantile lies between two data points.
            /// </summary>
            /// <param name="q">1D tensor of quantile values in the range [0, 1]</param>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            public Tensor quantile(Tensor q, long dim = -1, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_quantile(Handle, q.Handle, dim, keepdim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// This is a variant of torch.quantile() that “ignores” NaN values, computing the quantiles q as if NaN values in input did not exist.
            /// If all values in a reduced row are NaN then the quantiles for that reduction will be NaN.
            /// </summary>
            /// <seealso cref="Tensor.quantile(Tensor, long, bool)"/>
            /// <param name="q">1D tensor of quantile values in the range [0, 1]</param>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>

            public Tensor nanquantile(Tensor q, long dim = -1, bool keepdim = false)
            {
                var res = LibTorchSharp.THSTensor_nanquantile(Handle, q.Handle, dim, keepdim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a named tuple (values, indices) where values is the mode value of each row of the input tensor in the given dimension dim,
            /// i.e. a value which appears most often in that row, and indices is the index location of each mode value found.
            /// </summary>
            /// <param name="dim">The dimension to reduce, the last dimension by default.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not</param>


            public (Tensor values, Tensor indices) mode(long dim = -1L, bool keepdim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_mode(Handle, pa.CreateArray, dim, keepdim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (values: new Tensor(ptrArray[0]), indices: new Tensor(ptrArray[1]));
            }

            /// <summary>
            /// Returns the mean value of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.
            /// </summary>
            /// <param name="dimensions">The dimension or dimensions to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
            /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is cast to dtype before the operation is performed. This is useful for preventing data type overflows.</param>
            /// <remarks>
            /// If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
            /// Otherwise, dim is squeezed(see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
            /// </remarks>
            public Tensor mean(long[] dimensions, bool keepdim = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = LibTorchSharp.THSTensor_mean_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, keepdim, type.HasValue, (sbyte)type.GetValueOrDefault());
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public Tensor var(long[] dimensions, bool keepdim = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = LibTorchSharp.THSTensor_var_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, keepdim, type.HasValue, (sbyte)type.GetValueOrDefault());
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Returns the median of the values in input.
            /// </summary>
            /// <remarks>
            /// The median is not unique for input tensors with an even number of elements.
            /// In this case the lower of the two medians is returned. To compute the mean of both medians, use torch.quantile() with q=0.5 instead.
            /// </remarks>
            public Tensor median()
            {
                var res = LibTorchSharp.THSTensor_median(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the minimum value of all elements in the input tensor.
            /// </summary>
            public Tensor min()
            {
                var res = LibTorchSharp.THSTensor_min(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor minimum(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_min_elementwise(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a named tuple (values, indexes) where values is the minimum value of each row of the input tensor in the given dimension dim.
            /// And indices is the index location of each minimum value found (argmin).
            /// </summary>
            /// <param name="dim">the dimension to reduce.</param>
            /// <param name="keepdim">whether the output tensor has dim retained or not. Default: false.</param>
            /// <remarks>
            /// If keepdim is true, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
            /// Otherwise, dim is squeezed(see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.
            /// </remarks>
            public (Tensor values, Tensor indexes) min(long dim, bool keepdim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    LibTorchSharp.THSTensor_min_along_dimension(Handle, pa.CreateArray, dim, keepdim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            /// <summary>
            /// Sorts the elements of the input tensor along its first dimension in ascending order by value.
            /// </summary>
            public Tensor msort()
            {
                var res = LibTorchSharp.THSTensor_msort(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Sorts the elements of the input tensor along a given dimension in ascending order by value.
            /// </summary>
            /// <param name="dim">The dimension to sort along. If dim is not given, the last dimension of the input is chosen.</param>
            /// <param name="descending">Controls the sorting order (ascending or descending)</param>
            /// <param name="stable">Makes the sorting routine stable, which guarantees that the order of equivalent elements is preserved.</param>
            /// <returns>A named tuple of (values, indices) is returned, where the values are the sorted values and indices are the indices of the elements in the original input tensor.</returns>
            public (Tensor Values, Tensor Indices) sort(long dim = -1, bool descending = false, bool stable = false)
            {
                var res = LibTorchSharp.THSTensor_sort(Handle, dim, descending, stable, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }

            public Tensor ne(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_ne(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor not_equal(Tensor target) => ne(target);

            public Tensor ne_(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_ne_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor not_equal_(Tensor target) => ne_(target);

            public Tensor ne(Scalar target)
            {
                var res = LibTorchSharp.THSTensor_ne_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor ne_(Scalar target)
            {
                var res = LibTorchSharp.THSTensor_ne_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the p-norm of (input - other).
            /// The shapes of input and other must be broadcastable.
            /// </summary>
            /// <param name="other">Right-hand side input.</param>
            /// <param name="p">The norm to be computed.</param>
            /// <returns></returns>
            public Tensor dist(Tensor other, float p = 2.0f)
            {
                var res = LibTorchSharp.THSTensor_dist(Handle, other.Handle, p);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the matrix norm or vector norm of a given tensor.
            /// </summary>
            /// <param name="p">The norm to be computed.</param>
            public Tensor norm(float p = 2.0f)
            {
                var res = LibTorchSharp.THSTensor_norm(Handle, p);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the matrix norm or vector norm of a given tensor.
            /// </summary>
            public Tensor norm(int dim, bool keepdim = false, float p = 2.0f)
            {
                var res = LibTorchSharp.THSTensor_norm_along_dimension(Handle, dim, keepdim, p);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Outer product of input and vec2.
            /// </summary>
            /// <param name="vec2">1-D input vector.</param>
            /// <remarks>If input is a vector of size n and vec2 is a vector of size m, then out must be a matrix of size n×m.</remarks>
            public Tensor outer(Tensor vec2)
            {
                var res = LibTorchSharp.THSTensor_outer(Handle, vec2.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Outer product of input and vec2.
            /// </summary>
            /// <param name="vec2">1-D input vector.</param>
            /// <remarks>If input is a vector of size n and vec2 is a vector of size m, then out must be a matrix of size n×m.</remarks>
            public Tensor ger(Tensor vec2) => outer(vec2);

            /// <summary>
            /// Computes the dot product for 1D tensors.
            /// For higher dimensions, sums the product of elements from input and other along their last dimension.
            /// </summary>
            /// <param name="vec2"></param>
            /// <returns></returns>
            public Tensor inner(Tensor vec2)
            {
                var res = LibTorchSharp.THSTensor_inner(Handle, vec2.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Alias for torch.linalg.inv()
            /// </summary>
            public Tensor inverse() => torch.linalg.inv(this);

            public Tensor prelu(Tensor target)
            {
                var res = LibTorchSharp.THSTensor_prelu(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise maximum of input and other.
            ///
            /// This is like torch.maximum() except it handles NaNs differently: if exactly one of the two elements being compared is a NaN
            /// then the non-NaN element is taken as the maximum.
            /// Only if both elements are NaN is NaN propagated.
            /// </summary>
            /// <param name="other">The second input tensor</param>
            /// <returns></returns>
            public Tensor fmax(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_fmax(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise minimum of input and other.
            ///
            /// This is like torch.minimum() except it handles NaNs differently: if exactly one of the two elements being compared is a NaN
            /// then the non-NaN element is taken as the minimum.
            /// Only if both elements are NaN is NaN propagated.
            /// </summary>
            /// <param name="other">The second input tensor</param>
            public Tensor fmin(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_fmin(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
            /// </summary>
            /// <param name="p">The power for the norm computation</param>
            /// <param name="dim">The dimension to slice over to get the sub-tensors</param>
            /// <param name="maxnorm">The maximum norm to keep each sub-tensor under</param>
            /// <returns></returns>
            public Tensor renorm(float p, long dim, float maxnorm)
            {
                var res = LibTorchSharp.THSTensor_renorm(Handle, p, dim, maxnorm);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Alias for torch.special.expit()
            /// </summary>
            /// <returns></returns>
            public Tensor sigmoid()
            {
                var res = LibTorchSharp.THSTensor_sigmoid(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Alias for torch.special.expit(), works in place.
            /// </summary>
            public Tensor sigmoid_()
            {
                var res = LibTorchSharp.THSTensor_sigmoid_(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Calculates the standard deviation of all elements in the tensor.
            /// </summary>
            [Pure]
            public Tensor std(bool unbiased = true)
            {
                var res = LibTorchSharp.THSTensor_std(Handle, unbiased);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Compute variance of elements of input tensor.
            /// </summary>
            /// <param name="unbiased">If unbiased is true, Bessel’s correction will be used. Otherwise, the sample variance is calculated, without any correction.</param>
            /// <returns></returns>
            [Pure]
            public Tensor var(bool unbiased = true)
            {
                var res = LibTorchSharp.THSTensor_var(Handle, unbiased);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor std(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _std(dimensions, unbiased, keepdim, type);
            }

            ///<summary>Calculates the variance of all elements in the input tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor var(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _var(dimensions, unbiased, keepdim, type);
            }

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor std(long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _std(dimensions, unbiased, keepdim, type);
            }

            ///<summary>Calculates the variance of all elements in the input tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor var(long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _var(dimensions, unbiased, keepdim, type);
            }

            // private, shared implementation
            private unsafe Tensor _std(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                fixed (long* pdims = dimensions) {
                    var res = LibTorchSharp.THSTensor_std_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepdim);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            // private, shared implementation
            private unsafe Tensor _var(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                fixed (long* pdims = dimensions) {
                    var res = LibTorchSharp.THSTensor_var_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepdim);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor std(long dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => std(stackalloc[] { dim }, unbiased, keepdim, type);

            /// <summary>Calculates the variance of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor var(long dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => var(stackalloc[] { dim }, unbiased, keepdim, type);

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor std((long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => std(stackalloc[] { dim.Item1, dim.Item2 }, unbiased, keepdim, type);

            /// <summary>Calculates the variance of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor var((long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => var(stackalloc[] { dim.Item1, dim.Item2 }, unbiased, keepdim, type);

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor std((long, long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => std(stackalloc[] { dim.Item1, dim.Item2, dim.Item3 }, unbiased, keepdim, type);

            /// <summary>Calculates the variance of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [Pure]
            public Tensor var((long, long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => var(stackalloc[] { dim.Item1, dim.Item2, dim.Item3 }, unbiased, keepdim, type);

            /// <summary>
            /// Calculates the standard deviation and mean of all elements in the tensor.
            /// </summary>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) std_mean(bool unbiased = true)
            {
                var res = LibTorchSharp.THSTensor_std_mean(Handle, unbiased, out var mean);
                if (res == IntPtr.Zero || mean == IntPtr.Zero)
                    CheckForErrors();
                return (new Tensor(res), new Tensor(mean));
            }

            /// <summary>
            /// Calculates the variance and mean of all elements in the tensor.
            /// </summary>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
            [Pure]
            public (Tensor @var, Tensor mean) var_mean(bool unbiased = true)
            {
                var res = LibTorchSharp.THSTensor_var_mean(Handle, unbiased, out var mean);
                if (res == IntPtr.Zero || mean == IntPtr.Zero)
                    CheckForErrors();
                return (new Tensor(res), new Tensor(mean));
            }

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) std_mean(long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _std_mean(dimensions, unbiased, keepdim, type);
            }

            /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>

            [Pure]
            public (Tensor std, Tensor mean) var_mean(long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _var_mean(dimensions, unbiased, keepdim, type);
            }

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) std_mean(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _std_mean(dimensions, unbiased, keepdim, type);
            }

            /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) var_mean(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                return _var_mean(dimensions, unbiased, keepdim, type);
            }

            // private, shared implementation
            private unsafe (Tensor std, Tensor mean) _std_mean(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                fixed (long* pdims = dimensions) {
                    var res = LibTorchSharp.THSTensor_std_mean_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepdim, out var mean);
                    if (res == IntPtr.Zero || mean == IntPtr.Zero) { CheckForErrors(); }
                    return (new Tensor(res), new Tensor(mean));
                }
            }

            // private, shared implementation
            private unsafe (Tensor @var, Tensor mean) _var_mean(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            {
                fixed (long* pdims = dimensions) {
                    var res = LibTorchSharp.THSTensor_var_mean_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepdim, out var @var);
                    if (res == IntPtr.Zero || @var == IntPtr.Zero) { CheckForErrors(); }
                    return (new Tensor(res), new Tensor(@var));
                }
            }

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) std_mean(long dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => std_mean(new[] { dim }, unbiased, keepdim, type);

            /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
            [Pure]
            public (Tensor @var, Tensor mean) var_mean(long dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => var_mean(new[] { dim }, unbiased, keepdim, type);

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) std_mean((long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => std_mean(stackalloc[] { dim.Item1, dim.Item2 }, unbiased, keepdim, type);

            /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
            [Pure]
            public (Tensor @var, Tensor mean) var_mean((long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => var_mean(stackalloc[] { dim.Item1, dim.Item2 }, unbiased, keepdim, type);

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
            [Pure]
            public (Tensor std, Tensor mean) std_mean((long, long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => std_mean(stackalloc[] { dim.Item1, dim.Item2, dim.Item3 }, unbiased, keepdim, type);

            /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample variance is calculated, without any correction.
            /// </remarks>
            /// <param name="dim">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dim" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
            [Pure]
            public (Tensor @var, Tensor mean) var_mean((long, long, long) dim, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
                => var_mean(stackalloc[] { dim.Item1, dim.Item2, dim.Item3 }, unbiased, keepdim, type);

            /// <summary>
            /// Returns the sum of all elements in the :attr:`input` tensor.
            /// </summary>
            public Tensor sum(ScalarType? type = null)
            {
                var res = LibTorchSharp.THSTensor_sum(Handle, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            private unsafe Tensor _sum(ReadOnlySpan<long> dimensions, bool keepdim = false, ScalarType? type = null)
            {
                fixed (long* pdims = dimensions) {
                    var res = LibTorchSharp.THSTensor_sum_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, keepdim, type.HasValue, (sbyte)type.GetValueOrDefault());
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long[] dim, bool keepdim = false, ScalarType? type = null)
            {
                return _sum(dim, keepdim, type);
            }

            /// <summary>
            /// Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(ReadOnlySpan<long> dim, bool keepdim = false, ScalarType? type = null)
            {
                return _sum(dim, keepdim, type);
            }
            /// <summary>
            /// Returns the sum of each row of the input tensor in the given dimension.
            /// </summary>
            public Tensor sum(long dim, bool keepdim = false, ScalarType? type = null)
            {
                return _sum(stackalloc long[] { dim }, keepdim, type);
            }

            /// <summary>
            /// Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long dim0, long dim1, bool keepdim = false, ScalarType? type = null)
            {
                return _sum(stackalloc long[] { dim0, dim1 }, keepdim, type);
            }

            /// <summary>
            /// Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long dim0, long dim1, long dim2, bool keepdim = false, ScalarType? type = null)
            {
                return _sum(stackalloc long[] { dim0, dim1, dim2 }, keepdim, type);
            }

            /// <summary>
            /// Returns a new view of the tensor with singleton dimensions expanded to a larger size.
            /// </summary>
            public Tensor expand(ReadOnlySpan<long> sizes, bool isImplicit = false)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_expand(Handle, (IntPtr)psizes, sizes.Length, isImplicit);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public Tensor expand(Size sizes, bool isImplicit = false)
            {
                return expand((ReadOnlySpan<long>)sizes.Shape, isImplicit);
            }

            /// <summary>
            /// Returns a new view of the tensor with singleton dimensions expanded to a larger size.
            /// </summary>
            public Tensor expand(long[] sizes, bool isImplicit = false)
            {
                return expand((ReadOnlySpan<long>)sizes, isImplicit);
            }

            /// <summary>
            /// Expand this tensor to the same size as other.
            /// </summary>

            public Tensor expand_as(Tensor other) => expand(other.shape);


            /// <summary>
            /// Returns a new view of the tensor with singleton dimensions expanded to a larger size.
            /// </summary>
            public Tensor expand(params long[] sizes)
            {
                return expand((ReadOnlySpan<long>)sizes, false);
            }

            /// <summary>
            /// Repeats this tensor along the specified dimensions.
            /// </summary>
            public Tensor repeat(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_repeat(Handle, (IntPtr)psizes, sizes.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public Tensor repeat_interleave(Tensor repeats, long? dim = null, long? output_size = null)
            {
                long _dim = dim ?? long.MinValue;
                long _output_size = output_size ?? long.MinValue;
                var res = LibTorchSharp.THSTensor_repeat_interleave(Handle, repeats.Handle, _dim, _output_size);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor repeat_interleave(long repeats, long? dim = null, long? output_size = null)
            {
                long _dim = dim ?? long.MinValue;
                long _output_size = output_size ?? long.MinValue;
                var res = LibTorchSharp.THSTensor_repeat_interleave_int64(Handle, repeats, _dim, _output_size);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Broadcasts input to the shape shape. Equivalent to calling input.expand(shape).
            /// </summary>
            public Tensor broadcast_to(params long[] shape)
            {
                unsafe {
                    fixed (long* psizes = shape) {
                        var res = LibTorchSharp.THSTensor_broadcast_to(Handle, (IntPtr)psizes, shape.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public Tensor movedim(long[] source, long[] destination)
            {
                unsafe {
                    fixed (long* psource = source, pdest = destination) {
                        var res = LibTorchSharp.THSTensor_movedim(Handle, (IntPtr)psource, source.Length, (IntPtr)pdest, destination.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public Tensor moveaxis(long[] source, long[] destination) => movedim(source, destination);

            /// <summary>
            /// Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
            /// </summary>
            public Tensor randn_out(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_randn_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
            /// </summary>
            public Tensor rand_out(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_rand_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }
            /// <summary>
            /// Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
            /// </summary>
            public Tensor randint_out(long high, long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_randint_out(high, (IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1) .
            /// </summary>
            public Tensor rand_like(ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_rand_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_rand_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            /// Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
            /// </summary>
            public Tensor randn_like(ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_randn_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_randn_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            /// Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly in the range [low,high).
            /// </summary>
            public Tensor randint_like(long low, long high, ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_randint_like(Handle, low, high, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_randint_like(Handle, low, high, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            /// Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
            /// </summary>
            public Tensor randperm_out(long n)
            {
                var res = LibTorchSharp.THSTensor_randperm_out(n, Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Draws binary random numbers (0 or 1) from a Bernoulli distribution.
            /// The input tensor should be a tensor containing probabilities to be used for drawing the binary random number.
            /// </summary>
            /// <param name="generator">A pseudorandom number generator for sampling</param>
            /// <returns></returns>
            public Tensor bernoulli(torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_bernoulli(Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
            /// </summary>
            /// <param name="num_samples">number of samples to draw</param>
            /// <param name="replacement">whether to draw with replacement or not</param>
            /// <param name="generator">A pseudorandom number generator for sampling</param>
            /// <returns></returns>
            public Tensor multinomial(long num_samples, bool replacement = false, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_multinomial(Handle, num_samples, replacement, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input
            /// </summary>
            /// <param name="generator">Optional random number generator</param>
            public Tensor poisson(torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_poisson(Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            /// <summary>
            /// Fills each location of the input tensor with an independent sample from Bernoulli(p)
            /// </summary>
            /// <param name="p">Probability</param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <returns></returns>
            public Tensor bernoulli_(double p = 0.5, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_bernoulli_0(Handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills each location of the input tensor with an independent sample from Bernoulli(p)
            /// </summary>
            /// <param name="p">Probability tensor</param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <returns></returns>
            public Tensor bernoulli_(Tensor p, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_bernoulli_1(Handle, p.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor binomial(Tensor prob, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_binomial(Handle, prob.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the tensor with numbers drawn from the Cauchy distribution.
            /// </summary>
            /// <returns></returns>
            public Tensor cauchy_(double median = 0.0, double sigma = 1.0, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_cauchy_(Handle, median, sigma, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the input tensor with elements drawn from the exponential distribution
            /// </summary>
            /// <param name="lambda"></param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <returns></returns>
            public Tensor exponential_(double lambda = 1.0, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_exponential_(Handle, lambda, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the input tensor with elements drawn from a geometric distribution.
            /// </summary>
            /// <param name="p"></param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <returns></returns>
            public Tensor geometric_(double p, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_geometric_(Handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the input tensor with elements samples from the normal distribution parameterized by mean and std.
            /// </summary>
            /// <param name="mean">The mean of the underlying normal distribution.</param>
            /// <param name="std">The standard deviation of the underlying normal distribution.</param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <returns></returns>
            public Tensor normal_(double mean = 0.0, double std = 1.0, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_normal_(Handle, mean, std, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the input tensor with numbers samples from the log-normal distribution parameterized by the given mean and standard deviation.
            /// </summary>
            /// <param name="mean">The mean of the underlying normal distribution.</param>
            /// <param name="std">The standard deviation of the underlying normal distribution.</param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <remarks>Note that mean and std are the mean and standard deviation of the underlying normal distribution, and not of the returned distribution.</remarks>
            /// <returns></returns>
            public Tensor log_normal_(double mean = 0.0, double std = 1.0, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_log_normal_(Handle, mean, std, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the input tensor with numbers sampled from the discrete uniform distribution over [from, to - 1].
            /// If not specified, the values are usually only bounded by the input tensor’s data type.
            /// </summary>
            /// <param name="from">The lower bound.</param>
            /// <param name="to">The uppoer bound.</param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <remarks>
            /// For floating point types, if unspecified, the range will be [0, 2^mantissa] to ensure that every value is representable. For example, torch.tensor(1, dtype=torch.double).random_() will be uniform in [0, 2^53].
            /// </remarks>
            public Tensor random_(double from, double to, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_random_(Handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Fills the input tensor with numbers sampled from the continuous uniform distribution:
            /// </summary>
            /// <param name="from">Lower bound.</param>
            /// <param name="to">Upper bound</param>
            /// <param name="generator">Optional random-number generator.</param>
            /// <returns></returns>
            public Tensor uniform_(double from, double to, torch.Generator? generator = null)
            {
                var res = LibTorchSharp.THSTensor_uniform_(Handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Mutates the tensor to be filled with with values from interval [start, end) and
            /// common difference step, starting from start.
            /// </summary>
            public Tensor arange_out(Scalar start, Scalar stop, Scalar step)
            {
                var res = LibTorchSharp.THSTensor_arange_out(start.Handle, stop.Handle, step.Handle, Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a view of the original tensor with its dimensions permuted.
            /// </summary>
            /// <param name="permutation">The desired ordering of dimensions</param>
            public Tensor permute(params long[] permutation)
            {
                unsafe {
                    fixed (long* pPermutation = permutation) {
                        var res = LibTorchSharp.THSTensor_permute(Handle, (IntPtr)pPermutation, permutation.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Mutates the tensor to have the given size with all values set to 1
            /// </summary>
            public Tensor ones(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_ones_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Create a new tensor filled with ones
            /// </summary>
            private Tensor new_ones(ReadOnlySpan<long> size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.ones(size, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new tensor filled with ones
            /// </summary>
            private Tensor new_ones(long[] size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.ones(size, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 1-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_ones(new long[] { size }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 2-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long rows, long columns, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_ones(new long[] { rows, columns }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 3-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_ones(new long[] { dim0, dim1, dim2 }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 4-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_ones(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Mutates the tensor to have the given size with all values set to 0
            /// </summary>
            public Tensor zeros(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_zeros_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Fills the tensor with zeros.
            /// </summary>

            public Tensor zero_()
            {
                return zeros(shape);
            }

            /// <summary>
            /// Create a new tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long[] size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.zeros(size, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(ReadOnlySpan<long> size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.zeros(size, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 1-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_zeros(new long[] { size }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 2-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long rows, long columns, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_zeros(new long[] { rows, columns }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 3-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_zeros(new long[] { dim0, dim1, dim2 }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 4-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_zeros(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Returns a tensor filled with the scalar value 0, with the same size as input.
            /// </summary>
            public Tensor zeros_like(ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_zeros_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_zeros_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            /// Returns a tensor filled with the scalar value 1, with the same size as input.
            /// </summary>
            public Tensor ones_like(ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_ones_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_ones_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            /// Create a new tensor filled with empty
            /// </summary>
            public Tensor new_empty(long[] size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.empty(size, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new tensor filled with empty
            /// </summary>
            public Tensor new_empty(ReadOnlySpan<long> size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.empty(size, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 1-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_empty(new long[] { size }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 2-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long rows, long columns, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_empty(new long[] { rows, columns }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 3-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_empty(new long[] { dim0, dim1, dim2 }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 4-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_empty(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad);
            }

            /// <summary>
            /// Mutates the tensor to have the given size with all values uninitialized
            /// </summary>
            public Tensor empty(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_empty_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Returns an uninitialized tensor with the same size as input.
            /// </summary>
            public Tensor empty_like(ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_empty_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_empty_like(Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            /// Mutates the tensor to have the given size with all values uninitialized
            /// </summary>
            public Tensor full(long[] sizes, Scalar value)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_full_out((IntPtr)psizes, sizes.Length, value.Handle, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Mutates the tensor to have the given size with all values uninitialized
            /// </summary>
            public Tensor full(ReadOnlySpan<long> sizes, Scalar value)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = LibTorchSharp.THSTensor_full_out((IntPtr)psizes, sizes.Length, value.Handle, Handle);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Create a new tensor filled with a given value
            /// </summary>
            public Tensor new_full(long[] size, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.full(size, value, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new tensor filled with a given value
            /// </summary>
            public Tensor new_full(ReadOnlySpan<long> size, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.full(size, value, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 1-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long size, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_full(new long[] { size }, value, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 2-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long rows, long columns, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_full(new long[] { rows, columns }, value, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 3-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long dim0, long dim1, long dim2, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_full(new long[] { dim0, dim1, dim2 }, value, dtype, device, requires_grad);
            }

            /// <summary>
            /// Create a new 4-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                return new_full(new long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requires_grad);
            }


            /// <summary>
            /// Returns a tensor with the same size as input filled with 'value.'
            /// </summary>
            public Tensor full_like(Scalar value, ScalarType? dtype = null, torch.Device? device = null, bool requires_grad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = LibTorchSharp.THSTensor_full_like(Handle, value.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = LibTorchSharp.THSTensor_full_like(Handle, value.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                }
                if (result == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(result);
            }

            public Tensor detach()
            {
                var res = LibTorchSharp.THSTensor_detach(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor detach_()
            {
                var res = LibTorchSharp.THSTensor_detach_(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Mutates the tensor into a 2-D tensor with ones on the diagonal and zeros elsewhere.
            /// </summary>
            public Tensor eye(long rows, long columns)
            {
                var res = LibTorchSharp.THSTensor_eye_out(rows, columns, Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            /// <summary>
            /// Writes all values from the tensor src into the input tensor at the indices specified in the index tensor. For each
            /// value in src, its output index is specified by its index in src for dimension != dim and by the #
            /// corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter(long dim, Tensor index, Tensor src)
            {
                var res = LibTorchSharp.THSTensor_scatter(Handle, dim, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Adds all values from the tensor other into the input tensor at the indices specified in the index tensor in a similar fashion as scatter_().
            /// For each value in src, it is added to an index in the input tensor which is specified by its index in src for dimension != dim and by the
            /// corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter_(long dim, Tensor index, Tensor src)
            {
                var res = LibTorchSharp.THSTensor_scatter_(Handle, dim, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Adds all values from the tensor other into the input tensor at the indices specified in the index tensor in a similar fashion as scatter_().
            /// For each value in src, it is added to an index in the input tensor which is specified by its index in src for dimension != dim and by the
            /// corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter_add(long dim, Tensor index, Tensor src)
            {
                var res = LibTorchSharp.THSTensor_scatter_add(Handle, dim, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Writes all values from the tensor src into the input tensor at the indices specified in the index tensor. For each
            /// value in src, its output index is specified by its index in src for dimension != dim and by the #
            /// corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter_add_(long dim, Tensor index, Tensor src)
            {
                var res = LibTorchSharp.THSTensor_scatter_add_(Handle, dim, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            public Tensor diagonal_scatter(Tensor src, long offset = 0L, long dim1 = 0L, long dim2 = 1L)
            {
                var res = LibTorchSharp.THSTensor_diagonal_scatter(Handle, src.Handle, offset, dim1, dim2);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Embeds the values of the src tensor into input at the given index. This function returns a tensor with fresh storage; it does not create a view.
            /// </summary>
            /// <param name="src">The tensor to embed into 'this'</param>
            /// <param name="dim">The dimension to insert the slice into</param>
            /// <param name="index">The index to select with</param>
            /// <remarks>This function returns a tensor with fresh storage; it does not create a view.</remarks>
            public Tensor select_scatter(Tensor src, long dim, long index)
            {
                var res = LibTorchSharp.THSTensor_select_scatter(Handle, src.Handle, dim, index);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Embeds the values of the src tensor into input at the given dimension.
            /// </summary>
            /// <param name="src">The tensor to embed into 'this'.</param>
            /// <param name="dim">The dimension to insert the slice into</param>
            /// <param name="start">The start index of where to insert the slice</param>
            /// <param name="end">The end index of where to insert the slice</param>
            /// <param name="step">How many elements to skip</param>
            public unsafe Tensor slice_scatter(Tensor src, long dim = 0L, long? start = null, long? end = null, long step = 1L)
            {
                var _start = start.HasValue ? new long[] { start.Value } : null;
                var _end = end.HasValue ? new long[] { end.Value } : null;
                fixed (long* pstart = _start, pend = _end) {
                    var res = LibTorchSharp.THSTensor_slice_scatter(Handle, src.Handle, dim, (IntPtr)pstart, (IntPtr)pend, step);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Gathers values along an axis specified by dim.
            /// </summary>
            public Tensor gather(long dim, Tensor index)
            {
                var res = LibTorchSharp.THSTensor_gather(Handle, dim, index.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Reverse the order of a n-D tensor along given axis in dims.
            /// </summary>
            public Tensor flip(params long[] dims)
            {
                unsafe {
                    fixed (long* psizes = dims) {
                        var res = LibTorchSharp.THSTensor_flip(Handle, (IntPtr)psizes, dims.Length);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Flip tensor in the left/right direction, returning a new tensor.
            /// </summary>
            public Tensor fliplr()
            {
                var res = LibTorchSharp.THSTensor_fliplr(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Flip tensor in the up/down direction, returning a new tensor.
            /// </summary>
            public Tensor flipud()
            {
                var res = LibTorchSharp.THSTensor_flipud(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the mean of the values in input, ignoring NaN values.
            /// </summary>
            public Tensor nanmean(int? dim = null, bool keepdim = false, ScalarType? dtype = null)
            {
                var d = (dim is null) ? -1 : dim.Value;
                var t = (dtype is null) ? this.dtype : dtype.Value;
                var res = LibTorchSharp.THSTensor_nanmean(Handle, d, keepdim, (sbyte)t);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the median of the values in input, ignoring NaN values.
            /// </summary>
            public Tensor nanmedian()
            {
                var res = LibTorchSharp.THSTensor_nanmedian(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the sum of all elements in the input tensor, treating NaN as zero.
            /// </summary>
            public Tensor nansum()
            {
                var res = LibTorchSharp.THSTensor_nansum(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
            /// By default, NaN`s are replaced with zero, positive infinity is replaced with the greatest finite value representable by input’s dtype,
            /// and negative infinity is replaced with the least finite value representable by input’s dtype.
            /// </summary>
            public Tensor nan_to_num(double nan = 0d, double? posinf = null, double? neginf = null)
            {
                var _nan = new double[] { nan };
                var _posinf = posinf.HasValue ? new double[] { posinf.Value } : null;
                var _neginf = neginf.HasValue ? new double[] { neginf.Value } : null;
                unsafe {
                    fixed (double* pnan = _nan, pposinf = _posinf, pneginf = _neginf) {
                        var res =
                            LibTorchSharp.THSTensor_nan_to_num(Handle, (IntPtr)pnan, (IntPtr)pposinf, (IntPtr)pneginf);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Return the next floating-point value after input towards other, elementwise.
            /// </summary>
            public Tensor nextafter(Tensor other)
            {
                var res = LibTorchSharp.THSTensor_nextafter(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor that is a narrowed version of the input along one dimension. The
            /// dimension is input from start to start + length. The
            /// returned tensor and the input tensor share the same underlying storage.
            /// </summary>
            public Tensor narrow(long dim, long start, long length)
            {
                var res = LibTorchSharp.THSTensor_narrow(Handle, dim, start, length);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tensor containing the indices of all non-zero elements of input.
            /// Each row in the result contains the indices of a non-zero element in input.
            /// The result is sorted lexicographically, with the last index changing the fastest (C-style).
            /// </summary>
            public Tensor nonzero()
            {
                var res = LibTorchSharp.THSTensor_nonzero(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public IList<Tensor> nonzero_as_list()
            {
                var res = LibTorchSharp.THSTensor_nonzero(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }

                var t = new Tensor(res);
                return t.chunk(t.shape[1], dim: 1);
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll(long shifts, long? dims = null)
            {
                if (dims.HasValue) {
                    return _roll(stackalloc long[1] { shifts }, new long[1] { dims.Value });
                } else {
                    return _roll(stackalloc long[1] { shifts }, null);
                }
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll((long, long) shifts, (long, long) dims)
            {
                return _roll(stackalloc long[2] { shifts.Item1, shifts.Item2 }, new long[2] { dims.Item1, dims.Item2 });
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll(long[] shifts, long[] dims) => _roll(shifts, dims);

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll(long[] shifts) => _roll(shifts, new long[] { 0 });

            /// <summary>
            /// Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
            /// Rotation direction is from the first towards the second axis if k is greater than 0,
            /// and from the second towards the first for k less than 0.
            /// </summary>
            /// <param name="k">The number of times to rotate.</param>
            /// <param name="dims">Axes to rotate</param>
            public Tensor rot90(long k = 1, (long, long)? dims = null)
            {
                if (!dims.HasValue) {
                    dims = (0, 1);
                }

                var res = LibTorchSharp.THSTensor_rot90(Handle, k, dims.Value.Item1, dims.Value.Item2);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll((long, long, long) shifts, (long, long, long) dims)
            {
                return _roll(stackalloc long[3] { shifts.Item1, shifts.Item2, shifts.Item3 }, new long[3] { dims.Item1, dims.Item2, dims.Item3 });
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll(ReadOnlySpan<long> shifts, ReadOnlySpan<long> dims = default)
            {
                return _roll(shifts, dims);
            }

            private unsafe Tensor _roll(ReadOnlySpan<long> shifts, ReadOnlySpan<long> dims)
            {
                var dmLen = dims.Length;

                fixed (long* sh = shifts, dm = (dmLen == 0) ? null : dims) {
                    var res =
                        LibTorchSharp.THSTensor_roll(Handle, (IntPtr)sh, shifts.Length, (IntPtr)dm, dmLen);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Returns a new tensor that is a sliced version of the input along one dimension. The
            /// dimension is input from start to finish-1. The
            /// returned tensor and the input tensor share the same underlying storage.
            /// </summary>
            public Tensor slice(long dim, long start, long finish, long step)
            {
                if (step < 1) throw new ArgumentException($"step is {step}, but it should always be positive.");
                var res = LibTorchSharp.THSTensor_slice(Handle, dim, start, finish, step);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with a dimension of size one inserted at the specified position.
            /// The returned tensor shares the same underlying data with this tensor.
            /// </summary>
            public Tensor unsqueeze(long dim)
            {
                var res = LibTorchSharp.THSTensor_unsqueeze(Handle, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with a dimension of size one inserted at the specified position.
            /// The returned tensor shares the same underlying data with this tensor.
            /// </summary>
            public Tensor unsqueeze_(long dim)
            {
                var res = LibTorchSharp.THSTensor_unsqueeze_(Handle, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Return a tensor of elements selected from either x or y, depending on condition.
            /// </summary>
            /// <param name="condition"></param>
            /// <param name="y"></param>
            /// <returns></returns>
            /// <exception cref="ArgumentException"></exception>
            public Tensor where(Tensor condition, Tensor y)
            {
                if (condition.dtype != ScalarType.Bool) throw new ArgumentException("The condition to 'where' must be a boolean tensor.");

                var res = LibTorchSharp.THSTensor_where(condition.Handle, this.Handle, y.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            // Operators overloading

            public static Tensor operator ==(Tensor left, Tensor right)
            {
                return left.eq(right);
            }

            public static Tensor operator ==(Tensor left, Scalar right)
            {
                return left.eq(right);
            }

            public static Tensor operator ==(Scalar left, Tensor right)
            {
                return right.eq(left);
            }

            public static Tensor operator !=(Tensor left, Tensor right)
            {
                return left.ne(right);
            }

            public static Tensor operator !=(Tensor left, Scalar right)
            {
                return left.ne(right);
            }

            public static Tensor operator !=(Scalar left, Tensor right)
            {
                return right.ne(left);
            }

            public static Tensor operator <(Tensor left, Tensor right)
            {
                return left.lt(right);
            }

            public static Tensor operator <(Tensor left, Scalar right)
            {
                return left.lt(right);
            }

            public static Tensor operator <(Scalar left, Tensor right)
            {
                return right.gt(left);
            }

            public static Tensor operator <=(Tensor left, Tensor right)
            {
                return left.le(right);
            }

            public static Tensor operator <=(Tensor left, Scalar right)
            {
                return left.le(right);
            }

            public static Tensor operator <=(Scalar left, Tensor right)
            {
                return right.ge(left);
            }

            public static Tensor operator >(Tensor left, Tensor right)
            {
                return left.gt(right);
            }

            public static Tensor operator >(Tensor left, Scalar right)
            {
                return left.gt(right);
            }

            public static Tensor operator >(Scalar left, Tensor right)
            {
                return right.lt(left);
            }

            public static Tensor operator >=(Tensor left, Tensor right)
            {
                return left.ge(right);
            }

            public static Tensor operator >=(Tensor left, Scalar right)
            {
                return left.ge(right);
            }

            public static Tensor operator >=(Scalar left, Tensor right)
            {
                return right.le(left);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(byte value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(sbyte value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(short value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(int value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(long value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(float value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(double value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(bool value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor((float, float) value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when assigning a .NET numeric value to an index of a Tensor.
            /// </summary>
            /// <param name="value">The numeric value.</param>
            public static implicit operator Tensor(System.Numerics.Complex value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(byte[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(sbyte[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(short[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(int[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(long[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(float[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(double[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(bool[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor((float, float)[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// Useful when creating a tensor from a .NET array of numbers.
            /// </summary>
            /// <param name="value">The numeric value array.</param>
            public static implicit operator Tensor(System.Numerics.Complex[] value)
            {
                return torch.tensor(value);
            }

            /// <summary>
            /// This is only here in order to help the C# compiler make the right choice vis-a-vis
            /// implicit conversions.
            /// </summary>
            public static implicit operator Tensor(Scalar scalar)
            {
                throw new InvalidOperationException("Implicit conversion from Scalar to Tensor -- this should never be invoked, the operator is only here to guide the compiler's overload resolution.");
            }

            // Specifically added to make F# look good.
            public static Tensor op_MinusMinusGreater(Tensor t, torch.nn.Module<Tensor, Tensor> m) => m.call(t);

            public override string ToString() => ToMetadataString();

            /// <summary>
            /// Tensor-specific ToString(), for backward-compat with pre-0.96.4 versions.
            /// </summary>
            /// <returns></returns>
            public string ToString(bool disamb,
                                   string? fltFormat = null,
                                   int? width = null,
                                   CultureInfo? cultureInfo = null,
                                   string? newLine = null) => disamb ? ToString(torch.TensorStringStyle, fltFormat, width, cultureInfo, newLine) : ToMetadataString();

            /// <summary>
            /// Tensor-specific ToString()
            /// </summary>
            /// <param name="style">
            /// The style to use -- either 'default,' 'metadata,' 'julia,' or 'numpy'
            /// </param>
            /// <param name="fltFormat">The floating point format to use for each individual number.</param>
            /// <param name="width">The line width to enforce</param>
            /// <param name="cultureInfo">The culture, which affects how numbers are formatted.</param>
            /// <param name="newLine">The newline string to use, defaults to system default.</param>
            /// <returns></returns>
            public string ToString(TensorStringStyle style,
                                   string? fltFormat = null,
                                   int? width = null,
                                   CultureInfo? cultureInfo = null,
                                   string? newLine = null)
            {
                var w = width.HasValue ? width.Value : torch.lineWidth;
                var nl = newLine is null ? torch.newLine : newLine;
                var fmt = fltFormat is null ? torch.floatFormat : fltFormat;

                if (String.IsNullOrEmpty(newLine))
                    newLine = Environment.NewLine;

                if (device_type == DeviceType.META)
                    return ToMetadataString();

                return style switch {
                    TensorStringStyle.Default => ToString(torch.TensorStringStyle, fltFormat, width, cultureInfo, nl),
                    TensorStringStyle.Metadata => ToMetadataString(),
                    TensorStringStyle.Julia => ToJuliaString(fmt, w, cultureInfo, nl),
                    TensorStringStyle.Numpy => ToNumpyString(this, ndim, true, fmt, cultureInfo, nl),
                    _ => throw new InvalidEnumArgumentException($"Unsupported tensor string style: {style}")
                };
            }

            /// <summary>
            /// Get a string representation of the tensor.
            /// </summary>
            private string ToMetadataString()
            {
                if (Handle == IntPtr.Zero) return "";

                var sb = new StringBuilder("[");

                var n = Dimensions;
                if (n == 0) {
                    sb.Append(']');
                } else {
                    for (var i = 0; i < n; i++) {
                        sb.Append(size(i));
                        if (i + 1 < n)
                            sb.Append("x");
                    }

                    sb.Append("]");
                }
                sb.Append($", type = {dtype}, device = {device}");

                return sb.ToString();
            }

            private static string ToNumpyString(Tensor t, long mdim, bool isFCreate, string fltFormat, CultureInfo? cultureInfo, string newLine)
            {
                var actualCulturInfo = cultureInfo ?? CultureInfo.CurrentCulture;

                var dim = t.dim();
                if (t.size().Length == 0) return "";
                var sb = new StringBuilder(isFCreate ? string.Join("", Enumerable.Repeat(' ', (int)(mdim - dim))) : "");
                sb.Append('[');
                var currentSize = t.size()[0];
                if (dim == 1) {
                    if (currentSize <= 6) {
                        for (var i = 0; i < currentSize - 1; i++) {
                            PrintValue(sb, t.dtype, t[i].ToScalar(), fltFormat, actualCulturInfo);
                            sb.Append(' ');
                        }

                        PrintValue(sb, t.dtype, t[currentSize - 1].ToScalar(), fltFormat, actualCulturInfo);
                    } else {
                        for (var i = 0; i < 3; i++) {
                            PrintValue(sb, t.dtype, t[i].ToScalar(), fltFormat, actualCulturInfo);
                            sb.Append(' ');
                        }

                        sb.Append("... ");

                        for (var i = currentSize - 3; i < currentSize - 1; i++) {
                            PrintValue(sb, t.dtype, t[i].ToScalar(), fltFormat, actualCulturInfo);
                            sb.Append(' ');
                        }

                        PrintValue(sb, t.dtype, t[currentSize - 1].ToScalar(), fltFormat, actualCulturInfo);
                    }
                } else {
                    var newline = string.Join("", Enumerable.Repeat(newLine, (int)dim - 1).ToList());

                    if (currentSize == 1) {
                        sb.Append(ToNumpyString(t[0], mdim, false, fltFormat, cultureInfo, newLine));
                    } else if (currentSize <= 6) {
                        sb.Append(ToNumpyString(t[0], mdim, false, fltFormat, cultureInfo, newLine));
                        sb.Append(newline);
                        for (var i = 1; i < currentSize - 1; i++) {
                            sb.Append(ToNumpyString(t[i], mdim, true, fltFormat, cultureInfo, newLine));
                            sb.Append(newline);
                        }

                        sb.Append(ToNumpyString(t[currentSize - 1], mdim, true, fltFormat, cultureInfo, newLine));
                    } else {
                        sb.Append(ToNumpyString(t[0], mdim, false, fltFormat, cultureInfo, newLine));
                        sb.Append(newline);
                        for (var i = 1; i < 3; i++) {
                            sb.Append(ToNumpyString(t[i], mdim, true, fltFormat, cultureInfo, newLine));
                            sb.Append(newline);
                        }

                        sb.Append(string.Join("", Enumerable.Repeat(' ', (int)(mdim - dim))));
                        sb.Append(" ...");
                        sb.Append(newline);

                        for (var i = currentSize - 3; i < currentSize - 1; i++) {
                            sb.Append(ToNumpyString(t[i], mdim, true, fltFormat, cultureInfo, newLine));
                            sb.Append(newline);
                        }

                        sb.Append(ToNumpyString(t[currentSize - 1], mdim, true, fltFormat, cultureInfo, newLine));
                    }
                }

                sb.Append("]");
                return sb.ToString();
            }

            /// <summary>
            /// Get a verbose string representation of a tensor.
            /// </summary>
            /// <param name="fltFormat">The format string to use for floating point values.</param>
            /// <param name="width">The width of each line of the output string.</param>
            /// <param name="cultureInfo">The CulturInfo to use when formatting the text</param>
            /// <param name="newLine">The newline string to use, defaults to system default.</param>
            private string ToJuliaString(string fltFormat, int width, CultureInfo? cultureInfo, string newLine)
            {
                var actualCulturInfo = cultureInfo ?? CultureInfo.CurrentCulture;

                var builder = new StringBuilder(this.ToMetadataString());

                if (Dimensions == 0) {

                    builder.Append(", value = ");
                    PrintValue(builder, dtype, this.ToScalar(), fltFormat, actualCulturInfo);

                } else if (Dimensions == 1) {

                    var row = new List<string>();
                    BuildRow(row, this, width, fltFormat, actualCulturInfo);

                    var appendEllipsis = row.Count < shape[0];

                    builder.Append(newLine);
                    PrintOneRow(row, row.Select(str => str.Length).ToArray(), new bool[shape[0]], fltFormat, builder, this, appendEllipsis, newLine);

                } else if (Dimensions == 2) {

                    builder.Append(newLine);
                    PrintTwoDimensions(fltFormat, width, builder, this, actualCulturInfo, newLine);

                } else {
                    var indices = new List<TensorIndex>();
                    RecursivePrintDimensions(0, indices, fltFormat, width, builder, actualCulturInfo, newLine);
                }

                return builder.ToString();
            }

            private void RecursivePrintDimensions(int dim, IEnumerable<TensorIndex> indices, string fltFormat, int width, StringBuilder builder, CultureInfo cultureInfo, string newLine)
            {
                if (dim == Dimensions - 3) {
                    // We're at the third-last dimension. This is where we can print out the last two dimensions.

                    for (int i = 0; i < shape[dim]; i++) {

                        var idxs = indices.Append(TensorIndex.Single(i)).Append(TensorIndex.Ellipsis).Append(TensorIndex.Ellipsis).ToArray();
                        var str = IndicesToString(idxs);
                        builder.Append(newLine).Append($"{str} =").Append(newLine);
                        var slice = this.index(idxs);
                        PrintTwoDimensions(fltFormat, width, builder, slice, cultureInfo, newLine);
                    }
                } else {

                    for (int i = 0; i < shape[dim]; i++) {

                        RecursivePrintDimensions(dim + 1, indices.Append(TensorIndex.Single(i)), fltFormat, width, builder, cultureInfo, newLine);
                    }
                }
            }

            private string IndicesToString(IList<TensorIndex> indices)
            {
                var builder = new StringBuilder("[");
                for (int i = 0; i < indices.Count(); i++) {

                    if (i > 0) builder.Append(',');

                    if (indices[i].kind == TensorIndex.Kind.Ellipsis) {
                        builder.Append("..");
                    } else if (indices[i].kind == TensorIndex.Kind.Single) {
                        builder.Append(indices[i].startIndexOrBoolOrSingle);
                    }
                }
                return builder.Append(']').ToString();
            }

            private static void PrintTwoDimensions(string fltFormat, int width, StringBuilder builder, Tensor t, CultureInfo cultureInfo, string newLine)
            {
                // TODO: This code will align the first digits of each column, taking a leading '-' into account.
                //       An alternative would be to align periods, or to align the last character of each column.
                var rows = new List<List<string>>();
                var rowCount = t.shape[0];
                var colCount = t.shape[1];

                var columnSpace = new int[colCount];
                var hasMinus = new bool[colCount];



                for (int i = 0; i < rowCount; i++) {
                    var row = new List<string>();
                    BuildRow(row, t[i], width, fltFormat, cultureInfo);
                    rows.Add(row);
                }

                var shortestRow = rows.Select(r => r.Count).Min();

                var appendEllipsis = shortestRow < t.shape[1];

                for (int i = 0; i < rowCount; i++) {

                    var row = rows[i];

                    for (int j = 0; j < shortestRow; j++) {
                        hasMinus[j] = hasMinus[j] || row[j].StartsWith('-');
                        if (row[j].Length > columnSpace[j])
                            columnSpace[j] = row[j].Length;
                    }
                }

                for (int i = 0; i < rowCount; i++) {
                    PrintOneRow(rows[i].Take(shortestRow).ToList(), columnSpace, hasMinus, fltFormat, builder, t[i], appendEllipsis, newLine);
                }
            }

            private const string ellipsis = "...";

            private static void PrintOneRow(IList<string> row, int[] space, bool[] hasMinus, string fltFormat, StringBuilder builder, Tensor rowTensor, bool appendEllipsis, string newLine)
            {
                for (var i = 0; i < row.Count; i++) {
                    var pad = space[i] - row[i].Length;
                    builder.Append(' ');
                    //if (hasMinus[i] && !row[i].StartsWith('-')) { pad--; builder.Append(' '); }

                    for (int j = 0; j < pad; j++)
                        builder.Append(' ');

                    builder.Append(row[i]);
                }

                if (appendEllipsis) {
                    builder.Append(' ').Append(ellipsis);
                }
                builder.Append(newLine);
            }

            private static void BuildRow(List<string> row, Tensor t, int width, string fltFormat, CultureInfo cultureInfo)
            {
                var type = t.dtype;
                var endingWidth = ellipsis.Length + 1;

                for (int i = 0; i < t.shape[0]; i++) {

                    var builder = new StringBuilder();
                    PrintValue(builder, type, t[i].ToScalar(), fltFormat, cultureInfo);

                    var str = builder.ToString();

                    if (width - str.Length - endingWidth < 0) {
                        break;
                    }

                    row.Add(str);
                    width -= str.Length + 1;
                }
            }

            private static void PrintValue(StringBuilder builder, ScalarType type, Scalar value, string fltFormat, CultureInfo cultureInfo)
            {
                switch (type) {
                case ScalarType.Byte:
                    builder.Append(value.ToByte());
                    break;
                case ScalarType.Int8:
                    builder.Append(value.ToSByte());
                    break;
                case ScalarType.Int16:
                    builder.Append(value.ToInt16());
                    break;
                case ScalarType.Int32:
                    builder.Append(value.ToInt32());
                    break;
                case ScalarType.Int64:
                    builder.Append(value.ToInt64());
                    break;
                case ScalarType.Bool:
                    builder.Append(value.ToBoolean().ToString(cultureInfo));
                    break;
                case ScalarType.Float16:
                    builder.Append(value.ToSingle().ToString(fltFormat, cultureInfo));
                    break;
                case ScalarType.Float32:
                    builder.Append(value.ToSingle().ToString(fltFormat, cultureInfo));
                    break;
                case ScalarType.Float64:
                    builder.Append(value.ToDouble().ToString(fltFormat, cultureInfo));
                    break;
                case ScalarType.ComplexFloat32:
                    var val1 = value.ToComplexFloat32();
                    if (val1.Real != 0.0f || val1.Imaginary == 0.0f) {
                        builder.Append(val1.Real.ToString(fltFormat, cultureInfo));
                        if (val1.Imaginary != 0.0f)
                            builder.Append('+');
                    }
                    if (val1.Imaginary != 0.0f)
                        builder.Append(val1.Imaginary.ToString(fltFormat, cultureInfo)).Append('i');
                    break;
                case ScalarType.ComplexFloat64:
                    var val2 = value.ToComplexFloat64();
                    if (val2.Real != 0.0f || val2.Imaginary == 0.0f) {
                        builder.Append(val2.Real.ToString(fltFormat, cultureInfo));
                        if (val2.Imaginary != 0.0f)
                            builder.Append('+');
                    }
                    if (val2.Imaginary != 0.0f)
                        builder.Append(val2.Imaginary.ToString(fltFormat, cultureInfo)).Append('i');
                    break;
                }
            }

            public object tolist()
            {
                if (this.shape.Length == 0) {
                    return this.ToScalar();
                }

                var result = new System.Collections.ArrayList();
                if (shape.Length == 1) {
                    for (long idx = 0; idx < shape[0]; idx++) {
                        result.Add(this[idx].ToScalar());
                    }
                } else {
                    for (long idx = 0; idx < shape[0]; idx++) {
                        result.Add(this[idx].tolist());
                    }
                }

                return result;
            }

            public static explicit operator float(Tensor value) => value.ToSingle();
            public static explicit operator double(Tensor value) => value.ToDouble();
            public static explicit operator sbyte(Tensor value) => value.ToSByte();
            public static explicit operator byte(Tensor value) => value.ToByte();
            public static explicit operator short(Tensor value) => value.ToInt16();
            public static explicit operator int(Tensor value) => value.ToInt32();
            public static explicit operator long(Tensor value) => value.ToInt64();
            public static explicit operator bool(Tensor value) => value.ToBoolean();


            /// <summary>
            /// Create a block diagonal matrix from provided tensors.
            /// </summary>
            /// <param name="tensors">One or more tensors with 0, 1, or 2 dimensions.</param>
            /// <returns></returns>
            public static Tensor block_diag(params Tensor[] tensors) => torch.block_diag(tensors);

            /// <summary>
            /// Returns a 1-dimensional view of an input tensor with zero dimensions. Input tensors with one or more dimensions are returned as-is.
            /// </summary>
            /// <returns></returns>
            public Tensor atleast_1d()
            {
                var res = LibTorchSharp.THSTensor_atleast_1d(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a 2-dimensional view of an input tensor with zero dimensions. Input tensors with two or more dimensions are returned as-is.
            /// </summary>
            /// <returns></returns>
            public Tensor atleast_2d()
            {
                var res = LibTorchSharp.THSTensor_atleast_2d(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a 3-dimensional view of an input tensor with zero dimensions. Input tensors with three or more dimensions are returned as-is.
            /// </summary>
            /// <returns></returns>
            public Tensor atleast_3d()
            {
                var res = LibTorchSharp.THSTensor_atleast_3d(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Short-time Fourier transform (STFT).
            /// </summary>
            /// <param name="n_fft">size of Fourier transform</param>
            /// <param name="hop_length">the distance between neighboring sliding window frames</param>
            /// <param name="win_length">the size of window frame and STFT filter.</param>
            /// <param name="window">The optional window function.</param>
            /// <param name="center">whether to pad input on both sides so that the tt-th frame is centered at time t * hop_length</param>
            /// <param name="pad_mode"> controls the padding method used when center is True</param>
            /// <param name="normalized">controls whether to return the normalized STFT results</param>
            /// <param name="onesided">controls whether to return half of results to avoid redundancy for real inputs.</param>
            /// <param name="return_complex">whether to return a complex tensor, or a real tensor with an extra last dimension for the real and imaginary components.</param>
            /// <returns></returns>
            public Tensor stft(long n_fft, long hop_length = -1, long win_length = -1, Tensor? window = null, bool center = true, PaddingModes pad_mode = PaddingModes.Reflect, bool normalized = false, bool? onesided = null, bool? return_complex = null)
            {
                IntPtr _input = Handle;

                long _onesided = -1; // encoding of null
                if (onesided.HasValue) {
                    _onesided = (onesided.Value ? 1 : 0);
                }
                bool _return_complex = return_complex.HasValue ? return_complex.Value : is_complex();

                if (center) {
                    long signalDim = dim();
                    long pad = n_fft / 2;
                    var extendedShape = Enumerable.Repeat<long>(1, (int)(3 - signalDim)).Concat(shape).ToArray();

                    unsafe {
                        var paddedInput = torch.nn.functional.pad(view(extendedShape), stackalloc long[] { pad, pad }, pad_mode);
                        var paddedShape = paddedInput.shape;
                        var _shape = new long[signalDim];
                        for (int i = 0; i < signalDim; i++) {
                            _shape[i] = paddedShape[paddedShape.Length + i - signalDim];
                        }
                        paddedInput = paddedInput.view(_shape);
                        _input = paddedInput.Handle;
                    }
                }

                IntPtr _window = (window is null) ? IntPtr.Zero : window.Handle;
                var res = LibTorchSharp.THSTensor_stft(_input, n_fft, hop_length, win_length, _window, normalized, _onesided, _return_complex);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Inverse short time Fourier Transform. This is expected to be the inverse of stft().
            /// It has the same parameters (+ additional optional parameter of length) and it should return the least squares estimation of the original signal.
            /// The algorithm will check using the NOLA condition (nonzero overlap).
            /// </summary>
            /// <param name="n_fft">size of Fourier transform</param>
            /// <param name="hop_length">the distance between neighboring sliding window frames</param>
            /// <param name="win_length">the size of window frame and STFT filter.</param>
            /// <param name="window">The optional window function.</param>
            /// <param name="center">whether to pad input on both sides so that the tt-th frame is centered at time t * hop_length</param>
            /// <param name="normalized">controls whether to return the normalized STFT results</param>
            /// <param name="onesided">controls whether to return half of results to avoid redundancy for real inputs.</param>
            /// <param name="length">The amount to trim the signal by.</param>
            /// <param name="return_complex">whether to return a complex tensor, or a real tensor with an extra last dimension for the real and imaginary components.</param>
            /// <returns></returns>
            public Tensor istft(long n_fft, long hop_length = -1, long win_length = -1, Tensor? window = null, bool center = true, bool normalized = false, bool? onesided = null, long length = -1, bool return_complex = false)
            {
                var fft_size = shape[1];
                IntPtr _window = (window is null) ? IntPtr.Zero : window.Handle;

                long _onesided = -1; // encoding of null
                if (onesided.HasValue) {
                    _onesided = (onesided.Value ? 1 : 0);
                }

                var res = LibTorchSharp.THSTensor_istft(Handle, n_fft, hop_length, win_length, _window, center, normalized, _onesided, length, return_complex);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }
        }

        /// <summary>
        /// Type used to represent the variety of indexing capabilities that are
        /// available in Pyton, and therefore to PyTorch.
        /// </summary>
        public struct TensorIndex
        {
            internal enum Kind
            {
                None,
                Single,
                Null,
                Ellipsis,
                Bool,
                Tensor,
                Slice
            }
            internal long? startIndexOrBoolOrSingle;
            internal long? stopIndex;
            internal long? step;
            internal Kind kind;
            internal Tensor? tensor;

            static public TensorIndex Slice(long? start = null, long? stop = null, long? step = null)
            {
                return new TensorIndex() { startIndexOrBoolOrSingle = start, step = step, stopIndex = stop, kind = Kind.Slice };
            }

            static public TensorIndex Slice((int? start, int? end) range) => TensorIndex.Slice((long?)range.start, (long?)range.end);

#if !NETSTANDARD2_0_OR_GREATER
            static public TensorIndex Slice(System.Range range)
            {
                long? start = !range.Start.IsFromEnd ? range.Start.Value : -1 * range.Start.Value;
                long? end = !range.End.IsFromEnd ? range.End.Value : (range.End.Value == 0) ? null : -1 * range.End.Value;
                return TensorIndex.Slice(start, end);
            }
#endif // NETSTANDARD2_0_OR_GREATER
            static public TensorIndex Bool(bool value) => new TensorIndex() { startIndexOrBoolOrSingle = (value ? 1 : 0), kind = Kind.Bool };

            static public TensorIndex Single(long? index) => new TensorIndex() { startIndexOrBoolOrSingle = index, kind = Kind.Single };

            static public TensorIndex Tensor(Tensor tensor) => new TensorIndex() { tensor = tensor, kind = Kind.Tensor };

            static public TensorIndex Ellipsis = new TensorIndex() { kind = Kind.Ellipsis };

            static public TensorIndex None = new TensorIndex() { kind = Kind.None };

            static public TensorIndex Null = new TensorIndex() { kind = Kind.Null };

            static public TensorIndex Colon = Slice();

            public static implicit operator TensorIndex(long value)
            {
                return TensorIndex.Single(value);
            }

            public static implicit operator Tensor(TensorIndex value) { throw new InvalidOperationException("Implicit conversion from TensorIndex to Tensor -- this should never be invoked, the operator is only here to guide the compiler's overload resolution."); }

            public static implicit operator TensorIndex((int? start, int? end) range) => TensorIndex.Slice((long?)range.start, (long?)range.end);

#if !NETSTANDARD2_0_OR_GREATER
            public static implicit operator TensorIndex(System.Range range)
            {
                long? start = !range.Start.IsFromEnd ? range.Start.Value : -1 * range.Start.Value;
                long? end = !range.End.IsFromEnd ? range.End.Value : (range.End.Value == 0) ? null : -1 * range.End.Value;
                return TensorIndex.Slice(start, end);
            }
#endif // NETSTANDARD2_0_OR_GREATER
        }

        /// <summary>
        /// The element types of tensors.
        /// </summary>
        public enum ScalarType : sbyte
        {
            Byte = 0,
            Int8 = 1,
            Int16 = 2,
            Int32 = 3,
            Int64 = 4,
            Float16 = 5,
            Float32 = 6,
            Float64 = 7,
            //ComplexFloat16 = 8,
            ComplexFloat32 = 9,
            ComplexFloat64 = 10,
            Bool = 11,
            //QInt8 = 12,
            //QUInt8 = 13,
            //QUInt32 = 14,
            BFloat16 = 15
        }

        private static Dictionary<Type, ScalarType> typeMap = new()
        {
            { typeof(bool), ScalarType.Bool },
            { typeof(byte), ScalarType.Byte },
            { typeof(sbyte), ScalarType.Int8 },
            { typeof(short), ScalarType.Int16 },
            { typeof(int), ScalarType.Int32 },
            { typeof(long), ScalarType.Int64 },
            { typeof(float), ScalarType.Float32 },
            { typeof(double), ScalarType.Float64 },
            { typeof((float, float)), ScalarType.ComplexFloat32 },
            { typeof(System.Numerics.Complex), ScalarType.ComplexFloat64 },
        };

        internal static ScalarType ToScalarType(Type t)
        {
            if (typeMap.TryGetValue(t, out var scalarType)) return scalarType;
            throw new NotSupportedException($"The type {t.FullName} is not supported.");
        }

        public struct FInfo
        {
            public int bits;
            public double eps;
            public double max;
            public double min;
            public double tiny;
        }

        public static FInfo finfo(ScalarType dtype)
        {
            if (!is_floating_point(dtype) && !is_complex(dtype))
                throw new ArgumentException("'dtype' must be floating point or complex");

            if (dtype == ScalarType.ComplexFloat32)
                dtype = ScalarType.Float32;
            if (dtype == ScalarType.ComplexFloat64)
                dtype = ScalarType.Float64;

            FInfo result = new FInfo();

            switch (dtype) {
            case ScalarType.Float32:
                result.bits = 32;
                result.min = float.MinValue;
                result.max = float.MaxValue;
                result.eps = float.Epsilon;
                result.tiny = float.Epsilon;
                break;
            case ScalarType.Float64:
                result.bits = 64;
                result.min = double.MinValue;
                result.max = double.MaxValue;
                result.eps = double.Epsilon;
                result.tiny = double.Epsilon;
                break;
            }
            return result;
        }

        public static bool is_integral(ScalarType type)
        {
            switch (type) {
            case ScalarType.Byte:
            case ScalarType.Int8:
            case ScalarType.Int16:
            case ScalarType.Int32:
            case ScalarType.Int64:
            case ScalarType.Bool:
                return true;
            default:
                return false;
            }
        }

        public static bool is_floating_point(ScalarType type)
        {
            switch (type) {
            case ScalarType.BFloat16:
            case ScalarType.Float16:
            case ScalarType.Float32:
            case ScalarType.Float64:
                return true;
            default:
                return false;
            }
        }

        public static bool is_complex(ScalarType type)
        {
            switch (type) {
            case ScalarType.ComplexFloat32:
            case ScalarType.ComplexFloat64:
                return true;
            default:
                return false;
            }
        }

        public static bool is_integral(Tensor t) => is_integral(t.dtype);
        //public static bool is_floating_point(Tensor t) => is_floating_point(t.dtype);
        //public static bool is_complex(Tensor t) => is_complex(t.dtype);

        public static ScalarType @bool = ScalarType.Bool;

        public static ScalarType uint8 = ScalarType.Byte;
        public static ScalarType int8 = ScalarType.Int8;
        public static ScalarType int16 = ScalarType.Int16;
        public static ScalarType int32 = ScalarType.Int32;
        public static ScalarType int64 = ScalarType.Int64;

        public static ScalarType bfloat16 = ScalarType.BFloat16;
        public static ScalarType float16 = ScalarType.Float16;
        public static ScalarType float32 = ScalarType.Float32;
        public static ScalarType float64 = ScalarType.Float64;

        public static ScalarType complex64 = ScalarType.ComplexFloat32;
        public static ScalarType complex128 = ScalarType.ComplexFloat64;

        public static ScalarType @short = ScalarType.Int16;
        public static ScalarType @int = ScalarType.Int32;
        public static ScalarType @long = ScalarType.Int64;

        public static ScalarType half = ScalarType.Float16;
        public static ScalarType @float = ScalarType.Float32;
        public static ScalarType @double = ScalarType.Float64;

        public static ScalarType cfloat = ScalarType.ComplexFloat32;
        public static ScalarType cdouble = ScalarType.ComplexFloat64;

        /// <summary>
        /// Creates a new dispose scope for the current thread. Any tensor created within the dispose scope will
        /// be automatically disposed once the dispose scope is disposed.
        /// </summary>
        public static DisposeScope NewDisposeScope() => DisposeScopeManager.NewDisposeScope();

        /// <summary>
        /// Creates a new dispose scope for the current thread, wrapping an expression.
        /// </summary>
        public static Tensor WrappedTensorDisposeScope(Func<Tensor> expr)
        {
            using var scope = torch.NewDisposeScope();
            var result = expr();
            return result.MoveToOuterDisposeScope();
        }
    }
}