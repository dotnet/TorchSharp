// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

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
            ///  TBD
            /// </summary>
            /// <param name="obj"></param>

            public override bool Equals(object? obj)
            {
                return (obj is Tensor) && this.Equals((obj as Tensor)!);

            }
            /// <summary>
            ///  TBD
            /// </summary>
            public override int GetHashCode()
            {
                return base.GetHashCode();
            }

            /// <summary>
            /// A friendly name for the tensor. This is useful for debugging purposes.
            /// </summary>
            public string? name { get; set; }

            /// <summary>
            ///   Finalize the tensor. Releases the tensor and its associated data.
            /// </summary>
            ~Tensor() => Dispose(false);

            public void Dispose()
            {
                OwningDisposeScope?.MarkAsDisposed(this);
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            [DllImport("LibTorchSharp")]
            extern static void THSTensor_dispose(IntPtr handle);

            /// <summary>
            ///   Implements the .NET Dispose pattern.
            /// </summary>
            void Dispose(bool disposing)
            {
                if (handle != IntPtr.Zero) {
                    System.Threading.Interlocked.Decrement(ref _totalCount);
                    THSTensor_dispose(handle);
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
            /// Detatches the tensor completely from the DisposeScope system.
            /// </summary>
            /// <returns>The same tensor that the method was called on</returns>
            public torch.Tensor DetatchFromDisposeScope()
            {
                OwningDisposeScope?.Detach(this);
                return this;
            }

            [DllImport("LibTorchSharp")]
            extern static void THSTensor_free(IntPtr handle);

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

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_ndimension(IntPtr handle);

            /// <summary>
            ///  Returns the number of dimensions for this tensor
            /// </summary>
            public long Dimensions => THSTensor_ndimension(Handle);

            /// <summary>
            ///  Returns the number of dimensions for this tensor
            /// </summary>
            public long dim() => Dimensions;

            /// <summary>
            ///  Returns the number of dimensions for this tensor
            /// </summary>
            public long ndim => Dimensions;

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_element_size(IntPtr handle);

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_numel(IntPtr handle);

            /// <summary>
            ///  Get the number of elements in the tensor.
            /// </summary>
            public long NumberOfElements => THSTensor_numel(Handle);

            /// <summary>
            ///  Get the number of elements in the tensor.
            /// </summary>
            public long numel() => NumberOfElements;

            /// <summary>
            ///  Get the size of each element in the tensor.
            /// </summary>
            public long ElementSize => THSTensor_element_size(Handle);

            public long element_size() => THSTensor_element_size(Handle);

            public bool is_integral() => torch.is_integral(dtype);
            public bool is_floating_point() => torch.is_floating_point(dtype);
            public bool is_complex() => torch.is_complex(dtype);

            public bool is_cuda => device.type == DeviceType.CUDA;

            public bool is_meta => device.type == DeviceType.META;


            [DllImport("LibTorchSharp")]
            internal static extern long THSTensor_is_leaf(IntPtr handle);

            /// <summary>
            /// All Tensors that have requires_grad which is true will be leaf Tensors by convention.
            /// For Tensors that have requires_grad which is true, they will be leaf Tensors if they were created by the user.This means that they are not the result of an operation and so grad_fn is None.
            /// Only leaf Tensors will have their grad populated during a call to backward(). To get grad populated for non-leaf Tensors, you can use retain_grad().
            /// </summary>
            public bool is_leaf { get => THSTensor_is_leaf(Handle) != 0; }


            [DllImport("LibTorchSharp")]
            internal static extern IntPtr THSTensor_alias(IntPtr handle);

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
                var res = THSTensor_alias(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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

            [DllImport("LibTorchSharp")]
            internal static extern long THSTensor_storage_offset(IntPtr tensor);

            /// <summary>
            /// Returns the tensor’s offset in the underlying storage in terms of number of storage elements (not bytes).
            /// </summary>
            /// <returns></returns>
            public long storage_offset()
            {
                var res = THSTensor_storage_offset(Handle);
                torch.CheckForErrors();
                return res;
            }

            [DllImport("LibTorchSharp")]
            internal static extern IntPtr THSTensor_data(IntPtr handle);

            /// <summary>
            ///  Returns a pointer to the unmanaged data managed by this tensor.
            /// </summary>
            public Utils.TensorAccessor<T> data<T>() where T : unmanaged
            {
                if (device_type != DeviceType.CPU) {
                    throw new InvalidOperationException("Reading data from non-CPU memory is not supported. Move or copy the tensor to the cpu before reading.");
                }

                ValidateType(typeof(T));

                return new Utils.TensorAccessor<T>(this);
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
                        var res = THSTensor_data(handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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
                        var res = THSTensor_data(handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        // NOTE: there is no safety here.
                        var data = new Span<byte>((void*)res, value.Length);
                        value.CopyTo(data);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_real(IntPtr handle);

            public Tensor real {
                get {
                    var res = THSTensor_real(Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);

                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_imag(IntPtr handle);

            public Tensor imag {
                get {
                    var res = THSTensor_imag(Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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

            [DllImport("LibTorchSharp")]
            static extern float THSTensor_data_idx_float16(IntPtr handle, long i);

            /// <summary>
            /// Read the Float16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public float ReadCpuFloat16(long i)
            {
                if (i >= NumberOfElements) {
                    throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
                }
                return THSTensor_data_idx_float16(handle, i);
            }

            [DllImport("LibTorchSharp")]
            static extern float THSTensor_data_idx_bfloat16(IntPtr handle, long i);

            /// <summary>
            /// Read the BFloat16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            public float ReadCpuBFloat16(long i)
            {
                if (i >= NumberOfElements) {
                    throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
                }
                return THSTensor_data_idx_bfloat16(handle, i);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_item(IntPtr handle);

            /// <summary>
            /// Convert to a scalar.
            /// </summary>
            public Scalar ToScalar()
            {
                var res = THSTensor_item(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Scalar(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

            /// <summary>
            /// Fill the tensor with the provided scalar value.
            /// </summary>
            /// <param name="value">A scalar value</param>
            public Tensor fill_(Scalar value)
            {
                var res = THSTensor_fill_(Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern sbyte THSTensor_type(IntPtr handle);

            /// <summary>
            /// Gets the type of the tensor elements.
            /// </summary>
            public ScalarType dtype => (ScalarType)THSTensor_type(Handle);

            [DllImport("LibTorchSharp")]
            [return: MarshalAs(UnmanagedType.LPStr)]
            static extern string THSTensor_device_str(IntPtr handle);

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


            [DllImport("LibTorchSharp")]
            static extern int THSTensor_device_index(IntPtr handle);

            /// <summary>
            /// Gets a index of the device where the tensor is stored.
            /// </summary>
            public int device_index {
                get {
                    var res = THSTensor_device_index(Handle);
                    torch.CheckForErrors();
                    return res;
                }
            }


            [DllImport("LibTorchSharp")]
            static extern int THSTensor_device_type(IntPtr handle);

            /// <summary>
            /// Gets the type ('CPU', 'CUDA', etc.) of the device where the tensor is stored.
            /// </summary>
            public DeviceType device_type {
                get {
                    var res = THSTensor_device_type(Handle);
                    torch.CheckForErrors();
                    return (DeviceType)res;
                }
            }

            [DllImport("LibTorchSharp")]
            static extern bool THSTensor_is_sparse(IntPtr handle);

            /// <summary>
            /// Is the tensor a sparse tensor?
            /// </summary>
            public bool is_sparse {
                get {
                    var res = THSTensor_is_sparse(Handle);
                    torch.CheckForErrors();
                    return res;
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_load([MarshalAs(UnmanagedType.LPStr)] string location);

            /// <summary>
            /// Creates a tensor by loading it from a file.
            /// </summary>
            /// <param name="location">The file path where tensor values are stored.</param>
            public static Tensor load(string location)
            {
                var res = THSTensor_load(location);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_save(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string location);

            /// <summary>
            /// Save the contents of a tensor to a file.
            /// </summary>
            /// <param name="location">The file path where tensor values are to be stored.</param>
            public void save(string location)
            {
                THSTensor_save(Handle, location);
                torch.CheckForErrors();
            }

            [DllImport("LibTorchSharp")]
            static extern bool THSTensor_requires_grad(IntPtr handle);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set_requires_grad(IntPtr handle, bool requires_grad);

            /// <summary>
            /// Is the tensor tracking gradients?
            /// </summary>
            /// <remarks>Typically, gradients are tracked when the tensor is used as parameters of a module.</remarks>
            public bool requires_grad {
                get { return THSTensor_requires_grad(Handle); }
                set {
                    var res = THSTensor_set_requires_grad(Handle, value);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                }
            }

            public Tensor requires_grad_(bool requires_grad = true)
            {
                this.requires_grad = true;
                return this;
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_retain_grad(IntPtr handle);

            /// <summary>
            /// Enables this Tensor to have their grad populated during backward(). This is a no-op for leaf tensors.
            /// </summary>
            public void retain_grad()
            {
                THSTensor_retain_grad(Handle);
                torch.CheckForErrors();
            }

            /// <summary>
            /// Adds gradient tracking.
            /// </summary>
            public Tensor with_requires_grad(bool requires_grad = true)
            {
                this.requires_grad = requires_grad;
                return this;
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cpu(IntPtr handle);

            /// <summary>
            /// Moves the tensor data to the CPU device
            /// </summary>
            public Tensor cpu()
            {
                var res = THSTensor_cpu(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cuda(IntPtr handle);

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
                    ? THSTensor_cuda(Handle)
                    : THSTensor_to_device(Handle, (int)DeviceType.CUDA, device_index, false);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_device(IntPtr handle, int device_type, int device_index, bool copy);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type, bool copy);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_type_and_device(IntPtr handle, sbyte scalar_type, int device_type, int device_index, bool copy);

            /// <summary>
            /// Cast the tensor to the given element type.
            /// </summary>
            /// <param name="type">The target type</param>
            /// <param name="copy">When copy is set, a new Tensor is created even when the Tensor already matches the desired conversion.</param>
            public Tensor to_type(ScalarType type, bool copy = false)
            {
                var res = THSTensor_to_type(Handle, (sbyte)type, copy);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns this tensor cast to the type of the given tensor.
            /// </summary>
            public Tensor type_as(Tensor tensor) => to_type(tensor.dtype);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set_(IntPtr tensor, IntPtr source);

            /// <summary>
            /// Overwrite an existing tensor with the contents of another tensor.
            /// </summary>
            /// <param name="source">The source tensor</param>
            public Tensor set_(Tensor source)
            {
                var res = THSTensor_set_(Handle, source.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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
                var res = THSTensor_to_device(Handle, (int)deviceType, deviceIndex, copy);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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
                var res = THSTensor_to_type_and_device(Handle, (sbyte)type, (int)device.type, device.index, copy);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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

            public Tensor to(Tensor other) => to(other.device_type, other.device_index);

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_size(IntPtr handle, long dimension);

            /// <summary>
            ///  Retrieves the size of the specified dimension in the tensor.
            /// </summary>
            /// <param name="dim"></param>
            public long size(int dim)
            {
                var res = THSTensor_size(Handle, dim);
                torch.CheckForErrors();
                return res;
            }

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_sizes(IntPtr handle, AllocatePinnedArray allocator);

            /// <summary>
            ///  Retrieves the sizes of all dimensions of the tensor.
            /// </summary>
            public long[] size()
            {
                long[] ptrArray;

                using (var pa = new PinnedArray<long>()) {
                    THSTensor_sizes(Handle, pa.CreateArray);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray;
            }


            /// <summary>
            /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor,
            /// and each element is the size of the dimension
            /// </summary>
            /// <remarks>
            ///     An array of size 0 is used for constants, an array of size 1 is used
            ///     for single-dimension arrays, where the dimension is the value of the
            ///     first element.   And so on.
            /// </remarks>
            public long[] shape {
                get {
                    return size();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_indices(IntPtr handle);

            /// <summary>
            /// Return the indices tensor of a sparse COO tensor.
            /// </summary>
            public Tensor SparseIndices {
                get {
                    var res = THSTensor_indices(Handle);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_values(IntPtr handle);

            /// <summary>
            /// Return the values tensor of a sparse COO tensor.
            /// </summary>
            public Tensor SparseValues {
                get {
                    var res = THSTensor_values(Handle);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_vander(IntPtr handle, long N, bool increasing);

            /// <summary>
            ///
            /// </summary>
            public Tensor vander(long N = -1, bool increasing = false)
            {
                if (this.Dimensions != 1) throw new InvalidOperationException("Input argument for 'vander()' must be 1-D.");

                var res = THSTensor_vander(Handle, (N == -1) ? this.size(0) : N, increasing);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_stride(IntPtr handle, long dimension);

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_strides(IntPtr handle, AllocatePinnedArray allocator);

            /// <summary>
            ///  Retrieves the stride of all dimensions of the tensor.
            /// </summary>
            public long[] stride()
            {
                long[] ptrArray;

                using (var pa = new PinnedArray<long>()) {
                    THSTensor_strides(Handle, pa.CreateArray);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray;
            }

            /// <summary>
            ///  Retrieves the stride of the specified dimension in the tensor.
            /// </summary>
            public long stride(int dim)
            {
                var res = THSTensor_stride(Handle, dim);
                torch.CheckForErrors();
                return res;
            }



            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_as_strided(IntPtr input, IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, long storageOffset);

            /// <summary>
            ///  Create a view of an existing torch.Tensor input with specified size, stride and storage offset.
            /// </summary>
            public Tensor as_strided(long[] size, long[] strides, long storageOffset = 0L)
            {
                unsafe {
                    fixed (long* psizes = size, pstrides = strides) {
                        var result = THSTensor_as_strided(Handle, (IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, storageOffset);
                        if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(result);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_backward(IntPtr handle);

            /// <summary>
            /// Computes the gradient of current tensor w.r.t. graph leaves.
            /// </summary>
            public void backward()
            {
                THSTensor_backward(Handle);
                torch.CheckForErrors();
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_dense(IntPtr handle);

            /// <summary>
            /// Creates a strided copy of self.
            /// </summary>

            public Tensor to_dense()
            {
                var res = THSTensor_to_dense(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_clone(IntPtr handle);

            /// <summary>
            /// Returns a copy of the tensor input.
            /// </summary>

            public Tensor clone()
            {
                var res = THSTensor_clone(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, bool non_blocking);

            /// <summary>
            /// Copies the elements from source into the tensor and returns it.
            /// </summary>

            /// <remarks>The src tensor must be broadcastable with the target 'this' tensor. It may be of a different data type or reside on a different device.</remarks>
            public Tensor copy_(Tensor source, bool nonBlocking = false)
            {
                var res = THSTensor_copy_(Handle, source.Handle, nonBlocking);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static int THSTensor_is_contiguous(IntPtr handle);

            /// <summary>
            /// Returns true if the tensor is contiguous.
            /// </summary>
            public bool is_contiguous()
            {
                var res = THSTensor_is_contiguous(Handle);
                torch.CheckForErrors();
                return res != 0;
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_contiguous(IntPtr handle);

            /// <summary>
            /// Returns a contiguous in memory tensor containing the same data as the input tensor.
            /// If tensor is already in the specified memory format, this function returns the original tensor.
            /// </summary>
            public Tensor contiguous()
            {
                var res = THSTensor_contiguous(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static long THSTensor_is_pinned(IntPtr handle);

            /// <summary>
            /// Returns true if the tensor is contiguous.
            /// </summary>
            public bool is_pinned()
            {
                var res = THSTensor_is_pinned(Handle);
                torch.CheckForErrors();
                return res != 0;
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_pin_memory(IntPtr handle);

            /// <summary>
            /// Returns a contiguous in memory tensor containing the same data as the input tensor.
            /// If tensor is already in the specified memory format, this function returns the original tensor.
            /// </summary>
            public Tensor pin_memory()
            {
                var res = THSTensor_pin_memory(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_grad(IntPtr handle);

            /// <summary>
            /// This attribute is null by default and becomes a Tensor the first time a call to backward() computes gradients for the tensor.
            /// The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.
            /// </summary>
            public Tensor? grad()
            {
                var res = THSTensor_grad(Handle);
                torch.CheckForErrors();

                if (res == IntPtr.Zero)
                    return null;

                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_index(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_index_put_scalar_(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength, IntPtr value);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_index_put_(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength, IntPtr value);
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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get1(IntPtr handle, long i1);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_set1(IntPtr handle, long i1, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>

            [IndexerName("TensorItems")]
            public Tensor this[long i1] {
                get {
                    var res = THSTensor_get1(Handle, i1);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSTensor_set1(Handle, i1, value.Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>

            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2] {
                get {
                    var res = THSTensor_get2(Handle, i1, i2);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSTensor_set2(Handle, i1, i2, value.Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>

            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3] {
                get {
                    var res = THSTensor_get3(Handle, i1, i2, i3);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set3(Handle, i1, i2, i3, value.Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

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
                    var res = THSTensor_get4(Handle, i1, i2, i3, i4);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set4(Handle, i1, i2, i3, i4, value.Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get5(IntPtr handle, long i1, long i2, long i3, long i4, long i5);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_set5(IntPtr handle, long i1, long i2, long i3, long i4, long i5, IntPtr value);

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
                    var res = THSTensor_get5(Handle, i1, i2, i3, i4, i5);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set5(Handle, i1, i2, i3, i4, i5, value.Handle);
                    torch.CheckForErrors();
                }
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_set6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6, IntPtr value);

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
                    var res = THSTensor_get6(Handle, i1, i2, i3, i4, i5, i6);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set6(Handle, i1, i2, i3, i4, i5, i6, value.Handle);
                    torch.CheckForErrors();
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
                            var res = THSTensor_index(Handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length);
                            if (res == IntPtr.Zero)
                                torch.CheckForErrors();
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
                            var res = THSTensor_index_put_(Handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
                            if (res == IntPtr.Zero)
                                torch.CheckForErrors();
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
                            var res = THSTensor_index_put_scalar_(Handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
                            if (res == IntPtr.Zero)
                                torch.CheckForErrors();
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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_index_select(IntPtr tensor, long dimension, IntPtr index);

            /// <summary>
            /// Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
            /// </summary>
            /// <param name="dimension"></param>
            /// <param name="index"></param>

            public Tensor index_select(long dimension, Tensor index)
            {
                var res = THSTensor_index_select(Handle, dimension, index.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_select(IntPtr tensor, long dimension, long index);

            /// <summary>
            /// Slices the self tensor along the selected dimension at the given index.
            /// This function returns a view of the original tensor with the given dimension removed.
            /// </summary>
            /// <param name="dim">The dimension to slice</param>
            /// <param name="index">The index to select with</param>

            public Tensor select(long dim, long index)
            {
                var res = THSTensor_select(Handle, dim, index);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_take(IntPtr tensor, IntPtr index);

            /// <summary>
            /// Returns a new tensor with the elements of input at the given indices. The input tensor is treated as if it were viewed as a 1-D tensor.
            /// The result takes the same shape as the indices.
            /// </summary>
            /// <param name="index">The indices into tensor, an Int64 tensor.</param>

            public Tensor take(Tensor index)
            {
                var res = THSTensor_take(Handle, index.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_take_along_dim_dflt(IntPtr tensor, IntPtr indices);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_take_along_dim(IntPtr tensor, IntPtr indices, long dim);

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>

            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(Tensor indices)
            {
                var res = THSTensor_take_along_dim_dflt(Handle, indices.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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
            /// <param name="dimension">Dimension to select along.</param>

            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(Tensor indices, long dimension)
            {
                var res = THSTensor_take_along_dim(Handle, indices.Handle, dimension);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <param name="dim">Dimension to select along.</param>

            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(IEnumerable<long> indices, long dim) => take_along_dim(torch.tensor(indices.ToArray()), dim);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_reshape(IntPtr tensor, IntPtr shape, int length);

            /// <summary>
            /// Returns a tensor with the same data and number of elements as self but with the specified shape.
            /// </summary>
            /// <param name="shape">The new tensor shape.</param>
            public Tensor reshape(params long[] shape)
            {
                unsafe {
                    fixed (long* pshape = shape) {
                        var res = THSTensor_reshape(Handle, (IntPtr)pshape, shape.Length);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_flatten(IntPtr tensor, long start, long end);

            /// <summary>
            /// Flattens input by reshaping it into a one-dimensional tensor.
            /// </summary>
            /// <param name="start_dim">The first dim to flatten</param>
            /// <param name="end_dim">The last dim to flatten.</param>
            /// <remarks>Flattening a zero-dimensional tensor will return a one-dimensional view.</remarks>
            public Tensor flatten(long start_dim = 0, long end_dim = -1)
            {
                var res = THSTensor_flatten(Handle, start_dim, end_dim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unflatten(IntPtr tensor, long dimension, IntPtr shape, int length);

            /// <summary>
            /// Expands the dimension dim of the self tensor over multiple dimensions of sizes given by sizes.
            /// </summary>
            /// <param name="dim">Dimension to unflatten.</param>
            /// <param name="sizes">New shape of the unflattened dimension.</param>
            public Tensor unflatten(long dim, params long[] sizes)
            {
                unsafe {
                    fixed (long* pshape = sizes) {
                        var res = THSTensor_unflatten(Handle, dim, (IntPtr)pshape, sizes.Length);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unique(IntPtr tensor, bool sorted, bool return_inverse, bool return_counts, out IntPtr inverse_indices, out IntPtr counts);
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unique_dim(IntPtr tensor, long dim, bool sorted, bool return_inverse, bool return_counts, out IntPtr inverse_indices, out IntPtr counts);
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unique_consecutive(IntPtr tensor, bool return_inverse, bool return_counts, out IntPtr inverse_indices, out IntPtr counts);
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unique_dim_consecutive(IntPtr tensor, long dim, bool return_inverse, bool return_counts, out IntPtr inverse_indices, out IntPtr counts);

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
                    res = THSTensor_unique(Handle, sorted, return_inverse, return_counts, out inverse_indices, out counts);
                } else {
                    res = THSTensor_unique_dim(Handle, dim.Value, sorted, return_inverse, return_counts, out inverse_indices, out counts);
                }

                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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
                    ? THSTensor_unique_consecutive(Handle, return_inverse, return_counts, out inverse_indices, out counts)
                    : THSTensor_unique_dim_consecutive(Handle, dim.Value, return_inverse, return_counts, out inverse_indices, out counts);

                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), inverse_indices != IntPtr.Zero ? new Tensor(inverse_indices) : null, counts != IntPtr.Zero ? new Tensor(counts) : null);
            }

            /// <summary>
            /// Expands the dimension dim of the self tensor over multiple dimensions of sizes given by sizes.
            /// </summary>
            /// <param name="dim">Dimension to unflatten.</param>
            /// <param name="sizes">New shape of the unflattened dimension.</param>
            public Tensor unflatten(long dim, torch.Size sizes)
            {
                return unflatten(dim, sizes.Shape);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_squeeze(IntPtr tensor, long dimension);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_squeeze_no_dim(IntPtr tensor);

            /// <summary>
            /// Returns a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
            /// </summary>
            /// <param name="dim">If given, the input will be squeezed only in this dimension</param>

            public Tensor squeeze(long? dim = null)
            {
                var res = dim.HasValue ? THSTensor_squeeze(Handle, dim.Value) : THSTensor_squeeze_no_dim(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_t(IntPtr tensor);

            /// <summary>
            /// Expects input to be 1- or 2-D tensor and transposes dimensions 0 and 1.
            /// </summary>

            public Tensor t()
            {
                var res = THSTensor_t(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Is this Tensor with its dimensions reversed.
            /// </summary>

            public Tensor T {
                get {
                    return this.permute(Enumerable.Range(0, (int)ndim).Reverse().Select(i => (long)i).ToArray());
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_transpose(IntPtr tensor, long dim1, long dim2);

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// </summary>
            /// <param name="dim0"></param>
            /// <param name="dim1"></param>

            public Tensor transpose(long dim0, long dim1)
            {
                var res = THSTensor_transpose(Handle, dim0, dim1);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tril(IntPtr tensor, long diagonal);

            /// <summary>
            /// Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
            /// The lower triangular part of the matrix is defined as the elements on and below the diagonal.
            /// </summary>
            /// <param name="diagonal">The diagonal to consider</param>

            public Tensor tril(long diagonal = 0)
            {
                var res = THSTensor_tril(Handle, diagonal);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_triu(IntPtr tensor, long diagonal);

            /// <summary>
            /// Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
            /// The upper triangular part of the matrix is defined as the elements on and above the diagonal.
            /// </summary>
            /// <param name="diagonal">The diagonal to consider</param>

            public Tensor triu(long diagonal = 0)
            {
                var res = THSTensor_triu(Handle, diagonal);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_transpose_(IntPtr tensor, long dim1, long dim2);

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// Inplace version of transpose()
            /// </summary>
            /// <param name="dim0"></param>
            /// <param name="dim1"></param>

            public Tensor transpose_(long dim0, long dim1)
            {
                var res = THSTensor_transpose_(Handle, dim0, dim1);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_view(IntPtr tensor, IntPtr shape, int length);

            /// <summary>
            /// Returns a new tensor with the same data as the input tensor but of a different shape.
            /// </summary>
            /// <param name="shape">The shape of the view</param>

            public Tensor view(params long[] shape)
            {
                unsafe {
                    fixed (long* pshape = shape) {
                        var res = THSTensor_view(Handle, (IntPtr)pshape, shape.Length);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// View this tensor as the same size as other.
            /// </summary>
            /// <param name="other">The result tensor has the same size as other.</param>
            /// <remarks>
            /// self.view_as(other) is equivalent to self.view(other.size()).
            /// Please see view() for more information about view.
            /// </remarks>
            public Tensor view_as(Tensor other)
            {
                return view(other.shape);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_view_as_complex(IntPtr tensor);

            /// <summary>
            /// Returns a view of input as a complex tensor.
            /// </summary>
            public Tensor view_as_complex()
            {
                var result = THSTensor_view_as_complex(Handle);
                if (result == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_view_as_real(IntPtr tensor);

            /// <summary>
            /// Returns a view of input as a real tensor.
            /// </summary>
            public Tensor view_as_real()
            {
                var result = THSTensor_view_as_real(Handle);
                if (result == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_all(IntPtr tensor);

            /// <summary>
            ///
            /// </summary>

            public Tensor all()
            {
                var res = THSTensor_all(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_all_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

            /// <summary>
            ///
            /// </summary>
            /// <param name="dimension"></param>
            /// <param name="keepDim"></param>

            public Tensor all(long dimension, bool keepDim = false)
            {
                var res = THSTensor_all_along_dimension(Handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_amax(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_amax_out(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim, IntPtr _out);

            /// <summary>
            /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
            /// <param name="out">The output tensor -- optional.</param>

            public Tensor amax(long[] dims, bool keepDim = false, Tensor? @out = null)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = @out is null ?
                            THSTensor_amax(Handle, (IntPtr)pdims, dims.Length, keepDim) :
                            THSTensor_amax_out(Handle, (IntPtr)pdims, dims.Length, keepDim, @out.Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_amin(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim);
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_amin_out(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim, IntPtr _out);

            /// <summary>
            /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
            /// <param name="out">The output tensor -- optional.</param>

            public Tensor amin(long[] dims, bool keepDim = false, Tensor? @out = null)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = @out is null ?
                            THSTensor_amin(Handle, (IntPtr)pdims, dims.Length, keepDim) :
                            THSTensor_amin_out(Handle, (IntPtr)pdims, dims.Length, keepDim, @out.Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_aminmax(IntPtr tensor, long dim, bool keep_dim, out IntPtr max);


            /// <summary>
            /// Computes the minimum and maximum values of the input tensor.
            /// </summary>
            /// <param name="dim">The dimension along which to compute the values. If null, computes the values over the entire input tensor</param>
            /// <param name="keepDim"> If true, the reduced dimensions will be kept in the output tensor as dimensions with size 1 for broadcasting.</param>

            public (Tensor min, Tensor max) aminmax(long? dim = null, bool keepDim = false)
            {
                var res = THSTensor_aminmax(Handle, (dim is null) ? -1 : dim.Value, keepDim, out IntPtr maxHandle);
                if (res == IntPtr.Zero || maxHandle == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(maxHandle));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_any(IntPtr tensor);

            /// <summary>
            ///
            /// </summary>

            public Tensor any()
            {
                var res = THSTensor_any(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_any_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

            /// <summary>
            ///
            /// </summary>
            /// <param name="dimension"></param>
            /// <param name="keepDim"></param>

            public Tensor any(long dimension, bool keepDim = false)
            {
                var res = THSTensor_any_along_dimension(Handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argmax(IntPtr tensor);

            /// <summary>
            ///
            /// </summary>

            public Tensor argmax()
            {
                var res = THSTensor_argmax(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argmax_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

            /// <summary>
            ///
            /// </summary>
            /// <param name="dimension"></param>
            /// <param name="keepDim"></param>

            public Tensor argmax(long dimension, bool keepDim = false)
            {
                var res = THSTensor_argmax_along_dimension(Handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argmin(IntPtr tensor);

            /// <summary>
            ///
            /// </summary>

            public Tensor argmin()
            {
                var res = THSTensor_argmin(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argmin_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

            /// <summary>
            ///
            /// </summary>
            /// <param name="dimension"></param>
            /// <param name="keepDim"></param>

            public Tensor argmin(long dimension, bool keepDim = false)
            {
                var res = THSTensor_argmin_along_dimension(Handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argsort(IntPtr tensor, long dimension, bool descending);

            /// <summary>
            /// Returns the indices that sort a tensor along a given dimension in ascending order by value.
            /// </summary>
            /// <param name="dimension">The dimension to sort along</param>
            /// <param name="descending">Controls the sorting order (ascending or descending)</param>

            public Tensor argsort(long dimension = -1, bool descending = false)
            {
                var res = THSTensor_argsort(Handle, dimension, descending);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_deg2rad(IntPtr tensor);

            /// <summary>
            /// Convert each element from degrees to radians.
            /// </summary>

            public Tensor deg2rad()
            {
                var res = THSTensor_deg2rad(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rad2deg(IntPtr tensor);

            /// <summary>
            /// Convert each element from radians to degrees.
            /// </summary>

            public Tensor rad2deg()
            {
                var res = THSTensor_rad2deg(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_copysign(IntPtr tensor, IntPtr other);

            /// <summary>
            ///
            /// </summary>

            public Tensor copysign(Tensor other)
            {
                var res = THSTensor_copysign(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_count_nonzero(IntPtr tensor, IntPtr dim, int dim_len);

            public Tensor count_nonzero(long[]? dims = null)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSTensor_count_nonzero(Handle, (IntPtr)pdims, dims is null ? 0 : dims.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cov(IntPtr tensor, long correction, IntPtr fweights, IntPtr aweights);

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
                var res = THSTensor_cov(Handle, correction, fwHandle, awHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_corrcoef(IntPtr tensor);

            /// <summary>
            /// Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.
            /// </summary>
            /// <remarks>
            /// Due to floating point rounding, the resulting array may not be Hermitian and its diagonal elements may not be 1.
            /// The real and imaginary values are clipped to the interval [-1, 1] in an attempt to improve this situation.
            /// </remarks>
            public Tensor corrcoef()
            {
                var res = THSTensor_corrcoef(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tile(IntPtr tensor, IntPtr reps, int reps_len);

            /// <summary>
            /// Constructs a tensor by repeating the elements of input. The reps argument specifies the number of repetitions in each dimension.
            /// </summary>
            /// <param name="reps">The number of repetitions per dimension.</param>

            public Tensor tile(long[] reps)
            {
                unsafe {
                    fixed (long* pdims = reps) {
                        var res = THSTensor_tile(Handle, (IntPtr)pdims, reps.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_digamma(IntPtr tensor);

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input.
            /// </summary>

            public Tensor digamma()
            {
                var res = THSTensor_digamma(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_digamma_(IntPtr tensor);

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input, in place.
            /// </summary>

            public Tensor digamma_()
            {
                var res = THSTensor_digamma_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lgamma(IntPtr tensor);

            /// <summary>
            /// Computes the logarithm of the gamma function on input.
            /// </summary>

            public Tensor lgamma()
            {
                var res = THSTensor_lgamma(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lgamma_(IntPtr tensor);

            /// <summary>
            /// Computes the logarithm of the gamma function on input, in place.
            /// </summary>

            public Tensor lgamma_()
            {
                var res = THSTensor_lgamma_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mvlgamma(IntPtr tensor, long p);

            /// <summary>
            /// Computes the multivariate log-gamma function) with dimension pp element-wise
            /// </summary>
            /// <param name="p">The number of dimensions</param>

            public Tensor mvlgamma(long p)
            {
                var res = THSTensor_mvlgamma(Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mvlgamma_(IntPtr tensor, long p);

            /// <summary>
            /// Computes the multivariate log-gamma function) with dimension pp element-wise, in place.
            /// </summary>
            /// <param name="p">The number of dimensions</param>

            public Tensor mvlgamma_(long p)
            {
                var res = THSTensor_mvlgamma_(Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_polygamma(IntPtr tensor, long p);

            public Tensor polygamma(long p)
            {
                var res = THSTensor_polygamma(Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_polygamma_(IntPtr tensor, long p);

            public Tensor polygamma_(long p)
            {
                var res = THSTensor_polygamma_(Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_positive(IntPtr tensor);

            /// <summary>
            /// Returns input. Throws a runtime error if input is a bool tensor.
            /// </summary>

            public Tensor positive()
            {
                if (this.dtype == ScalarType.Bool) throw new ArgumentException("Boolean tensor");
                var res = THSTensor_positive(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_softplus(IntPtr tensor);

            public Tensor softplus()
            {
                var res = THSTensor_softplus(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ravel(IntPtr tensor);

            public Tensor ravel()
            {
                var res = THSTensor_ravel(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu(IntPtr tensor);

            public Tensor relu()
            {
                var res = THSTensor_relu(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu_(IntPtr tensor);

            public Tensor relu_()
            {
                var res = THSTensor_relu_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu6(IntPtr tensor);

            public Tensor relu6()
            {
                var res = THSTensor_relu6(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu6_(IntPtr tensor);

            public Tensor relu6_()
            {
                var res = THSTensor_relu6_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_celu(IntPtr tensor);

            public Tensor celu()
            {
                var res = THSTensor_celu(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_celu_(IntPtr tensor);

            public Tensor celu_()
            {
                var res = THSTensor_celu_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_elu(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

            public Tensor elu(Scalar alpha, Scalar scale, Scalar input_scale)
            {
                var res = THSTensor_elu(Handle, alpha.Handle, scale.Handle, input_scale.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_elu_(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

            public Tensor elu_(Scalar alpha, Scalar scale, Scalar input_scale)
            {
                var res = THSTensor_elu_(Handle, alpha.Handle, scale.Handle, input_scale.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gelu(IntPtr tensor);

            public Tensor gelu()
            {
                var res = THSTensor_gelu(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardsigmoid(IntPtr tensor);

            public Tensor hardsigmoid()
            {
                var res = THSTensor_hardsigmoid(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardsigmoid_(IntPtr tensor);

            public Tensor hardsigmoid_()
            {
                var res = THSTensor_hardsigmoid_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardswish(IntPtr tensor);

            public Tensor hardswish()
            {
                var res = THSTensor_hardswish(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardswish_(IntPtr tensor);

            public Tensor hardswish_()
            {
                var res = THSTensor_hardswish_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardtanh(IntPtr tensor, IntPtr min, IntPtr max);

            public Tensor hardtanh(Scalar min, Scalar max)
            {
                var res = THSTensor_hardtanh(Handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardtanh_(IntPtr tensor, IntPtr min, IntPtr max);

            public Tensor hardtanh_(Scalar min, Scalar max)
            {
                var res = THSTensor_hardtanh_(Handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_heaviside(IntPtr tensor, IntPtr other);

            public Tensor heaviside(Tensor other)
            {
                var res = THSTensor_heaviside(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_igamma(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Computes the regularized lower incomplete gamma function
            /// </summary>
            /// <param name="other">The second non-negative input tensor</param>

            public Tensor igamma(Tensor other)
            {
                var res = THSTensor_igamma(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_igammac(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Computes the regularized upper incomplete gamma function.
            /// </summary>
            /// <param name="other">The second non-negative input tensor</param>

            public Tensor igammac(Tensor other)
            {
                var res = THSTensor_igammac(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_i0(IntPtr tensor);

            /// <summary>
            /// Computes the zeroth order modified Bessel function of the first kind for each element of input.
            /// </summary>

            public Tensor i0()
            {
                var res = THSTensor_i0(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isclose(IntPtr tensor, IntPtr other, double rtol, double atol, bool nanEqual);

            /// <summary>
            /// Returns a new tensor with boolean elements representing if each element of input is “close” to the corresponding element of other.
            /// </summary>
            /// <param name="other">Second tensor to compare</param>
            /// <param name="rtol">Relative tolerance</param>
            /// <param name="atol">Absolute tolerance</param>
            /// <param name="nanEqual">If true, then two NaN s will be considered equal</param>
            public Tensor isclose(Tensor other, double rtol = 1e-05, double atol = 1e-08, bool nanEqual = false)
            {
                var res = THSTensor_isclose(Handle, other.Handle, rtol, atol, nanEqual);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isin(IntPtr elements, IntPtr test_elements, bool assume_unique, bool invert);

            /// <summary>
            /// Tests if each element of elements is in test_elements.
            /// Returns a boolean tensor of the same shape as elements that is true for elements in test_elements and false otherwise.
            /// </summary>
            /// <param name="test_elements">Values against which to test for each input element</param>
            /// <param name="assumeUnique">If true, assumes both elements and test_elements contain unique elements, which can speed up the calculation.</param>
            /// <param name="invert">If true, inverts the boolean return tensor, resulting in true values for elements not in test_elements.</param>
            public Tensor isin(Tensor test_elements, bool assumeUnique = false, bool invert = false)
            {
                var res = THSTensor_isin(Handle, test_elements.Handle, assumeUnique, invert);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isinf(IntPtr tensor);

            public Tensor isinf()
            {
                var res = THSTensor_isinf(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isfinite(IntPtr tensor);

            public Tensor isfinite()
            {
                var res = THSTensor_isfinite(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isposinf(IntPtr tensor);

            public Tensor isposinf()
            {
                var res = THSTensor_isposinf(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isneginf(IntPtr tensor);

            public Tensor isneginf()
            {
                var res = THSTensor_isneginf(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isreal(IntPtr tensor);

            public Tensor isreal()
            {
                var res = THSTensor_isreal(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_leaky_relu(IntPtr tensor, IntPtr negative_slope);

            public Tensor leaky_relu(Scalar negative_slope)
            {
                var res = THSTensor_leaky_relu(Handle, negative_slope.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_leaky_relu_(IntPtr tensor, IntPtr negative_slope);

            public Tensor leaky_relu_(Scalar negative_slope)
            {
                var res = THSTensor_leaky_relu_(Handle, negative_slope.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_selu(IntPtr tensor);

            public Tensor selu()
            {
                var res = THSTensor_selu(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_selu_(IntPtr tensor);

            public Tensor selu_()
            {
                var res = THSTensor_selu_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_silu(IntPtr tensor);

            public Tensor silu()
            {
                var res = THSTensor_silu(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_silu_(IntPtr tensor);

            public Tensor silu_()
            {
                var res = THSTensor_silu_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log_sigmoid(IntPtr tensor);

            public Tensor log_sigmoid()
            {
                var res = THSTensor_log_sigmoid(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lerp(IntPtr tensor, IntPtr end, IntPtr weight);

            public Tensor lerp(Tensor end, Tensor weight)
            {
                var res = THSTensor_lerp(Handle, end.Handle, weight.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lerp_(IntPtr tensor, IntPtr end, IntPtr weight);

            public Tensor lerp_(Tensor end, Tensor weight)
            {
                var res = THSTensor_lerp_(Handle, end.Handle, weight.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
                float alpha);

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
                var res = THSTensor_baddbmm(Handle, batch1.Handle, batch2.Handle, beta, alpha);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

            /// <summary>
            /// Performs a batch matrix-matrix product of matrices stored in input and mat2.
            /// </summary>
            /// <param name="batch2">the second batch of matrices to be multiplied</param>
            /// <returns></returns>
            public Tensor bmm(Tensor batch2)
            {
                var res = THSTensor_bmm(Handle, batch2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bucketize(IntPtr input, IntPtr boundaries, bool out_int32, bool right);

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
                var res = THSTensor_bucketize(Handle, boundaries.Handle, outInt32, right);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bincount(IntPtr tensor, IntPtr weights, long minlength);

            /// <summary>
            /// Count the frequency of each value in an array of non-negative ints.
            /// </summary>
            public Tensor bincount(Tensor? weights, long minlength = 0)
            {
                var weightsHandle = (weights is null ? IntPtr.Zero : weights.Handle);
                var res = THSTensor_bincount(Handle, weightsHandle, minlength);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            public Tensor @bool() => this.to_type(ScalarType.Bool);

            public Tensor @byte() => this.to_type(ScalarType.Byte);

            public Tensor @char() => this.to_type(ScalarType.Int8);

            public Tensor @int() => this.to_type(ScalarType.Int32);

            public Tensor @long() => this.to_type(ScalarType.Int64);

            public Tensor @float() => this.to_type(ScalarType.Float32);

            public Tensor @double() => this.to_type(ScalarType.Float64);


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_channel_shuffle(IntPtr input, long groups);

            /// <summary>
            /// Divide the channels in a tensor into g groups and rearrange them.
            ///
            /// See: https://pytorch.org/docs/1.10/generated/torch.nn.ChannelShuffle.html#channelshuffle
            /// </summary>
            /// <param name="groups">The number of groups to divide channels in.</param>
            public Tensor channel_shuffle(long groups)
            {
                var res = THSTensor_channel_shuffle(Handle, groups);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_tensor(IntPtr input, IntPtr min, IntPtr max);

            public Tensor clamp(Scalar? min = null, Scalar? max = null)
            {
                var res = THSTensor_clamp(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clamp(Tensor? min = null, Tensor? max = null)
            {
                var res = THSTensor_clamp_tensor(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clip(Scalar min, Scalar max) => clamp(min, max);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_tensor_(IntPtr input, IntPtr min, IntPtr max);

            public Tensor clamp_(Scalar? min = null, Scalar? max = null)
            {
                var res = THSTensor_clamp_(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clamp_(Tensor? min = null, Tensor? max = null)
            {
                var res = THSTensor_clamp_tensor_(Handle, min?.Handle ?? IntPtr.Zero, max?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

            public Tensor clamp_max(Scalar max)
            {
                var res = THSTensor_clamp_max(Handle, max.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

            public Tensor clamp_max_(Scalar max)
            {
                var res = THSTensor_clamp_max_(Handle, max.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

            public Tensor clamp_min(Scalar min)
            {
                var res = THSTensor_clamp_min(Handle, min.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

            public Tensor clamp_min_(Scalar min)
            {
                var res = THSTensor_clamp_min_(Handle, min.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diff(IntPtr tensor, long n, long dim, IntPtr prepend, IntPtr append);

            public Tensor diff(long n = 1, long dim = -1, Tensor? prepend = null, Tensor? append = null)
            {
                if (n != 1) throw new NotImplementedException("Tensor.diff with n != 1");
                var res = THSTensor_diff(Handle, n, dim, (prepend is Tensor) ? (IntPtr)prepend.Handle : IntPtr.Zero, (append is Tensor) ? (IntPtr)append.Handle : IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diag(IntPtr tensor, long dimension);

            public Tensor diag(long dimension = 0)
            {
                var res = THSTensor_diag(Handle, dimension);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diagflat(IntPtr tensor, long offset);

            public Tensor diagflat(long offset = 0)
            {
                var res = THSTensor_diagflat(Handle, offset);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diagonal(IntPtr tensor, long offset, long dim1, long dim2);

            /// <summary>
            /// Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.
            /// The argument offset controls which diagonal to consider:
            ///
            ///     If offset = 0, it is the main diagonal.
            ///     If offset &gt; 0, it is above the main diagonal.
            ///     If offset &lt; 0, it is below the main diagonal.
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
                var res = THSTensor_diagonal(Handle, offset, dim1, dim2);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erf(IntPtr tensor);

            public Tensor erf()
            {
                var res = THSTensor_erf(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erf_(IntPtr tensor);

            public Tensor erf_()
            {
                var res = THSTensor_erf_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfc(IntPtr tensor);

            public Tensor erfc()
            {
                var res = THSTensor_erfc(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfc_(IntPtr tensor);

            public Tensor erfc_()
            {
                var res = THSTensor_erfc_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfinv(IntPtr tensor);

            public Tensor erfinv()
            {
                var res = THSTensor_erfinv(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfinv_(IntPtr tensor);

            public Tensor erfinv_()
            {
                var res = THSTensor_erfinv_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq(IntPtr tensor, IntPtr trg);

            public Tensor eq(Tensor target)
            {
                var res = THSTensor_eq(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor equal(Tensor target) => eq(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq_(IntPtr tensor, IntPtr trg);

            public Tensor eq_(Tensor target)
            {
                var res = THSTensor_eq_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq_scalar(IntPtr tensor, IntPtr trg);

            public Tensor eq(Scalar target)
            {
                var res = THSTensor_eq_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor eq_(Scalar target)
            {
                var res = THSTensor_eq_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern bool THSTensor_equal(IntPtr tensor, IntPtr trg);

            public bool Equals(Tensor target)
            {
                var res = THSTensor_equal(Handle, target.Handle);
                torch.CheckForErrors();
                return res;
            }

            [DllImport("LibTorchSharp")]
            static extern bool THSTensor_allclose(IntPtr tensor, IntPtr trg, double rtol, double atol, bool equal_nan);

            /// <summary>
            /// This function checks if all input and other lie within a certain distance from each other
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rtol">Relative tolerance</param>
            /// <param name="atol">Absolute tolerance</param>
            /// <param name="equal_nan">If true, then two NaN s will be considered equal</param>

            public bool allclose(Tensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
            {
                var res = THSTensor_allclose(Handle, target.Handle, rtol, atol, equal_nan);
                torch.CheckForErrors();
                return res;
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge(IntPtr tensor, IntPtr trg);

            public Tensor ge(Tensor target)
            {
                var res = THSTensor_ge(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor greater_equal(Tensor target) => ge(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge_(IntPtr tensor, IntPtr trg);

            public Tensor ge_(Tensor target)
            {
                var res = THSTensor_ge_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge_scalar(IntPtr tensor, IntPtr trg);

            public Tensor ge(Scalar target)
            {
                var res = THSTensor_ge_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor ge_(Scalar target)
            {
                var res = THSTensor_ge_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt(IntPtr tensor, IntPtr trg);

            public Tensor gt(Tensor target)
            {
                var res = THSTensor_gt(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor greater(Tensor target) => gt(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt_(IntPtr tensor, IntPtr trg);

            public Tensor gt_(Tensor target)
            {
                var res = THSTensor_gt_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt_scalar(IntPtr tensor, IntPtr trg);

            public Tensor gt(Scalar target)
            {
                var res = THSTensor_gt_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor gt_(Scalar target)
            {
                var res = THSTensor_gt_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_kron(IntPtr tensor, IntPtr other);

            public Tensor kron(Tensor other)
            {
                var res = THSTensor_kron(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lcm(IntPtr tensor, IntPtr other);

            public Tensor lcm(Tensor other)
            {
                var res = THSTensor_lcm(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lcm_(IntPtr tensor, IntPtr other);

            public Tensor lcm_(Tensor other)
            {
                var res = THSTensor_lcm_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ldexp(IntPtr right, IntPtr left);

            /// <summary>
            /// Multiplies input by pow(2,other).
            /// </summary>
            /// <param name="other">A tensor of exponents, typically integers</param>

            /// <remarks>Typically this function is used to construct floating point numbers by multiplying mantissas in input with integral powers of two created from the exponents in other.</remarks>
            public Tensor ldexp(Tensor other)
            {
                var res = THSTensor_ldexp(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ldexp_(IntPtr right, IntPtr left);

            public Tensor ldexp_(Tensor other)
            {
                var res = THSTensor_ldexp_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le(IntPtr tensor, IntPtr trg);

            public Tensor le(Tensor target)
            {
                var res = THSTensor_le(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less_equal(Tensor target) => le(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le_(IntPtr tensor, IntPtr trg);

            public Tensor le_(Tensor target)
            {
                var res = THSTensor_le_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less_equal_(Tensor target) => le_(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le_scalar(IntPtr tensor, IntPtr trg);

            public Tensor le(Scalar target)
            {
                var res = THSTensor_le_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor le_(Scalar target)
            {
                var res = THSTensor_le_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt(IntPtr tensor, IntPtr trg);

            public Tensor lt(Tensor target)
            {
                var res = THSTensor_lt(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less(Tensor target) => lt(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt_(IntPtr tensor, IntPtr trg);

            public Tensor lt_(Tensor target)
            {
                var res = THSTensor_lt_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt_scalar(IntPtr tensor, IntPtr trg);

            public Tensor lt(Scalar target)
            {
                var res = THSTensor_lt_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor lt_(Scalar target)
            {
                var res = THSTensor_lt_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_fill(IntPtr tensor, IntPtr mask, IntPtr value);

            public Tensor masked_fill(Tensor mask, Scalar value)
            {
                var res = THSTensor_masked_fill(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_scatter(IntPtr tensor, IntPtr mask, IntPtr value);

            public Tensor masked_scatter(Tensor mask, Tensor value)
            {
                var res = THSTensor_masked_scatter(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_scatter_(IntPtr tensor, IntPtr mask, IntPtr value);

            public Tensor masked_scatter_(Tensor mask, Tensor value)
            {
                var res = THSTensor_masked_scatter_(Handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_select(IntPtr tensor, IntPtr mask);

            public Tensor masked_select(Tensor mask)
            {
                if (mask.dtype != ScalarType.Bool) throw new ArgumentException("The mask tensor must be Boolean.");
                var res = THSTensor_masked_select(Handle, mask.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_topk(IntPtr tensor, AllocatePinnedArray allocator, int k,
                long dimension, bool largest, bool sorted);

            public (Tensor values, Tensor indexes) topk(int k, int dimension = -1, bool largest = true, bool sorted = true)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_topk(Handle, pa.CreateArray, k, dimension, largest, sorted);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }


            [DllImport("LibTorchSharp")]
            static extern void THSTensor_unbind(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

            /// <summary>
            /// Removes a tensor dimension.
            /// </summary>
            /// <param name="dimension">The dimension to remove.</param>
            /// <returns>An array of all slices along a given dimension, already without it.</returns>
            public Tensor[] unbind(int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_unbind(Handle, pa.CreateArray, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unfold(IntPtr tensor, long dimension, long size, long step);

            /// <summary>
            /// Returns a view of the original tensor which contains all slices of size 'size' from the tensor in the given dimension.
            /// </summary>
            /// <param name="dimension">Dimension in which unfolding happens</param>
            /// <param name="size">The size of each slice that is unfolded</param>
            /// <param name="step">The step between each slice</param>
            public Tensor unfold(long dimension, long size, long step)
            {
                var res = THSTensor_unfold(Handle, dimension, size, step);
                if (res == IntPtr.Zero) torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="size">The size of a single chunk</param>
            /// <param name="dimension">The dimension along which to split the tensor.</param>

            public Tensor[] split(long size, int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_split_with_size(Handle, pa.CreateArray, size, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="sizes">A list of sizes for each chunk</param>
            /// <param name="dimension">The dimension along which to split the tensor.</param>

            public Tensor[] split(long[] sizes, int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            THSTensor_split_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                            torch.CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_tensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

            public Tensor[] tensor_split(long size, int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_tensor_split_with_size(Handle, pa.CreateArray, size, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_tensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

            public Tensor[] tensor_split(long[] sizes, int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            THSTensor_tensor_split_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                            torch.CheckForErrors();
                        }
                    }
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_tensor_split_with_tensor_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr indices, long dimension);

            public Tensor[] tensor_split(Tensor indices, int dimension = 0)
            {
                if (indices.dtype != ScalarType.Int64) throw new ArgumentException("Tensor indices should be Int64 in 'tensor_split");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_tensor_split_with_tensor_sizes(Handle, pa.CreateArray, indices.Handle, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_vsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to sizes.
            /// </summary>
            /// <param name="size">The size of each chunk</param>

            public Tensor[] vsplit(long size)
            {
                if (this.shape[0] % size != 0) throw new ArgumentException("The first dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_vsplit_with_size(Handle, pa.CreateArray, size);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_vsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

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
                            THSTensor_vsplit_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
                            torch.CheckForErrors();
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


            [DllImport("LibTorchSharp")]
            static extern void THSTensor_hsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to sizes.
            /// </summary>
            /// <param name="size">The size of each chunk</param>

            public Tensor[] hsplit(long size)
            {
                if (this.shape[1] % size != 0) throw new ArgumentException("The second dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_hsplit_with_size(Handle, pa.CreateArray, size);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_hsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

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
                            THSTensor_hsplit_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
                            torch.CheckForErrors();
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

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_dsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="size">The size of each chunk</param>

            public Tensor[] dsplit(long size)
            {
                if (this.shape[2] % size != 0) throw new ArgumentException("The third dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_dsplit_with_size(Handle, pa.CreateArray, size);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_dsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

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
                            THSTensor_dsplit_with_sizes(Handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
                            torch.CheckForErrors();
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


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_chunk(IntPtr tensor, AllocatePinnedArray allocator, long chunks, long dim);

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
                    THSTensor_chunk(Handle, pa.CreateArray, chunks, dim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_kthvalue(IntPtr tensor, long k, long dim, bool keepdim, out IntPtr _out);

            /// <summary>
            /// Returns a namedtuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim. And indices is the index location of each element found.
            /// If dim is not given, the last dimension of the input is chosen.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="k">k for the k-th smallest element</param>
            /// <param name="dim">The dimension to find the kth value along</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>

            public static (Tensor, Tensor) kthvalue(Tensor input, long k, long? dim, bool keepdim = false)
            {
                var values = THSTensor_kthvalue(input.Handle, k, dim.HasValue ? dim.Value : -1, keepdim, out var indices);
                if (values == IntPtr.Zero || indices == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(values), new Tensor(indices));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_max(IntPtr tensor);

            public Tensor max()
            {
                var res = THSTensor_max(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_max_elementwise(IntPtr tensor, IntPtr other);

            public Tensor maximum(Tensor other)
            {
                var res = THSTensor_max_elementwise(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_max_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dimension,
                bool keep_dim);

            public (Tensor values, Tensor indexes) max(long dimension, bool keepDim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_max_along_dimension(Handle, pa.CreateArray, dimension, keepDim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mean(IntPtr tensor);


            public Tensor mean()
            {
                var res = THSTensor_mean(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_quantile(IntPtr tensor, IntPtr q, long dim, bool keepdim);

            /// <summary>
            /// Returns the q-th quantiles of all elements in the input tensor, doing a linear interpolation when the q-th quantile lies between two data points.
            /// </summary>
            /// <param name="q">1D tensor of quantile values in the range [0, 1]</param>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>

            Tensor quantile(Tensor q, long dim = -1, bool keepdim = false)
            {
                var res = THSTensor_quantile(Handle, q.Handle, dim, keepdim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_nanquantile(IntPtr tensor, IntPtr q, long dim, bool keepdim);

            /// <summary>
            /// This is a variant of torch.quantile() that “ignores” NaN values, computing the quantiles q as if NaN values in input did not exist.
            /// If all values in a reduced row are NaN then the quantiles for that reduction will be NaN.
            /// </summary>
            /// <seealso cref="Tensor.quantile(Tensor, long, bool)"/>
            /// <param name="q">1D tensor of quantile values in the range [0, 1]</param>
            /// <param name="dim">The dimension to reduce.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>

            Tensor nanquantile(Tensor q, long dim = -1, bool keepdim = false)
            {
                var res = THSTensor_nanquantile(Handle, q.Handle, dim, keepdim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mode(IntPtr tensor, AllocatePinnedArray allocator, long dim, bool keepdim);

            /// <summary>
            /// Returns a namedtuple (values, indices) where values is the mode value of each row of the input tensor in the given dimension dim,
            /// i.e. a value which appears most often in that row, and indices is the index location of each mode value found.
            /// </summary>
            /// <param name="dim">The dimension to reduce, the last dimension by default.</param>
            /// <param name="keepdim">Whether the output tensor has dim retained or not</param>


            public (Tensor values, Tensor indices) mode(long dim = -1L, bool keepdim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_mode(Handle, pa.CreateArray, dim, keepdim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (values: new Tensor(ptrArray[0]), indices: new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

            public Tensor mean(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_mean_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_median(IntPtr tensor);

            public Tensor median()
            {
                var res = THSTensor_median(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_min(IntPtr tensor);

            public Tensor min()
            {
                var res = THSTensor_min(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_min_elementwise(IntPtr tensor, IntPtr other);

            public Tensor minimum(Tensor other)
            {
                var res = THSTensor_min_elementwise(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_min_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dimension,
                bool keep_dim);

            public (Tensor values, Tensor indexes) min(long dimension, bool keepDim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_min_along_dimension(Handle, pa.CreateArray, dimension, keepDim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_msort(IntPtr tensor);

            public Tensor msort()
            {
                var res = THSTensor_msort(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sort(IntPtr tensor, long dim, bool descending, bool stable, out IntPtr indices);

            public (Tensor Values, Tensor Indices) sort(long dim = -1, bool descending = false, bool stable = false)
            {
                var res = THSTensor_sort(Handle, dim, descending, stable, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne(IntPtr tensor, IntPtr trg);

            public Tensor ne(Tensor target)
            {
                var res = THSTensor_ne(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor not_equal(Tensor target) => ne(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne_(IntPtr tensor, IntPtr trg);

            public Tensor ne_(Tensor target)
            {
                var res = THSTensor_ne_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor not_equal_(Tensor target) => ne_(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne_scalar(IntPtr tensor, IntPtr trg);

            public Tensor ne(Scalar target)
            {
                var res = THSTensor_ne_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor ne_(Scalar target)
            {
                var res = THSTensor_ne_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_dist(IntPtr tensor, IntPtr other, float p);

            public Tensor dist(Tensor other, float p = 2.0f)
            {
                var res = THSTensor_dist(Handle, other.Handle, p);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_norm(IntPtr tensor, float p);

            public Tensor norm(float p = 2.0f)
            {
                var res = THSTensor_norm(Handle, p);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_norm_along_dimension(IntPtr tensor, int dimension, bool keepdim, float p);

            public Tensor norm(int dimension, bool keepdim = false, float p = 2.0f)
            {
                var res = THSTensor_norm_along_dimension(Handle, dimension, keepdim, p);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_outer(IntPtr input, IntPtr vec2);

            public Tensor outer(Tensor vec2)
            {
                var res = THSTensor_outer(Handle, vec2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_inner(IntPtr input, IntPtr vec2);

            public Tensor inner(Tensor vec2)
            {
                var res = THSTensor_inner(Handle, vec2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_inverse(IntPtr tensor);

            public Tensor inverse()
            {
                var res = THSTensor_inverse(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_prelu(IntPtr tensor, IntPtr trg);

            public Tensor prelu(Tensor target)
            {
                var res = THSTensor_prelu(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmax(IntPtr tensor, IntPtr trg);

            public Tensor fmax(Tensor target)
            {
                var res = THSTensor_fmax(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmin(IntPtr tensor, IntPtr trg);

            public Tensor fmin(Tensor target)
            {
                var res = THSTensor_fmin(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_renorm(IntPtr tensor, float p, long dim, float maxnorm);

            public Tensor renorm(Scalar scalar, float p, long dim, float maxnorm)
            {
                var res = THSTensor_renorm(Handle, p, dim, maxnorm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sigmoid(IntPtr tensor);

            public Tensor sigmoid()
            {
                var res = THSTensor_sigmoid(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sigmoid_(IntPtr tensor);

            public Tensor sigmoid_()
            {
                var res = THSTensor_sigmoid_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_std(IntPtr tensor, bool unbiased);

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            [System.Diagnostics.Contracts.Pure]
            public Tensor std(bool unbiased = true)
            {
                var res = THSTensor_std(Handle, unbiased);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_std_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool unbiased, bool keepdim);

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public Tensor std(long[] dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_std_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepDimension);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public Tensor std(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_std_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepDimension);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimension">The dimension to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dimension" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public Tensor std(long dimension, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
                => std(stackalloc[] { dimension }, unbiased, keepDimension, type);

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dimensions" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public Tensor std((long,long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
                => std(stackalloc[] { dimensions.Item1, dimensions.Item2 }, unbiased, keepDimension, type);

            /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dimensions" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public Tensor std((long, long, long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
                => std(stackalloc[] { dimensions.Item1, dimensions.Item2, dimensions.Item3 }, unbiased, keepDimension, type);


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_std_mean(IntPtr tensor, bool unbiased, out IntPtr mean);

            /// <summary>
            /// Calculates the standard deviation and mean of all elements in the tensor.
            /// </summary>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            [System.Diagnostics.Contracts.Pure]
            public (Tensor std, Tensor mean) std_mean(bool unbiased = true)
            {
                var res = THSTensor_std_mean(Handle, unbiased, out var mean);
                if (res == IntPtr.Zero || mean == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(mean));
            }
            
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_std_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool unbiased, bool keepdim, out IntPtr mean);

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public (Tensor std, Tensor mean) std_mean(long[] dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_std_mean_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepDimension, out var mean);
                        if (res == IntPtr.Zero || mean == IntPtr.Zero) { torch.CheckForErrors(); }
                        return (new Tensor(res), new Tensor(mean));
                    }
                }
            }

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public (Tensor std, Tensor mean) std_mean(ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_std_mean_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, unbiased, keepDimension, out var mean);
                        if (res == IntPtr.Zero || mean == IntPtr.Zero) { torch.CheckForErrors(); }
                        return (new Tensor(res), new Tensor(mean));
                    }
                }
            }

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimension">The dimension to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dimension" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public (Tensor std, Tensor mean) std_mean(long dimension, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
                => std_mean(new[] { dimension }, unbiased, keepDimension, type);

            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dimensions" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public (Tensor std, Tensor mean) std_mean((long, long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
                => std_mean(stackalloc[] { dimensions.Item1, dimensions.Item2 }, unbiased, keepDimension, type);


            /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
            /// <remarks>
            /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
            /// Otherwise, the sample deviation is calculated, without any correction.
            /// </remarks>
            /// <param name="dimensions">The dimensions to reduce.</param>
            /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
            /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has <paramref name="dimensions" /> retained or not.</param>
            /// <param name="type"></param>
            /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
            [System.Diagnostics.Contracts.Pure]
            public (Tensor std, Tensor mean) std_mean((long, long, long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
                => std_mean(stackalloc[] { dimensions.Item1, dimensions.Item2, dimensions.Item3 }, unbiased, keepDimension, type);


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sum(IntPtr tensor, bool has_type, sbyte scalar_type);

            /// <summary>
            /// Returns the sum of all elements in the :attr:`input` tensor.
            /// </summary>
            public Tensor sum(ScalarType? type = null)
            {
                var res = THSTensor_sum(Handle, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sum_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

            /// <summary>
            ///  Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long[] dimensions, bool keepdim = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_sum_along_dimensions(Handle, (IntPtr)pdims, dimensions.Length, keepdim, type.HasValue, (sbyte)type.GetValueOrDefault());
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            ///  Returns the sum of each row of the input tensor in the given dimension.
            /// </summary>
            public Tensor sum(long dim, bool keepdim = false, ScalarType? type = null)
            {
                return sum(new long[] { dim }, keepdim, type);
            }

            /// <summary>
            ///  Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long dim0, long dim1, bool keepdim = false, ScalarType? type = null)
            {
                return sum(new long[] { dim0, dim1 }, keepdim, type);
            }

            /// <summary>
            ///  Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long dim0, long dim1, long dim2, bool keepDimension = false, ScalarType? type = null)
            {
                return sum(new long[] { dim0, dim1, dim2 }, keepDimension, type);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_expand(IntPtr tensor, IntPtr psizes, int length, bool isImplicit);

            /// <summary>
            ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
            /// </summary>
            public Tensor expand(long[] sizes, bool isImplicit = false)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_expand(Handle, (IntPtr)psizes, sizes.Length, isImplicit);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Expand this tensor to the same size as other.
            /// </summary>

            public Tensor expand_as(Tensor other) => expand(other.shape);


            /// <summary>
            ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
            /// </summary>
            public Tensor expand(params long[] sizes)
            {
                return expand(sizes, false);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_repeat(IntPtr tensor, IntPtr psizes, int length);

            /// <summary>
            /// Repeats this tensor along the specified dimensions.
            /// </summary>
            public Tensor repeat(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_repeat(Handle, (IntPtr)psizes, sizes.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_broadcast_to(IntPtr tensor, IntPtr psizes, int length);

            /// <summary>
            /// Broadcasts input to the shape shape. Equivalent to calling input.expand(shape).
            /// </summary>
            public Tensor broadcast_to(params long[] shape)
            {
                unsafe {
                    fixed (long* psizes = shape) {
                        var res = THSTensor_broadcast_to(Handle, (IntPtr)psizes, shape.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_movedim(IntPtr tensor, IntPtr src, int src_len, IntPtr dst, int dst_len);

            public Tensor movedim(long[] source, long[] destination)
            {
                unsafe {
                    fixed (long* psource = source, pdest = destination) {
                        var res = THSTensor_movedim(Handle, (IntPtr)psource, source.Length, (IntPtr)pdest, destination.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public Tensor moveaxis(long[] source, long[] destination) => movedim(source, destination);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randn_out(IntPtr psizes, int length, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
            /// </summary>
            public Tensor randn_out(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_randn_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_rand_out(IntPtr psizes, int length, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
            /// </summary>
            public Tensor rand_out(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_rand_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randint_out(long high, IntPtr psizes, int length, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
            /// </summary>
            public Tensor randint_out(long high, long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_randint_out(high, (IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_rand_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1) .
            /// </summary>
            public Tensor rand_like(ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_rand_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_rand_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randn_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
            /// </summary>
            public Tensor randn_like(ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_randn_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_randn_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randint_like(IntPtr input, long low, long high, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly in the range [low,high).
            /// </summary>
            public Tensor randint_like(long low, long high, ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_randint_like(Handle, low, high, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_randint_like(Handle, low, high, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_randperm_out(long n, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
            /// </summary>
            public Tensor randperm_out(long n)
            {
                var res = THSTensor_randperm_out(n, Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bernoulli(IntPtr tensor, IntPtr gen);

            public Tensor bernoulli(torch.Generator? generator = null)
            {
                var res = THSTensor_bernoulli(Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_multinomial(IntPtr tensor, long num_samples, bool replacement, IntPtr gen);

            public Tensor multinomial(long num_samples, bool replacement = false, torch.Generator? generator = null)
            {
                var res = THSTensor_multinomial(Handle, num_samples, replacement, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_poisson(IntPtr tensor, IntPtr gen);

            public Tensor poisson(torch.Generator? generator = null)
            {
                var res = THSTensor_poisson(Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_bernoulli_0(IntPtr tensor, double p, IntPtr gen);

            public Tensor bernoulli_(double p = 0.5, torch.Generator? generator = null)
            {
                var res = THSTensor_bernoulli_0(Handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_bernoulli_1(IntPtr tensor, IntPtr p_tensor, IntPtr gen);

            public Tensor bernoulli_(Tensor p, torch.Generator? generator = null)
            {
                var res = THSTensor_bernoulli_1(Handle, p.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_binomial(IntPtr count, IntPtr prob, IntPtr gen);

            public Tensor binomial(Tensor prob, torch.Generator? generator = null)
            {
                var res = THSTensor_binomial(Handle, prob.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_cauchy_(IntPtr tensor, double median, double sigma, IntPtr gen);

            public Tensor cauchy_(double median = 0.0, double sigma = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_cauchy_(Handle, median, sigma, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_exponential_(IntPtr tensor, double lambda, IntPtr gen);

            public Tensor exponential_(double lambda = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_exponential_(Handle, lambda, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_geometric_(IntPtr tensor, double p, IntPtr gen);

            public Tensor geometric_(double p, torch.Generator? generator = null)
            {
                var res = THSTensor_geometric_(Handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

            public Tensor normal_(double mean = 0.0, double stddev = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_normal_(Handle, mean, stddev, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_log_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

            public Tensor log_normal_(double mean = 0.0, double stddev = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_log_normal_(Handle, mean, stddev, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_random_(IntPtr tensor, double low, double high, IntPtr gen);

            public Tensor random_(double from, double to, torch.Generator? generator = null)
            {
                var res = THSTensor_random_(Handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_uniform_(IntPtr tensor, double low, double high, IntPtr gen);

            public Tensor uniform_(double from, double to, torch.Generator? generator = null)
            {
                var res = THSTensor_uniform_(Handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_arange_out(IntPtr start, IntPtr strp, IntPtr step, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to be filled with with values from interval [start, end) and
            /// common difference step, starting from start.
            /// </summary>
            public Tensor arange_out(Scalar start, Scalar stop, Scalar step)
            {
                var res = THSTensor_arange_out(start.Handle, stop.Handle, step.Handle, Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_permute(IntPtr tensor, IntPtr psizes, int length);

            /// <summary>
            ///  Returns a view of the original tensor with its dimensions permuted.
            /// </summary>
            /// <param name="permutation">The desired ordering of dimensions</param>
            public Tensor permute(params long[] permutation)
            {
                unsafe {
                    fixed (long* pPermutation = permutation) {
                        var res = THSTensor_permute(Handle, (IntPtr)pPermutation, permutation.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_ones_out(IntPtr psizes, int length, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to have the given size with all values set to 1
            /// </summary>
            public Tensor ones(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_ones_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            ///  Create a new tensor filled with ones
            /// </summary>
            public Tensor new_ones(long[] size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.ones(size, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 1-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_ones(new long[] { size }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 2-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long rows, long columns, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_ones(new long[] { rows, columns }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 3-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_ones(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 4-D tensor filled with ones
            /// </summary>
            public Tensor new_ones(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_ones(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_zeros_out(IntPtr psizes, int length, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to have the given size with all values set to 0
            /// </summary>
            public Tensor zeros(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_zeros_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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
            ///  Create a new tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long[] size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.zeros(size, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 1-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_zeros(new long[] { size }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 2-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long rows, long columns, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_zeros(new long[] { rows, columns }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 3-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_zeros(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 4-D tensor filled with zeros
            /// </summary>
            public Tensor new_zeros(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_zeros(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
            }


            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_zeros_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Returns a tensor filled with the scalar value 0, with the same size as input.
            /// </summary>
            public Tensor zeros_like(ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_zeros_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_zeros_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_ones_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Returns a tensor filled with the scalar value 1, with the same size as input.
            /// </summary>
            public Tensor ones_like(ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_ones_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_ones_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            /// <summary>
            ///  Create a new tensor filled with empty
            /// </summary>
            public Tensor new_empty(long[] size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.empty(size, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 1-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long size, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_empty(new long[] { size }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 2-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long rows, long columns, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_empty(new long[] { rows, columns }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 3-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long dim0, long dim1, long dim2, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_empty(new long[] { dim0, dim1, dim2 }, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 4-D tensor filled with empty
            /// </summary>
            public Tensor new_empty(long dim0, long dim1, long dim2, long dim3, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_empty(new long[] { dim0, dim1, dim2, dim3 }, dtype, device, requiresGrad);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_empty_out(IntPtr psizes, int length, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to have the given size with all values uninitialized
            /// </summary>
            public Tensor empty(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_empty_out((IntPtr)psizes, sizes.Length, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_empty_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            ///  Returns an uninitialized tensor with the same size as input.
            /// </summary>
            public Tensor empty_like(ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_empty_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_empty_like(Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_full_out(IntPtr psizes, int length, IntPtr value, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor to have the given size with all values uninitialized
            /// </summary>
            public Tensor full(long[] sizes, Scalar value)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_full_out((IntPtr)psizes, sizes.Length, value.Handle, Handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            ///  Create a new tensor filled with a given value
            /// </summary>
            public Tensor new_full(long[] size, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                if (device == null) device = this.device;
                if (dtype == null) dtype = this.dtype;
                return torch.full(size, value, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 1-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long size, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_full(new long[] { size }, value, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 2-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long rows, long columns, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_full(new long[] { rows, columns }, value, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 3-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long dim0, long dim1, long dim2, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_full(new long[] { dim0, dim1, dim2 }, value, dtype, device, requiresGrad);
            }

            /// <summary>
            ///  Create a new 4-D tensor filled with a given value
            /// </summary>
            public Tensor new_full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                return new_full(new long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requiresGrad);
            }


            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_full_like(IntPtr input, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

            /// <summary>
            /// Returns a tensor with the same size as input filled with 'value.'
            /// </summary>
            public Tensor full_like(Scalar value, ScalarType? dtype = null, torch.Device? device = null, bool requiresGrad = false)
            {
                dtype = (dtype is null) ? this.dtype : dtype;
                device = (device is null) ? this.device : device;

                var result = THSTensor_full_like(Handle, value.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_full_like(Handle, value.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_detach(IntPtr tensor);

            public Tensor detach()
            {
                var res = THSTensor_detach(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_detach_(IntPtr tensor);

            public Tensor detach_()
            {
                var res = THSTensor_detach_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_eye_out(long rows, long columns, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor into a 2-D tensor with ones on the diagonal and zeros elsewhere.
            /// </summary>
            public Tensor eye(long rows, long columns)
            {
                var res = THSTensor_eye_out(rows, columns, Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_scatter(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_scatter_(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_scatter_add(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_scatter_add_(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

            /// <summary>
            ///  Writes all values from the tensor src into self at the indices specified in the index tensor. For each
            ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
            ///  corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter(long dimension, Tensor index, Tensor src)
            {
                var res = THSTensor_scatter(Handle, dimension, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Adds all values from the tensor other into self at the indices specified in the index tensor in a similar fashion as scatter_().
            /// For each value in src, it is added to an index in self which is specified by its index in src for dimension != dim and by the
            /// corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter_(long dimension, Tensor index, Tensor src)
            {
                var res = THSTensor_scatter_(Handle, dimension, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Adds all values from the tensor other into self at the indices specified in the index tensor in a similar fashion as scatter_().
            /// For each value in src, it is added to an index in self which is specified by its index in src for dimension != dim and by the
            /// corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter_add(long dimension, Tensor index, Tensor src)
            {
                var res = THSTensor_scatter_add(Handle, dimension, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            ///  Writes all values from the tensor src into self at the indices specified in the index tensor. For each
            ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
            ///  corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter_add_(long dimension, Tensor index, Tensor src)
            {
                var res = THSTensor_scatter_add_(Handle, dimension, index.Handle, src.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_gather(IntPtr tensor, long dimension, IntPtr index);

            /// <summary>
            /// Gathers values along an axis specified by dim.
            /// </summary>
            public Tensor gather(long dimension, Tensor index)
            {
                var res = THSTensor_gather(Handle, dimension, index.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_flip(IntPtr tensor, IntPtr psizes, int length);

            /// <summary>
            ///  Reverse the order of a n-D tensor along given axis in dims.
            /// </summary>
            public Tensor flip(params long[] sizes)
            {
                unsafe {
                    fixed (long* psizes = sizes) {
                        var res = THSTensor_flip(Handle, (IntPtr)psizes, sizes.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_fliplr(IntPtr tensor);

            /// <summary>
            /// Flip tensor in the left/right direction, returning a new tensor.
            /// </summary>
            public Tensor fliplr()
            {
                var res = THSTensor_fliplr(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_flipud(IntPtr tensor);

            /// <summary>
            /// Flip tensor in the up/down direction, returning a new tensor.
            /// </summary>
            public Tensor flipud()
            {
                var res = THSTensor_flipud(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_nanmean(IntPtr tensor, long dim, bool keepdim, sbyte scalar_type);

            /// <summary>
            /// Returns the mean of the values in input, ignoring NaN values.
            /// </summary>
            public Tensor nanmean(int? dim = null, bool keepdim = false, ScalarType? dtype = null)
            {
                var d = (dim is null) ? -1 : dim.Value;
                var t = (dtype is null) ? this.dtype : dtype.Value;
                var res = THSTensor_nanmean(Handle, d, keepdim, (sbyte)t);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_nanmedian(IntPtr tensor);

            /// <summary>
            /// Returns the median of the values in input, ignoring NaN values.
            /// </summary>
            public Tensor nanmedian()
            {
                var res = THSTensor_nanmedian(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_nansum(IntPtr tensor);

            /// <summary>
            /// Returns the sum of all elements in the input tensor, treating NaN as zero.
            /// </summary>
            public Tensor nansum()
            {
                var res = THSTensor_nansum(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_nan_to_num(IntPtr tensor, IntPtr nan, IntPtr posinf, IntPtr neginf);

            /// <summary>
            /// Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
            /// By default, NaN`s are replaced with zero, positive infinity is replaced with the greatest finite value representable by input’s dtype,
            /// and negative infinity is replaced with the least finite value representable by input’s dtype.
            /// </summary>
            public Tensor nan_to_num(double? nan = null, double? posinf = null, double? neginf = null)
            {
                var _nan = nan.HasValue ? new double[] { nan.Value } : null;
                var _posinf = posinf.HasValue ? new double[] { posinf.Value } : null;
                var _neginf = neginf.HasValue ? new double[] { neginf.Value } : null;
                unsafe {
                    fixed (double* pnan = _nan, pposinf = _posinf, pneginf = _neginf) {
                        var res =
                            THSTensor_nan_to_num(Handle, (IntPtr)pnan, (IntPtr)pposinf, (IntPtr)pneginf);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_nextafter(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Return the next floating-point value after input towards other, elementwise.
            /// </summary>
            public Tensor nextafter(Tensor other)
            {
                var res = THSTensor_nextafter(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_narrow(IntPtr tensor, long dimension, long start, long length);

            /// <summary>
            ///  Returns a new tensor that is a narrowed version of the input along one dimension. The
            /// dimension is input from start to start + length. The
            /// returned tensor and the input tensor share the same underlying storage.
            /// </summary>
            public Tensor narrow(long dimension, long start, long length)
            {
                var res = THSTensor_narrow(Handle, dimension, start, length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_nonzero(IntPtr tensor);

            /// <summary>
            ///
            /// </summary>

            public Tensor nonzero()
            {
                var res = THSTensor_nonzero(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public IList<Tensor> nonzero_as_list()
            {
                var res = THSTensor_nonzero(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }

                var t = new Tensor(res);
                return t.chunk(t.shape[1], dim: 1);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_roll(IntPtr tensor, IntPtr shifts, int shLength, IntPtr dims, long dimLength);

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll(long shifts, long? dims = null)
            {
                if (dims.HasValue) {
                    return _roll(stackalloc long[1] { shifts }, new long[1] { dims.Value });
                }
                else {
                    return _roll(stackalloc long[1] { shifts }, null);
                }
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll((long,long) shifts, (long,long) dims)
            {
                return _roll(stackalloc long[2] { shifts.Item1, shifts.Item2 }, new long[2] { dims.Item1, dims.Item2 });
            }

            /// <summary>
            /// Roll the tensor along the given dimension(s).
            /// Elements that are shifted beyond the last position are re-introduced at the first position.
            /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
            /// </summary>
            public Tensor roll(long[] shifts, long[]? dims = null)
            {
                return _roll(shifts, dims);
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
                        THSTensor_roll(Handle, (IntPtr)sh, shifts.Length, (IntPtr)dm, dmLen);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_slice(IntPtr tensor, long dimension, long start, long length, long step);

            /// <summary>
            /// Returns a new tensor that is a sliced version of the input along one dimension. The
            /// dimension is input from start to finish-1. The
            /// returned tensor and the input tensor share the same underlying storage.
            /// </summary>
            public Tensor slice(long dimension, long start, long finish, long step)
            {
                if (step < 1) throw new ArgumentException($"step is {step}, but it should always be positive.");
                var res = THSTensor_slice(Handle, dimension, start, finish, step);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unsqueeze(IntPtr tensor, long dimension);

            /// <summary>
            ///  Returns a new tensor with a dimension of size one inserted at the specified position.
            ///  The returned tensor shares the same underlying data with this tensor.
            /// </summary>
            public Tensor unsqueeze(long dim)
            {
                var res = THSTensor_unsqueeze(Handle, dim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_unsqueeze_(IntPtr tensor, long dimension);

            /// <summary>
            ///  Returns a new tensor with a dimension of size one inserted at the specified position.
            ///  The returned tensor shares the same underlying data with this tensor.
            /// </summary>
            public Tensor unsqueeze_(long dim)
            {
                var res = THSTensor_unsqueeze_(Handle, dim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_where(IntPtr condition, IntPtr x, IntPtr y);

            public Tensor where(Tensor condition, Tensor other)
            {
                if (condition.dtype != ScalarType.Bool) throw new ArgumentException("The condition to 'where' must be a boolean tensor.");

                var res = THSTensor_where(condition.Handle, this.Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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
            public static Tensor op_MinusMinusGreater(Tensor t, torch.nn.Module m) => m.forward(t);

            public static TensorStringStyle DefaultOutputStyle = TensorStringStyle.Metadata;

            public override string ToString() => ToMetadataString();

            /// <summary>
            /// Tensor-specific ToString(), for backward-compat with pre-0.96.4 versions.
            /// </summary>
            /// <returns></returns>
            public string ToString(bool disamb,
                                   string fltFormat = "g5",
                                   int width = 100,
                                   CultureInfo? cultureInfo = null,
                                   string newLine = "") => disamb ? ToString(TensorStringStyle.Julia, fltFormat, width, cultureInfo, newLine) : ToMetadataString();

            /// <summary>
            /// Tensor-specific ToString()
            /// </summary>
            /// <param name="style">
            /// The style to use -- either 'metadata,' 'julia,' or 'numpy'
            /// </param>
            /// <param name="fltFormat">The floating point format to use for each individual number.</param>
            /// <param name="width">The line width to enforce</param>
            /// <param name="cultureInfo">The culture, which affects how numbers are formatted.</param>
            /// <param name="newLine">The newline string to use, defaults to system default.</param>
            /// <returns></returns>
            public string ToString(TensorStringStyle style,
                                   string fltFormat = "g5",
                                   int width = 100,
                                   CultureInfo? cultureInfo = null,
                                   string newLine = "")
            {
                if (String.IsNullOrEmpty(newLine))
                    newLine = Environment.NewLine;

                if (device_type == DeviceType.META)
                    return ToMetadataString();

                return style switch {
                    TensorStringStyle.Metadata => ToMetadataString(),
                    TensorStringStyle.Julia => ToJuliaString(fltFormat, width, cultureInfo, newLine),
                    TensorStringStyle.Numpy => ToNumpyString(this, ndim, true, fltFormat, cultureInfo, newLine),
                    _ => throw new InvalidEnumArgumentException($"Unsupported tensor string style: {style}")
                };
            }

            /// <summary>
            ///   Get a string representation of the tensor.
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
                var sb = new StringBuilder(isFCreate ? string.Join("", Enumerable.Repeat(' ', (int) (mdim - dim))) : "");
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
                    var newline = string.Join("", Enumerable.Repeat(newLine, (int) dim - 1).ToList());
                    if (currentSize <= 6) {
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

                        sb.Append(string.Join("", Enumerable.Repeat(' ', (int) (mdim - dim))));
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
                    builder.Append(value.ToBoolean());
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
                }
                else {
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


            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_block_diag(IntPtr tensor, int len);

            public static Tensor block_diag(params Tensor[] tensors)
            {
                using (var parray = new PinnedArray<IntPtr>()) {
                    IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                    var res = THSTensor_block_diag(tensorsRef, parray.Array.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_atleast_1d(IntPtr tensor);

            public Tensor atleast_1d()
            {
                var res = THSTensor_atleast_1d(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_atleast_2d(IntPtr tensor);

            public Tensor atleast_2d()
            {
                var res = THSTensor_atleast_2d(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_atleast_3d(IntPtr tensor);

            public Tensor atleast_3d()
            {
                var res = THSTensor_atleast_3d(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_stft(IntPtr x, long n_fft, long hop_length, long win_length, IntPtr window, bool normalized, bool onesided, bool return_complex);

            public Tensor stft(long n_fft, long hop_length = -1, long win_length = -1, Tensor? window = null, bool center = true, PaddingModes pad_mode = PaddingModes.Reflect, bool normalized = false, bool? onesided = null, bool? return_complex = null)
            {
                IntPtr _input = Handle;
                IntPtr _window = (window is null) ? IntPtr.Zero : window.Handle;
                bool _onesided = onesided.HasValue ? onesided.Value : !is_complex();
                bool _return_complex = return_complex.HasValue ? return_complex.Value : is_complex();
                if (center) {
                    long signalDim = dim();
                    long pad = n_fft / 2;
                    var _shape = shape;
                    var extendedShape = new long[] { 1, 1, 1 };
                    for (int i = 0; i < signalDim; i++) {
                        extendedShape[3 + i - signalDim] = _shape[i];
                    }
                    var paddedInput = torch.nn.functional.pad(view(extendedShape), new long[] { pad, pad }, pad_mode);
                    var paddedShape = paddedInput.shape;
                    for (int i = 0; i < signalDim; i++) {
                        _shape[i] = paddedShape[paddedShape.Length + i - signalDim];
                    }
                    paddedInput = paddedInput.view(_shape);
                    _input = paddedInput.Handle;
                }
                var res = THSTensor_stft(_input, n_fft, hop_length, win_length, _window, normalized, _onesided, _return_complex);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_istft(IntPtr x, long n_fft, long hop_length, long win_length, IntPtr window, bool center, bool normalized, bool onesided, long length, bool return_complex);

            public Tensor istft(long n_fft, long hop_length = -1, long win_length = -1, Tensor? window = null, bool center = true, bool normalized = false, bool? onesided = null, long length = -1, bool return_complex = false)
            {
                IntPtr _window = (window is null) ? IntPtr.Zero : window.Handle;
                bool _onesided = onesided.HasValue ? onesided.Value : !is_complex();
                var res = THSTensor_istft(Handle, n_fft, hop_length, win_length, _window, center, normalized, _onesided, length, return_complex);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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
        public static bool is_floating_point(Tensor t) => is_floating_point(t.dtype);
        public static bool is_complex(Tensor t) => is_complex(t.dtype);

        /// <summary>
        /// Returns a view of input as a real tensor.
        /// For an input complex tensor of size m1, m2, …, mi, this function returns a new real tensor of size m1, m2, …, mi, 2, where the last dimension of size 2 represents the real and imaginary components of complex numbers.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static Tensor view_as_real(Tensor input) => input.view_as_real();

        /// <summary>
        /// Returns a view of input as a complex tensor.
        /// For an input complex tensor of size m1, m2, …, mi, 2, this function returns a new complex tensor of size m1, m2, …, mi where the last dimension of the input tensor is expected to represent the real and imaginary components of complex numbers.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static Tensor view_as_complex(Tensor input) => input.view_as_complex();

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
    }
}