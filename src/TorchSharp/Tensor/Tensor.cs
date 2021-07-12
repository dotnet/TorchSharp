// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
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
        public sealed partial class Tensor : IDisposable
        {
            internal IntPtr handle;

            internal Tensor(IntPtr handle)
            {
                this.handle = handle;
            }

            /// <summary>
            ///  TBD
            /// </summary>
            /// <param name="obj"></param>
            /// <returns></returns>
            public override bool Equals(object? obj)
            {
                return (obj is Tensor) && this.Equals((obj as Tensor)!);

            }

            /// <summary>
            ///  TBD
            /// </summary>
            /// <returns></returns>
            public override int GetHashCode()
            {
                return base.GetHashCode();
            }

            /// <summary>
            ///   Finalize the tensor. Releases the tensor and its associated data.
            /// </summary>
            ~Tensor() => Dispose(false);

            public void Dispose()
            {
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
                    THSTensor_dispose(handle);
                    handle = IntPtr.Zero;
                }
            }

            /// <summary>
            /// 
            /// </summary>
            public IntPtr Handle => handle;

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_ndimension(IntPtr handle);

            /// <summary>
            ///  Returns the number of dimensions for this tensor
            /// </summary>
            public long Dimensions => THSTensor_ndimension(handle);

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
            public long NumberOfElements => THSTensor_numel(handle);

            /// <summary>
            ///  Get the number of elements in the tensor.
            /// </summary>
            public long numel() => NumberOfElements;

            /// <summary>
            ///  Get the size of each element in the tensor.
            /// </summary>
            public long ElementSize => THSTensor_element_size(handle);

            public long element_size() => THSTensor_element_size(handle);

            public bool is_integral() => torch.is_integral(dtype);
            public bool is_floating_point() => torch.is_floating_point(dtype);
            public bool is_complex() => torch.is_complex(dtype);

            public bool is_cuda { get { return device.type == DeviceType.CUDA; } } 

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_data(IntPtr handle);

            /// <summary>
            ///  Returns a pointer to the unmanaged data managed by this tensor.
            /// </summary>
            public Span<T> Data<T>()
            {
                if (NumberOfElements > int.MaxValue) {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                if (device_type != DeviceType.CPU) {
                    throw new InvalidOperationException("Reading data from non-CPU memory is not supported. Move or copy the tensor to the cpu before reading.");
                }
                unsafe {
                    var res = THSTensor_data(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    // NOTE: there is no safety here.
                    return new Span<T>((void*)res, (int)NumberOfElements);
                }
            }

            public Span<byte> Bytes()
            {
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

            public void SetBytes(Span<byte> value)
            {
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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_real(IntPtr handle);

            public Tensor Real {
                get {
                    var res = THSTensor_real(Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);

                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_imag(IntPtr handle);

            public Tensor Imag {
                get {
                    var res = THSTensor_imag(Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Returns the singleton value of a scalar tensor.
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <returns>The scalar held in the tensor</returns>
            public T DataItem<T>()
            {
                if (NumberOfElements != 1) throw new ArgumentException("Number of elements in the tensor must be 1");

                return Data<T>()[0];
            }

            /// <summary>
            /// Read the double-precision value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public double ReadCpuDouble(long i) => Data<double>()[(int)i];

            /// <summary>
            /// Read the single-precision float value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public float ReadCpuSingle(long i) => Data<float>()[(int)i];

            /// <summary>
            /// Read the 32-bit integer float value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public int ReadCpuInt32(long i) => Data<int>()[(int)i];

            /// <summary>
            /// Read the 64-bit integer value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public long ReadCpuInt64(long i) => Data<long>()[(int)i];

            /// <summary>
            /// Read the byte value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public byte ReadCpuByte(long i) => Data<byte>()[(int)i];

            /// <summary>
            /// Read the short value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public sbyte ReadCpuSByte(long i) => Data<sbyte>()[(int)i];

            /// <summary>
            /// Read the int16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public short ReadCpuInt16(long i) => Data<short>()[(int)i];

            /// <summary>
            /// Read the Boolean value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
            public bool ReadCpuBool(long i) => Data<bool>()[(int)i];

            [DllImport("LibTorchSharp")]
            static extern float THSTensor_data_idx_float16(IntPtr handle, long i);

            /// <summary>
            /// Read the Float16 value at the given index.
            /// </summary>
            /// <param name="i">The index.</param>
            /// <returns></returns>
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
            /// <returns></returns>
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
            /// <returns></returns>
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
            /// <returns></returns>
            public Tensor fill_(Scalar value)
            {
                var res = THSTensor_fill_(handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern sbyte THSTensor_type(IntPtr handle);

            /// <summary>
            /// Gets the type of the tensor elements.
            /// </summary>
            public ScalarType dtype => (ScalarType)THSTensor_type(handle);

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
                    var res = THSTensor_device_index(handle);
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
                    var res = THSTensor_device_type(handle);
                    torch.CheckForErrors();
                    return (DeviceType)res;
                }
            }

            [DllImport("LibTorchSharp")]
            static extern bool THSTensor_is_sparse(IntPtr handle);

            /// <summary>
            /// Is the tensor a sparse tensor?
            /// </summary>
            public bool IsSparse {
                get {
                    var res = THSTensor_is_sparse(handle);
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
            /// <returns></returns>
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
                THSTensor_save(handle, location);
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
                get { return THSTensor_requires_grad(handle); }
                set {
                    var res = THSTensor_set_requires_grad(handle, value);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                }
            }

            /// <summary>
            /// Adds gradient tracking.
            /// </summary>
            public Tensor with_requires_grad()
            {
                this.requires_grad = true;
                return this;
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cpu(IntPtr handle);

            /// <summary>
            /// Moves the tensor data to the CPU device
            /// </summary>
            /// <returns></returns>
            public Tensor cpu()
            {
                var res = THSTensor_cpu(handle);
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
            /// <returns></returns>
            public Tensor cuda()
            {
                torch.InitializeDeviceType(DeviceType.CUDA);
                var res = THSTensor_cuda(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_device(IntPtr handle, int device_type, int device_index);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_type_and_device(IntPtr handle, sbyte scalar_type, int device_type, int device_index);

            /// <summary>
            /// Cast the tensor to the given element type.
            /// </summary>
            public Tensor to_type(ScalarType type)
            {
                var res = THSTensor_to_type(handle, (sbyte)type);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Moves the tensor data to a specific device.
            /// </summary>
            /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
            /// <param name="deviceIndex">The optional device index.</param>
            /// <returns></returns>
            public Tensor to(DeviceType deviceType, int deviceIndex = -1)
            {
                torch.InitializeDeviceType(deviceType);
                var res = THSTensor_to_device(handle, (int)deviceType, deviceIndex);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Moves the tensor data and casts it to the given element type.
            /// </summary>
            /// <param name="type"></param>
            /// <param name="device"></param>
            /// <returns></returns>
            public Tensor to(ScalarType type, torch.Device device)
            {
                torch.InitializeDevice(device);
                var res = THSTensor_to_type_and_device(handle, (sbyte)type, (int)device.type, device.index);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                //var res = THSTensor_to_type(handle, (sbyte)type);
                //if (res == IntPtr.Zero)
                //    Torch.CheckForErrors();

                //res = THSTensor_to_device(res, (int)device.type, device.index);
                //if (res == IntPtr.Zero)
                //    Torch.CheckForErrors();

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
            /// <returns></returns>
            public Tensor to(string device) => to(new torch.Device(device));

            /// <summary>
            /// Moves the tensor data.
            /// </summary>
            /// <param name="device">The target device</param>
            /// <returns></returns>
            public Tensor to(torch.Device device) => to(device.type, device.index);

            /// <summary>
            /// Moves the tensor data.
            /// </summary>
            /// <param name="other">The tensor serving as a template.</param>
            /// <returns></returns>
            public Tensor to(Tensor other) => to(other.device_type, other.device_index);


            [DllImport("LibTorchSharp")]
            static extern long THSTensor_size(IntPtr handle, long dimension);

            /// <summary>
            ///  Retrieves the size of the specified dimension in the tensor.
            /// </summary>
            /// <param name="dim"></param>
            /// <returns></returns>
            public long size(int dim)
            {
                var res = THSTensor_size(handle, dim);
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
                    THSTensor_sizes(handle, pa.CreateArray);
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
                    var res = THSTensor_indices(handle);
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
                    var res = THSTensor_values(handle);
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

                var res = THSTensor_vander(handle, (N == -1) ? this.size(0) : N, increasing);
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
                    THSTensor_strides(handle, pa.CreateArray);
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
                var res = THSTensor_stride(handle, dim);
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
                        var result = THSTensor_as_strided(handle, (IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, storageOffset);
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
                THSTensor_backward(handle);
                torch.CheckForErrors();
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_to_dense(IntPtr handle);

            /// <summary>
            /// Creates a strided copy of self.
            /// </summary>
            /// <returns></returns>
            public Tensor to_dense()
            {
                var res = THSTensor_to_dense(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_clone(IntPtr handle);

            /// <summary>
            /// Returns a copy of the tensor input.
            /// </summary>
            /// <returns></returns>
            public Tensor clone()
            {
                var res = THSTensor_clone(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, bool non_blocking);

            /// <summary>
            /// Copies the elements from src into self tensor and returns self.
            /// </summary>
            /// <returns></returns>
            /// <remarks>The src tensor must be broadcastable with the target 'this' tensor. It may be of a different data type or reside on a different device.</remarks>
            public Tensor copy_(Tensor source, bool nonBlocking = false)
            {
                var res = THSTensor_copy_(handle, source.Handle, nonBlocking);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_contiguous(IntPtr handle);

            /// <summary>
            /// Returns a contiguous in memory tensor containing the same data as the input tensor.
            /// If tensor is already in the specified memory format, this function returns the original tensor.
            /// </summary>
            /// <returns></returns>
            public Tensor contiguous()
            {
                var res = THSTensor_contiguous(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_grad(IntPtr handle);

            /// <summary>
            /// This attribute is None by default and becomes a Tensor the first time a call to backward() computes gradients for the tensor.
            /// The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.
            /// </summary>
            /// <returns></returns>
            public Tensor grad()
            {
                var res = THSTensor_grad(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
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
            static extern IntPtr THSTensor_set1(IntPtr handle, long i1, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <returns></returns>
            [IndexerName("TensorItems")]
            public Tensor this[long i1] {
                get {
                    var res = THSTensor_get1(handle, i1);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSTensor_set1(handle, i1, value.ToScalar().Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <returns></returns>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2] {
                get {
                    var res = THSTensor_get2(handle, i1, i2);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                set {
                    THSTensor_set2(handle, i1, i2, value.ToScalar().Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <returns></returns>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3] {
                get {
                    var res = THSTensor_get3(handle, i1, i2, i3);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set3(handle, i1, i2, i3, value.ToScalar().Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <param name="i4">The fourth-dimension index</param>
            /// <returns></returns>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3, long i4] {
                get {
                    var res = THSTensor_get4(handle, i1, i2, i3, i4);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set4(handle, i1, i2, i3, i4, value.ToScalar().Handle);
                    torch.CheckForErrors();
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get5(IntPtr handle, long i1, long i2, long i3, long i4, long i5);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set5(IntPtr handle, long i1, long i2, long i3, long i4, long i5, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <param name="i4">The fourth-dimension index</param>
            /// <param name="i5">The fifth-dimension index</param>
            /// <returns></returns>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3, long i4, long i5] {
                get {
                    var res = THSTensor_get5(handle, i1, i2, i3, i4, i5);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set5(handle, i1, i2, i3, i4, i5, value.ToScalar().Handle);
                    torch.CheckForErrors();
                }
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_get6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_set6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6, IntPtr value);

            /// <summary>
            /// Tensor indexer.
            /// </summary>
            /// <param name="i1">The first-dimension index.</param>
            /// <param name="i2">The second-dimension index.</param>
            /// <param name="i3">The third-dimension index</param>
            /// <param name="i4">The fourth-dimension index</param>
            /// <param name="i5">The fifth-dimension index</param>
            /// <param name="i6">The sixth-dimension index</param>
            /// <returns></returns>
            [IndexerName("TensorItems")]
            public Tensor this[long i1, long i2, long i3, long i4, long i5, long i6] {
                get {
                    var res = THSTensor_get6(handle, i1, i2, i3, i4, i5, i6);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
                set {
                    THSTensor_set6(handle, i1, i2, i3, i4, i5, i6, value.ToScalar().Handle);
                    torch.CheckForErrors();
                }
            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions.
            /// </summary>
            /// <returns></returns>
            public Tensor index(params TensorIndex[] indices)
            {
                EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
                unsafe {
                    fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                        fixed (IntPtr* ptrTensors = arrTensors) {
                            var res = THSTensor_index(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length);
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
            /// <returns></returns>
            public Tensor index(params Tensor[] indices)
            {
                return index(indices.Select(t => TensorIndex.Tensor(t)).ToArray());
            }

            /// <summary>
            /// Index into the tensor using Python-like indexing expressions and place a tensor at the index.
            /// </summary>
            /// <returns></returns>
            public Tensor index_put_(Tensor value, params TensorIndex[] indices)
            {
                EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
                unsafe {
                    fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                        fixed (IntPtr* ptrTensors = arrTensors) {
                            var res = THSTensor_index_put_(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
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
            /// <returns></returns>
            public Tensor index_put_(Tensor value, params Tensor[] indices)
            {
                return index_put_(value, indices.Select(t => TensorIndex.Tensor(t)).ToArray());
            }


            /// <summary>
            /// Index into the tensor using Python-like indexing expressions and place a scalar tensor at the index.
            /// </summary>
            /// <returns></returns>
            public Tensor index_put_(Scalar value, params TensorIndex[] indices)
            {
                EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
                unsafe {
                    fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                        fixed (IntPtr* ptrTensors = arrTensors) {
                            var res = THSTensor_index_put_scalar_(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
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
            /// <returns></returns>
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
            /// <returns></returns>
            public Tensor index_select(long dimension, Tensor index)
            {
                var res = THSTensor_index_select(handle, dimension, index.Handle);
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
            /// <returns></returns>
            public Tensor take(Tensor index)
            {
                var res = THSTensor_take(handle, index.Handle);
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
            /// <returns></returns>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(Tensor indices)
            {
                var res = THSTensor_take_along_dim_dflt(handle, indices.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <returns></returns>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(IEnumerable<long> indices) => take_along_dim(Int64Tensor.from(indices.ToArray()));

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <param name="dimension">Dimension to select along.</param>
            /// <returns></returns>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(Tensor indices, long dimension)
            {
                var res = THSTensor_take_along_dim(handle, indices.Handle, dimension);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Selects values from input at the 1-dimensional indices from indices along the given dim.
            /// </summary>
            /// <param name="indices">The indices into input. Must have long dtype.</param>
            /// <param name="dim">Dimension to select along.</param>
            /// <returns></returns>
            /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
            public Tensor take_along_dim(IEnumerable<long> indices, long dim) => take_along_dim(Int64Tensor.from(indices.ToArray()), dim);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_reshape(IntPtr tensor, IntPtr shape, int length);

            /// <summary>
            /// Returns a tensor with the same data and number of elements as self but with the specified shape.
            /// </summary>
            /// <param name="shape">The new tensor shape.</param>
            /// <returns></returns>
            public Tensor reshape(params long[] shape)
            {
                unsafe {
                    fixed (long* pshape = shape) {
                        var res = THSTensor_reshape(handle, (IntPtr)pshape, shape.Length);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_squeeze(IntPtr tensor, long dimension);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_squeeze_no_dim(IntPtr tensor);

            /// <summary>
            /// Returns a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
            /// </summary>
            /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
            /// <returns></returns>
            public Tensor squeeze(long? dim = null)
            {
                var res = dim.HasValue ? THSTensor_squeeze(handle, dim.Value) : THSTensor_squeeze_no_dim(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_t(IntPtr tensor);

            /// <summary>
            /// Expects input to be 1- or 2-D tensor and transposes dimensions 0 and 1.
            /// </summary>
            /// <returns></returns>
            public Tensor t()
            {
                var res = THSTensor_t(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_transpose(IntPtr tensor, long dim1, long dim2);

            /// <summary>
            /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
            /// </summary>
            /// <param name="dim0"></param>
            /// <param name="dim1"></param>
            /// <returns></returns>
            public Tensor transpose(long dim0, long dim1)
            {
                var res = THSTensor_transpose(handle, dim0, dim1);
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
            /// <returns></returns>
            public Tensor tril(long diagonal = 0)
            {
                var res = THSTensor_tril(handle, diagonal);
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
            /// <returns></returns>
            public Tensor triu(long diagonal = 0)
            {
                var res = THSTensor_triu(handle, diagonal);
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
            /// <returns></returns>
            public Tensor transpose_(long dim0, long dim1)
            {
                return new Tensor(THSTensor_transpose_(handle, dim0, dim1));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_view(IntPtr tensor, IntPtr shape, int length);

            /// <summary>
            /// Returns a new tensor with the same data as the input tensor but of a different shape.
            /// </summary>
            /// <param name="shape">The shape of the view</param>
            /// <returns></returns>
            public Tensor view(params long[] shape)
            {
                unsafe {
                    fixed (long* pshape = shape) {
                        var res = THSTensor_view(handle, (IntPtr)pshape, shape.Length);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_view_as_complex(IntPtr tensor);

            /// <summary>
            /// Returns a view of input as a complex tensor. 
            /// </summary>
            public Tensor view_as_complex()
            {
                var result = THSTensor_view_as_complex(handle);
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
                var result = THSTensor_view_as_real(handle);
                if (result == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_all(IntPtr tensor);

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public Tensor all()
            {
                var res = THSTensor_all(handle);
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
            /// <returns></returns>
            public Tensor all(long dimension, bool keepDim = false)
            {
                var res = THSTensor_all_along_dimension(handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_amax(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim);

            /// <summary>
            /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
            /// <returns></returns>
            public Tensor amax(long[] dims, bool keepDim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSTensor_amax(handle, (IntPtr)pdims, dims.Length, keepDim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_amin(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim);

            /// <summary>
            /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
            /// </summary>
            /// <param name="dims">The dimension or dimensions to reduce.</param>
            /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
            /// <returns></returns>
            public Tensor amin(long[] dims, bool keepDim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSTensor_amin(handle, (IntPtr)pdims, dims.Length, keepDim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_any(IntPtr tensor);

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public Tensor any()
            {
                var res = THSTensor_any(handle);
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
            /// <returns></returns>
            public Tensor any(long dimension, bool keepDim = false)
            {
                var res = THSTensor_any_along_dimension(handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argmax(IntPtr tensor);

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public Tensor argmax()
            {
                var res = THSTensor_argmax(handle);
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
            /// <returns></returns>
            public Tensor argmax(long dimension, bool keepDim = false)
            {
                var res = THSTensor_argmax_along_dimension(handle, dimension, keepDim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_argmin(IntPtr tensor);

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public Tensor argmin()
            {
                var res = THSTensor_argmin(handle);
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
            /// <returns></returns>
            public Tensor argmin(long dimension, bool keepDim = false)
            {
                var res = THSTensor_argmin_along_dimension(handle, dimension, keepDim);
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
            /// <returns></returns>
            public Tensor argsort(long dimension = -1, bool descending = false)
            {
                var res = THSTensor_argsort(handle, dimension, descending);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_deg2rad(IntPtr tensor);

            /// <summary>
            /// Convert each element from degrees to radians.
            /// </summary>
            /// <returns></returns>
            public Tensor deg2rad()
            {
                var res = THSTensor_deg2rad(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rad2deg(IntPtr tensor);

            /// <summary>
            /// Convert each element from radians to degrees.
            /// </summary>
            /// <returns></returns>
            public Tensor rad2deg()
            {
                var res = THSTensor_rad2deg(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_copysign(IntPtr tensor, IntPtr other);

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public Tensor copysign(Tensor other)
            {
                var res = THSTensor_copysign(handle, other.handle);
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
                        var res = THSTensor_count_nonzero(handle, (IntPtr)pdims, dims is null ? 0 : dims.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tile(IntPtr tensor, IntPtr reps, int reps_len);

            /// <summary>
            /// Constructs a tensor by repeating the elements of input. The reps argument specifies the number of repetitions in each dimension.
            /// </summary>
            /// <param name="reps">The number of repetitions per dimension.</param>
            /// <returns></returns>
            public Tensor tile(long[] reps)
            {
                unsafe {
                    fixed (long* pdims = reps) {
                        var res = THSTensor_tile(handle, (IntPtr)pdims, reps.Length);
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
            /// <returns></returns>
            public Tensor digamma()
            {
                var res = THSTensor_digamma(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_digamma_(IntPtr tensor);

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input, in place.
            /// </summary>
            /// <returns></returns>
            public Tensor digamma_()
            {
                var res = THSTensor_digamma_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lgamma(IntPtr tensor);

            /// <summary>
            /// Computes the logarithm of the gamma function on input.
            /// </summary>
            /// <returns></returns>
            public Tensor lgamma()
            {
                var res = THSTensor_lgamma(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lgamma_(IntPtr tensor);

            /// <summary>
            /// Computes the logarithm of the gamma function on input, in place.
            /// </summary>
            /// <returns></returns>
            public Tensor lgamma_()
            {
                var res = THSTensor_lgamma_(handle);
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
            /// <returns></returns>
            public Tensor mvlgamma(long p)
            {
                var res = THSTensor_mvlgamma(handle, p);
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
            /// <returns></returns>
            public Tensor mvlgamma_(long p)
            {
                var res = THSTensor_mvlgamma_(handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_polygamma(IntPtr tensor, long p);

            public Tensor polygamma(long p)
            {
                var res = THSTensor_polygamma(handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_polygamma_(IntPtr tensor, long p);

            public Tensor polygamma_(long p)
            {
                var res = THSTensor_polygamma_(handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_positive(IntPtr tensor);

            /// <summary>
            /// Returns input. Throws a runtime error if input is a bool tensor. 
            /// </summary>
            /// <returns></returns>
            public Tensor positive()
            {
                if (this.dtype == ScalarType.Bool) throw new ArgumentException("Boolean tensor");
                var res = THSTensor_positive(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_softplus(IntPtr tensor);

            public Tensor softplus()
            {
                var res = THSTensor_softplus(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ravel(IntPtr tensor);

            public Tensor ravel()
            {
                var res = THSTensor_ravel(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu(IntPtr tensor);

            public Tensor relu()
            {
                var res = THSTensor_relu(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu_(IntPtr tensor);

            public Tensor relu_()
            {
                var res = THSTensor_relu_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu6(IntPtr tensor);

            public Tensor relu6()
            {
                var res = THSTensor_relu6(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_relu6_(IntPtr tensor);

            public Tensor relu6_()
            {
                var res = THSTensor_relu6_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_celu(IntPtr tensor);

            public Tensor celu()
            {
                var res = THSTensor_celu(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_celu_(IntPtr tensor);

            public Tensor celu_()
            {
                var res = THSTensor_celu_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_elu(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

            public Tensor elu(Scalar alpha, Scalar scale, Scalar input_scale)
            {
                var res = THSTensor_elu(handle, alpha.Handle, scale.Handle, input_scale.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_elu_(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

            public Tensor elu_(Scalar alpha, Scalar scale, Scalar input_scale)
            {
                var res = THSTensor_elu_(handle, alpha.Handle, scale.Handle, input_scale.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gelu(IntPtr tensor);

            public Tensor gelu()
            {
                var res = THSTensor_gelu(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardsigmoid(IntPtr tensor);

            public Tensor hardsigmoid()
            {
                var res = THSTensor_hardsigmoid(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardsigmoid_(IntPtr tensor);

            public Tensor hardsigmoid_()
            {
                var res = THSTensor_hardsigmoid_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardswish(IntPtr tensor);

            public Tensor hardswish()
            {
                var res = THSTensor_hardswish(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardswish_(IntPtr tensor);

            public Tensor hardswish_()
            {
                var res = THSTensor_hardswish_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardtanh(IntPtr tensor, IntPtr min, IntPtr max);

            public Tensor hardtanh(Scalar min, Scalar max)
            {
                var res = THSTensor_hardtanh(handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hardtanh_(IntPtr tensor, IntPtr min, IntPtr max);

            public Tensor hardtanh_(Scalar min, Scalar max)
            {
                var res = THSTensor_hardtanh_(handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_heaviside(IntPtr tensor, IntPtr other);

            public Tensor heaviside(Tensor other)
            {
                var res = THSTensor_heaviside(handle, other.handle);
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
            /// <returns></returns>
            public Tensor igamma(Tensor other)
            {
                var res = THSTensor_igamma(handle, other.handle);
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
            /// <returns></returns>
            public Tensor igammac(Tensor other)
            {
                var res = THSTensor_igammac(handle, other.handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_i0(IntPtr tensor);

            /// <summary>
            /// Computes the zeroth order modified Bessel function of the first kind for each element of input.
            /// </summary>
            /// <returns></returns>
            public Tensor i0()
            {
                var res = THSTensor_i0(handle);
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
            /// <returns></returns>
            public Tensor isclose(Tensor other, double rtol = 1e-05, double atol = 1e-08, bool nanEqual = false)
            {
                var res = THSTensor_isclose(handle, other.Handle, rtol, atol, nanEqual);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isinf(IntPtr tensor);

            public Tensor isinf()
            {
                var res = THSTensor_isinf(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isfinite(IntPtr tensor);

            public Tensor isfinite()
            {
                var res = THSTensor_isfinite(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isposinf(IntPtr tensor);

            public Tensor isposinf()
            {
                var res = THSTensor_isposinf(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isneginf(IntPtr tensor);

            public Tensor isneginf()
            {
                var res = THSTensor_isneginf(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_isreal(IntPtr tensor);

            public Tensor isreal()
            {
                var res = THSTensor_isreal(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_leaky_relu(IntPtr tensor, IntPtr negative_slope);

            public Tensor leaky_relu(Scalar negative_slope)
            {
                var res = THSTensor_leaky_relu(handle, negative_slope.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_leaky_relu_(IntPtr tensor, IntPtr negative_slope);

            public Tensor leaky_relu_(Scalar negative_slope)
            {
                var res = THSTensor_leaky_relu_(handle, negative_slope.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_selu(IntPtr tensor);

            public Tensor selu()
            {
                var res = THSTensor_selu(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_selu_(IntPtr tensor);

            public Tensor selu_()
            {
                var res = THSTensor_selu_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_silu(IntPtr tensor);

            public Tensor silu()
            {
                var res = THSTensor_silu(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_silu_(IntPtr tensor);

            public Tensor silu_()
            {
                var res = THSTensor_silu_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log_sigmoid(IntPtr tensor);

            public Tensor log_sigmoid()
            {
                var res = THSTensor_log_sigmoid(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lerp(IntPtr tensor, IntPtr end, IntPtr weight);

            public Tensor lerp(Tensor end, Tensor weight)
            {
                var res = THSTensor_lerp(handle, end.Handle, weight.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lerp_(IntPtr tensor, IntPtr end, IntPtr weight);

            public Tensor lerp_(Tensor end, Tensor weight)
            {
                var res = THSTensor_lerp_(handle, end.Handle, weight.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
                float alpha);

            public Tensor baddbmm(Tensor batch2, Tensor mat, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_baddbmm(handle, batch2.Handle, mat.Handle, beta, alpha);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

            public Tensor bmm(Tensor batch2)
            {
                var res = THSTensor_bmm(handle, batch2.Handle);
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
            /// <returns></returns>
            public Tensor bucketize(Tensor boundaries, bool outInt32 = false, bool right = false)
            {
                var res = THSTensor_bucketize(handle, boundaries.Handle, outInt32, right);
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
                var res = THSTensor_bincount(handle, weightsHandle, minlength);
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
            static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

            public Tensor clamp(Scalar min, Scalar max)
            {
                var res = THSTensor_clamp(handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor clip(Scalar min, Scalar max) => clamp(min, max);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

            public Tensor clamp_(Scalar min, Scalar max)
            {
                var res = THSTensor_clamp_(handle, min.Handle, max.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

            public Tensor clamp_max(Scalar max)
            {
                var res = THSTensor_clamp_max(handle, max.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

            public Tensor clamp_max_(Scalar max)
            {
                var res = THSTensor_clamp_max_(handle, max.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

            public Tensor clamp_min(Scalar min)
            {
                var res = THSTensor_clamp_min(handle, min.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

            public Tensor clamp_min_(Scalar min)
            {
                var res = THSTensor_clamp_min_(handle, min.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diff(IntPtr tensor, long n, long dim, IntPtr prepend, IntPtr append);

            public Tensor diff(long n = 1, long dim = -1, Tensor? prepend = null, Tensor? append = null)
            {
                if (n != 1) throw new NotImplementedException("Tensor.diff with n != 1");
                var res = THSTensor_diff(handle, n, dim, (prepend is Tensor) ? (IntPtr)prepend.handle : IntPtr.Zero, (append is Tensor) ? (IntPtr)append.handle : IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diag(IntPtr tensor, long dimension);

            public Tensor diag(long dimension = 0)
            {
                var res = THSTensor_diag(handle, dimension);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diagflat(IntPtr tensor, long offset);

            public Tensor diagflat(long offset = 0)
            {
                var res = THSTensor_diagflat(handle, offset);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_diagonal(IntPtr tensor, long offset, long dim1, long dim2);

            public Tensor diagonal(long offset = 0, long dim1 = 0, long dim2 = 0)
            {
                var res = THSTensor_diagonal(handle, offset, dim1, dim2);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erf(IntPtr tensor);

            public Tensor erf()
            {
                var res = THSTensor_erf(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erf_(IntPtr tensor);

            public Tensor erf_()
            {
                var res = THSTensor_erf_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfc(IntPtr tensor);

            public Tensor erfc()
            {
                var res = THSTensor_erfc(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfc_(IntPtr tensor);

            public Tensor erfc_()
            {
                var res = THSTensor_erfc_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfinv(IntPtr tensor);

            public Tensor erfinv()
            {
                var res = THSTensor_erfinv(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_erfinv_(IntPtr tensor);

            public Tensor erfinv_()
            {
                var res = THSTensor_erfinv_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq(IntPtr tensor, IntPtr trg);

            public Tensor eq(Tensor target)
            {
                var res = THSTensor_eq(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor equal(Tensor target) => eq(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq_(IntPtr tensor, IntPtr trg);

            public Tensor eq_(Tensor target)
            {
                var res = THSTensor_eq_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq_scalar(IntPtr tensor, IntPtr trg);

            public Tensor eq(Scalar target)
            {
                var res = THSTensor_eq_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eq_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor eq_(Scalar target)
            {
                var res = THSTensor_eq_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern bool THSTensor_equal(IntPtr tensor, IntPtr trg);

            public bool Equals(Tensor target)
            {
                var res = THSTensor_equal(handle, target.Handle);
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
            /// <returns></returns>
            public bool allclose(Tensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
            {
                var res = THSTensor_allclose(handle, target.Handle, rtol, atol, equal_nan);
                torch.CheckForErrors();
                return res;
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge(IntPtr tensor, IntPtr trg);

            public Tensor ge(Tensor target)
            {
                var res = THSTensor_ge(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor greater_equal(Tensor target) => ge(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge_(IntPtr tensor, IntPtr trg);

            public Tensor ge_(Tensor target)
            {
                var res = THSTensor_ge_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge_scalar(IntPtr tensor, IntPtr trg);

            public Tensor ge(Scalar target)
            {
                var res = THSTensor_ge_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ge_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor ge_(Scalar target)
            {
                var res = THSTensor_ge_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt(IntPtr tensor, IntPtr trg);

            public Tensor gt(Tensor target)
            {
                var res = THSTensor_gt(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor greater(Tensor target) => gt(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt_(IntPtr tensor, IntPtr trg);

            public Tensor gt_(Tensor target)
            {
                var res = THSTensor_gt_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt_scalar(IntPtr tensor, IntPtr trg);

            public Tensor gt(Scalar target)
            {
                var res = THSTensor_gt_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gt_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor gt_(Scalar target)
            {
                var res = THSTensor_gt_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_kron(IntPtr tensor, IntPtr other);

            public Tensor kron(Tensor other)
            {
                var res = THSTensor_kron(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lcm(IntPtr tensor, IntPtr other);

            public Tensor lcm(Tensor other)
            {
                var res = THSTensor_lcm(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lcm_(IntPtr tensor, IntPtr other);

            public Tensor lcm_(Tensor other)
            {
                var res = THSTensor_lcm_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ldexp(IntPtr right, IntPtr left);

            public Tensor ldexp(Tensor other)
            {
                var res = THSTensor_ldexp(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le(IntPtr tensor, IntPtr trg);

            public Tensor le(Tensor target)
            {
                var res = THSTensor_le(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less_equal(Tensor target) => le(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le_(IntPtr tensor, IntPtr trg);

            public Tensor le_(Tensor target)
            {
                var res = THSTensor_le_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le_scalar(IntPtr tensor, IntPtr trg);

            public Tensor le(Scalar target)
            {
                var res = THSTensor_le_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_le_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor le_(Scalar target)
            {
                var res = THSTensor_le_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt(IntPtr tensor, IntPtr trg);

            public Tensor lt(Tensor target)
            {
                var res = THSTensor_lt(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor less(Tensor target) => lt(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt_(IntPtr tensor, IntPtr trg);

            public Tensor lt_(Tensor target)
            {
                var res = THSTensor_lt_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt_scalar(IntPtr tensor, IntPtr trg);

            public Tensor lt(Scalar target)
            {
                var res = THSTensor_lt_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_lt_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor lt_(Scalar target)
            {
                var res = THSTensor_lt_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_fill(IntPtr tensor, IntPtr mask, IntPtr value);

            public Tensor masked_fill(Tensor mask, Scalar value)
            {
                var res = THSTensor_masked_fill(handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_scatter(IntPtr tensor, IntPtr mask, IntPtr value);

            public Tensor masked_scatter(Tensor mask, Tensor value)
            {
                var res = THSTensor_masked_scatter(handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_scatter_(IntPtr tensor, IntPtr mask, IntPtr value);

            public Tensor masked_scatter_(Tensor mask, Tensor value)
            {
                var res = THSTensor_masked_scatter_(handle, mask.Handle, value.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_masked_select(IntPtr tensor, IntPtr mask);

            public Tensor masked_select(Tensor mask)
            {
                if (mask.dtype != ScalarType.Bool) throw new ArgumentException("The mask tensor must be Boolean.");
                var res = THSTensor_masked_select(handle, mask.Handle);
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
                    THSTensor_topk(handle, pa.CreateArray, k, dimension, largest, sorted);
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
                    THSTensor_unbind(handle, pa.CreateArray, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

            /// <summary>
            /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
            /// </summary>
            /// <param name="size">The size of a single chunk</param>
            /// <param name="dimension">The dimension along which to split the tensor.</param>
            /// <returns></returns>
            public Tensor[] split(long size, int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_split_with_size(handle, pa.CreateArray, size, dimension);
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
            /// <returns></returns>
            public Tensor[] split(long[] sizes, int dimension = 0)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            THSTensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
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
                    THSTensor_tensor_split_with_size(handle, pa.CreateArray, size, dimension);
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
                            THSTensor_tensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
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
                    THSTensor_tensor_split_with_tensor_sizes(handle, pa.CreateArray, indices.Handle, dimension);
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
            /// <returns></returns>
            public Tensor[] vsplit(long size)
            {
                if (this.shape[0] % size != 0) throw new ArgumentException("The first dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_vsplit_with_size(handle, pa.CreateArray, size);
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
            /// <returns></returns>
            public Tensor[] vsplit(params long[] sizes)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            THSTensor_vsplit_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
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
            /// <returns></returns>
            public Tensor[] vsplit(Tensor indices) => tensor_split(indices, 0);


            [DllImport("LibTorchSharp")]
            static extern void THSTensor_hsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

            /// <summary>
            /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to sizes.
            /// </summary>
            /// <param name="size">The size of each chunk</param>
            /// <returns></returns>
            public Tensor[] hsplit(long size)
            {
                if (this.shape[1] % size != 0) throw new ArgumentException("The second dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_hsplit_with_size(handle, pa.CreateArray, size);
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
            /// <returns></returns>
            public Tensor[] hsplit(params long[] sizes)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    unsafe {
                        fixed (long* psizes = sizes) {
                            THSTensor_hsplit_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
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
            /// <returns></returns>
            public Tensor[] hsplit(Tensor indices) => tensor_split(indices, 1);

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_dsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

            /// <summary>
            /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
            /// </summary>
            /// <param name="size">The size of each chunk</param>
            /// <returns></returns>
            public Tensor[] dsplit(long size)
            {
                if (this.shape[2] % size != 0) throw new ArgumentException("The third dimension must be evenly divisible by the size");
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_dsplit_with_size(handle, pa.CreateArray, size);
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
                            THSTensor_dsplit_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length);
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
            /// <returns></returns>
            /// <remarks>The last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.</remarks>
            public Tensor[] chunk(long chunks, long dim = 0L)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_chunk(handle, pa.CreateArray, chunks, dim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return ptrArray.Select(x => new Tensor(x)).ToArray();
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_max(IntPtr tensor);

            public Tensor max()
            {
                var res = THSTensor_max(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_max_elementwise(IntPtr tensor, IntPtr other);

            public Tensor max(Tensor other)
            {
                var res = THSTensor_max_elementwise(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_maximum(IntPtr tensor, IntPtr other);

            public Tensor maximum(Tensor other)
            {
                var res = THSTensor_maximum(handle, other.handle);
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
                    THSTensor_max_along_dimension(handle, pa.CreateArray, dimension, keepDim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mean(IntPtr tensor);


            public Tensor mean()
            {
                var res = THSTensor_mean(handle);
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
            /// <returns></returns>
            Tensor quantile(Tensor q, long dim = -1, bool keepdim = false)
            {
                var res = THSTensor_quantile(handle, q.handle, dim, keepdim);
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
            /// <returns></returns>
            Tensor nanquantile(Tensor q, long dim = -1, bool keepdim = false)
            {
                var res = THSTensor_nanquantile(handle, q.handle, dim, keepdim);
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
            /// <returns></returns>

            public (Tensor values, Tensor indices) mode(long dim = -1L, bool keepdim = false)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_mode(handle, pa.CreateArray, dim, keepdim);
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
                        var res = THSTensor_mean_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_median(IntPtr tensor);

            public Tensor median()
            {
                var res = THSTensor_median(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_min(IntPtr tensor);

            public Tensor min()
            {
                var res = THSTensor_min(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_minimum(IntPtr tensor, IntPtr other);

            public Tensor min(Tensor other)
            {
                var res = THSTensor_minimum(handle, other.handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_min_elementwise(IntPtr tensor, IntPtr other);

            public Tensor minimum(Tensor other)
            {
                var res = THSTensor_min_elementwise(handle, other.Handle);
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
                    THSTensor_min_along_dimension(handle, pa.CreateArray, dimension, keepDim);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_msort(IntPtr tensor);

            public Tensor msort()
            {
                var res = THSTensor_msort(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sort(IntPtr tensor, long dim, bool descending, bool stable, out IntPtr indices);

            public (Tensor Values, Tensor Indices) sort(long dim = -1, bool descending = false, bool stable = false)
            {
                var res = THSTensor_sort(handle, dim, descending, stable, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne(IntPtr tensor, IntPtr trg);

            public Tensor ne(Tensor target)
            {
                var res = THSTensor_ne(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor not_equal(Tensor target) => ne(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne_(IntPtr tensor, IntPtr trg);

            public Tensor ne_(Tensor target)
            {
                var res = THSTensor_ne_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor not_equal_(Tensor target) => ne_(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne_scalar(IntPtr tensor, IntPtr trg);

            public Tensor ne(Scalar target)
            {
                var res = THSTensor_ne_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ne_scalar_(IntPtr tensor, IntPtr trg);

            public Tensor ne_(Scalar target)
            {
                var res = THSTensor_ne_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_dist(IntPtr tensor, IntPtr other, float p);

            public Tensor dist(Tensor other, float p = 2.0f)
            {
                var res = THSTensor_dist(handle, other.Handle, p);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_norm(IntPtr tensor, float p);

            public Tensor norm(float p = 2.0f)
            {
                var res = THSTensor_norm(handle, p);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_norm_along_dimension(IntPtr tensor, int dimension, bool keepdim, float p);

            public Tensor norm(int dimension, bool keepdim = false, float p = 2.0f)
            {
                var res = THSTensor_norm_along_dimension(handle, dimension, keepdim, p);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_outer(IntPtr input, IntPtr vec2);

            public Tensor outer(Tensor vec2)
            {
                var res = THSTensor_outer(handle, vec2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_inner(IntPtr input, IntPtr vec2);

            public Tensor inner(Tensor vec2)
            {
                var res = THSTensor_inner(handle, vec2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_inverse(IntPtr tensor);

            public Tensor inverse()
            {
                var res = THSTensor_inverse(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_prelu(IntPtr tensor, IntPtr trg);

            public Tensor prelu(Tensor target)
            {
                var res = THSTensor_prelu(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmax(IntPtr tensor, IntPtr trg);

            public Tensor fmax(Tensor target)
            {
                var res = THSTensor_fmax(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmin(IntPtr tensor, IntPtr trg);

            public Tensor fmin(Tensor target)
            {
                var res = THSTensor_fmin(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_renorm(IntPtr tensor, float p, long dim, float maxnorm);

            public Tensor renorm(Scalar scalar, float p, long dim, float maxnorm)
            {
                var res = THSTensor_renorm(handle, p, dim, maxnorm);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sigmoid(IntPtr tensor);

            public Tensor sigmoid()
            {
                var res = THSTensor_sigmoid(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sigmoid_(IntPtr tensor);

            public Tensor sigmoid_()
            {
                var res = THSTensor_sigmoid_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_std(IntPtr tensor);

            public Tensor std()
            {
                var res = THSTensor_std(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_std_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool unbiased, bool keepdim);

            public Tensor std(long[] dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_std_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, unbiased, keepDimension);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sum(IntPtr tensor, bool has_type, sbyte scalar_type);

            /// <summary>
            /// Returns the sum of all elements in the :attr:`input` tensor.
            /// </summary>
            public Tensor sum(ScalarType? type = null)
            {
                var res = THSTensor_sum(handle, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sum_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

            /// <summary>
            ///  Returns the sum of each row of the input tensor in the given dimensions.
            /// </summary>
            public Tensor sum(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
            {
                unsafe {
                    fixed (long* pdims = dimensions) {
                        var res = THSTensor_sum_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
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
                        var res = THSTensor_expand(handle, (IntPtr)psizes, sizes.Length, isImplicit);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Expand this tensor to the same size as other.
            /// </summary>
            /// <returns></returns>
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
                        var res = THSTensor_repeat(handle, (IntPtr)psizes, sizes.Length);
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
                        var res = THSTensor_broadcast_to(handle, (IntPtr)psizes, shape.Length);
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
                        var res = THSTensor_movedim(handle, (IntPtr)psource, source.Length, (IntPtr)pdest, destination.Length);
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
                        var res = THSTensor_randn_out((IntPtr)psizes, sizes.Length, handle);
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
                        var res = THSTensor_rand_out((IntPtr)psizes, sizes.Length, handle);
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
                        var res = THSTensor_randint_out(high, (IntPtr)psizes, sizes.Length, handle);
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

                var result = THSTensor_rand_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_rand_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
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

                var result = THSTensor_randn_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_randn_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
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

                var result = THSTensor_randint_like(handle, low, high, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_randint_like(handle, low, high, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
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
                var res = THSTensor_randperm_out(n, handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bernoulli(IntPtr tensor, IntPtr gen);

            public Tensor bernoulli(torch.Generator? generator = null)
            {
                var res = THSTensor_bernoulli(handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_multinomial(IntPtr tensor, long num_samples, bool replacement, IntPtr gen);

            public Tensor multinomial(long num_samples, bool replacement = false, torch.Generator ? generator = null)
            {
                var res = THSTensor_multinomial(handle, num_samples, replacement, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_poisson(IntPtr tensor, IntPtr gen);

            public Tensor poisson(torch.Generator? generator = null)
            {
                var res = THSTensor_poisson(handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_bernoulli_0(IntPtr tensor, double p, IntPtr gen);

            public Tensor bernoulli_(double p = 0.5, torch.Generator? generator = null)
            {
                var res = THSTensor_bernoulli_0(handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_bernoulli_1(IntPtr tensor, IntPtr p_tensor, IntPtr gen);

            public Tensor bernoulli_(Tensor p, torch.Generator? generator = null)
            {
                var res = THSTensor_bernoulli_1(handle, p.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_cauchy_(IntPtr tensor, double median, double sigma, IntPtr gen);

            public Tensor cauchy_(double median = 0.0, double sigma = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_cauchy_(handle, median, sigma, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_exponential_(IntPtr tensor, double lambda, IntPtr gen);

            public Tensor exponential_(double lambda = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_exponential_(handle, lambda, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_geometric_(IntPtr tensor, double p, IntPtr gen);

            public Tensor geometric_(double p, torch.Generator? generator = null)
            {
                var res = THSTensor_geometric_(handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

            public Tensor normal_(double mean = 0.0, double stddev = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_normal_(handle, mean, stddev, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_log_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

            public Tensor log_normal_(double mean = 0.0, double stddev = 1.0, torch.Generator? generator = null)
            {
                var res = THSTensor_log_normal_(handle, mean, stddev, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_random_(IntPtr tensor, double low, double high, IntPtr gen);

            public Tensor random_(double from, double to, torch.Generator? generator = null)
            {
                var res = THSTensor_random_(handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_uniform_(IntPtr tensor, double low, double high, IntPtr gen);

            public Tensor uniform_(double from, double to, torch.Generator? generator = null)
            {
                var res = THSTensor_uniform_(handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
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
                var res = THSTensor_arange_out(start.Handle, stop.Handle, step.Handle, handle);
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
                        var res = THSTensor_permute(handle, (IntPtr)pPermutation, permutation.Length);
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
                        var res = THSTensor_ones_out((IntPtr)psizes, sizes.Length, handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
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
                        var res = THSTensor_zeros_out((IntPtr)psizes, sizes.Length, handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
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

                var result = THSTensor_zeros_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_zeros_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
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

                var result = THSTensor_ones_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_ones_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
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
                        var res = THSTensor_empty_out((IntPtr)psizes, sizes.Length, handle);
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

                var result = THSTensor_empty_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_empty_like(handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
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
                        var res = THSTensor_full_out((IntPtr)psizes, sizes.Length, value.Handle, handle);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
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

                var result = THSTensor_full_like(handle, value.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                if (result == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    result = THSTensor_full_like(handle, value.Handle, (sbyte)dtype, (int)device.type, device.index, requiresGrad);
                }
                if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(result);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_eye_out(long rows, long columns, IntPtr tensorOut);

            /// <summary>
            ///  Mutates the tensor into a 2-D tensor with ones on the diagonal and zeros elsewhere.
            /// </summary>
            public Tensor eye(long rows, long columns)
            {
                var res = THSTensor_eye_out(rows, columns, handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_scatter(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

            /// <summary>
            ///  Writes all values from the tensor src into self at the indices specified in the index tensor. For each
            ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
            ///  corresponding value in index for dimension = dim.
            /// </summary>
            public Tensor scatter(long dimension, Tensor index, Tensor src)
            {
                var res = THSTensor_scatter(handle, dimension, index.Handle, src.Handle);
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
                var res = THSTensor_gather(handle, dimension, index.Handle);
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
                        var res = THSTensor_flip(handle, (IntPtr)psizes, sizes.Length);
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
                var res = THSTensor_fliplr(handle);
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
                var res = THSTensor_flipud(handle);
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
                var res = THSTensor_nanmedian(handle);
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
                var res = THSTensor_nansum(handle);
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
                            THSTensor_nan_to_num(handle, (IntPtr)pnan, (IntPtr)pposinf, (IntPtr)pneginf);
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
                var res = THSTensor_nextafter(handle, other.handle);
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
                var res = THSTensor_narrow(handle, dimension, start, length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_nonzero(IntPtr tensor);

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public Tensor nonzero()
            {
                var res = THSTensor_nonzero(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public IList<Tensor> nonzero_as_list()
            {
                var res = THSTensor_nonzero(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }

                var t = new Tensor(res);
                return t.chunk(t.shape[1], dim: 1);
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
                var res = THSTensor_slice(handle, dimension, start, finish, step);
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
                var res = THSTensor_unsqueeze(handle, dim);
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
                var res = THSTensor_unsqueeze_(handle, dim);
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

            // Specifically added to make F# look good.
            public static Tensor op_MinusMinusGreater(Tensor t, torch.nn.Module m) => m.forward(t);

            /// <summary>
            ///   Get a string representation of the tensor.
            /// </summary>
            public override string ToString()
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

        public string ToString(bool withData, string fltFormat = "g5", int width = 100)
        {
            if (!withData) return this.ToString();

            var builder = new StringBuilder(this.ToString());

            if (Dimensions == 0) {

                builder.Append(", value = ");
                PrintValue(builder, dtype, this.ToScalar(), fltFormat);

            } else if (Dimensions == 1) {

                var row = new List<string>();
                BuildRow(row, this, width, fltFormat);

                var appendEllipsis = row.Count < shape[0];

                builder.AppendLine();
                PrintOneRow(row, row.Select(str => str.Length).ToArray(), new bool[shape[0]], fltFormat, builder, this, appendEllipsis);

            } else if (Dimensions == 2) {

                builder.AppendLine().AppendLine();
                PrintTwoDimensions(fltFormat, width, builder, this);

            } else {
                builder.AppendLine();
                var indices = new List<TensorIndex>();
                RecursivePrintDimensions(0, indices, fltFormat, width, builder);
            }

            return builder.ToString();
        }

        private void RecursivePrintDimensions(int dim, IEnumerable<TensorIndex> indices, string fltFormat, int width, StringBuilder builder)
        {
            if (dim == Dimensions-3) {
                // We're at the third-last dimension. This is where we can print out the last two dimensions.

                for (int i = 0; i < shape[dim]; i++) {

                    var idxs = indices.Append(TensorIndex.Single(i)).Append(TensorIndex.Ellipsis).Append(TensorIndex.Ellipsis).ToArray();
                    var str = IndicesToString(idxs);
                    builder.AppendLine().AppendLine($"{str} =");
                    var slice = this.index(idxs);
                    PrintTwoDimensions(fltFormat, width, builder, slice);
                }
            }
            else {

                for (int i = 0; i < shape[dim]; i++) {

                    RecursivePrintDimensions(dim+1, indices.Append(TensorIndex.Single(i)), fltFormat, width, builder);
                }
            }
        }

        private string IndicesToString(IList<TensorIndex> indices)
        {
            var builder = new StringBuilder("[");
            for (int i = 0; i < indices.Count(); i++) {

                if (i > 0) builder.Append(',');

                if (indices[i].kind == TensorIndex.Kind.Ellipsis) {
                    builder.Append(':');
                }
                else if (indices[i].kind == TensorIndex.Kind.Single) {
                    builder.Append(indices[i].startIndexOrBoolOrSingle);
                }
            }
            return builder.Append(']').ToString();
        }

        private static void PrintTwoDimensions(string fltFormat, int width, StringBuilder builder, Tensor t)
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
                BuildRow(row, t[i], width, fltFormat);
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
                PrintOneRow(rows[i].Take(shortestRow).ToList(), columnSpace, hasMinus, fltFormat, builder, t[i], appendEllipsis);
            }
        }

        private const string ellipsis = "...";

        private static void PrintOneRow(IList<string> row, int[] space, bool[] hasMinus, string fltFormat, StringBuilder builder, Tensor rowTensor, bool appendEllipsis)
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
            builder.AppendLine();
        }

        private static void BuildRow(List<string> row, Tensor t, int width, string fltFormat)
        {
            var type = t.dtype;
            var endingWidth = ellipsis.Length+1;
            
            for (int i = 0; i < t.shape[0]; i++) {

                var builder = new StringBuilder();
                PrintValue(builder, type, t[i].ToScalar(), fltFormat);

                var str = builder.ToString();

                if (width - str.Length - endingWidth < 0) {
                    break;
                }

                row.Add(str);
                width -= str.Length+1;
            }
        }

        private static void PrintValue(StringBuilder builder, ScalarType type, Scalar value, string fltFormat)
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
                builder.Append(value.ToSingle().ToString(fltFormat));
                break;
            case ScalarType.Float32:
                builder.Append(value.ToSingle().ToString(fltFormat));
                break;
            case ScalarType.Float64:
                builder.Append(value.ToDouble().ToString(fltFormat));
                break;
            case ScalarType.ComplexFloat32:
                var val1 = value.ToComplexFloat32();
                if (val1.Real != 0.0f || val1.Imaginary == 0.0f)
                    builder.Append(val1.Real.ToString(fltFormat));
                if (val1.Real != 0.0f || val1.Imaginary != 0.0f)
                    builder.Append('+');
                if (val1.Imaginary != 0.0f)
                    builder.Append(val1.Imaginary.ToString(fltFormat)).Append('i');
                break;
            case ScalarType.ComplexFloat64:
                var val2 = value.ToComplexFloat64();
                if (val2.Real != 0.0f || val2.Imaginary == 0.0f)
                    builder.Append(val2.Real.ToString(fltFormat)).Append('+');
                if (val2.Real != 0.0f || val2.Imaginary != 0.0f)
                    builder.Append('+');
                if (val2.Imaginary != 0.0f)
                    builder.Append(val2.Imaginary.ToString(fltFormat)).Append('i');
                break;
            }
        }

        public static explicit operator float (Tensor value) => value.ToSingle();
        public static explicit operator double (Tensor value) => value.ToDouble();
        public static explicit operator sbyte (Tensor value) => value.ToSByte();
        public static explicit operator byte (Tensor value) => value.ToByte();
        public static explicit operator short (Tensor value) => value.ToInt16();
        public static explicit operator int (Tensor value) => value.ToInt32();
        public static explicit operator long (Tensor value) => value.ToInt64();
        public static explicit operator bool (Tensor value) => value.ToBoolean();

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
                var res = THSTensor_atleast_1d(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_atleast_2d(IntPtr tensor);

            public Tensor atleast_2d()
            {
                var res = THSTensor_atleast_2d(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSTensor_atleast_3d(IntPtr tensor);

            public Tensor atleast_3d()
            {
                var res = THSTensor_atleast_3d(handle);
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

            public static implicit operator TensorIndex(System.Range value)
            {
                return TensorIndex.Slice(value.Start.Value, value.End.Value);
            }
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
    }
}