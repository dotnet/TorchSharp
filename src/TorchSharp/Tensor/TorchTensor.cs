// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable
namespace TorchSharp.Tensor
{
    /// <summary>
    /// Represents a TorchSharp tensor.
    /// </summary>
    public sealed partial class TorchTensor : IDisposable
    {
        internal IntPtr handle;

        internal TorchTensor(IntPtr handle)
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
            return (obj is TorchTensor) && this.Equals((obj as TorchTensor)!);

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
        ~TorchTensor() => Dispose(false);

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

        public TorchTensor Real { get {
                var res = THSTensor_real(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);

            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_imag(IntPtr handle);

        public TorchTensor Imag {
            get {
                var res = THSTensor_imag(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
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
        public TorchScalar ToScalar()
        {
            var res = THSTensor_item(Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchScalar(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

        /// <summary>
        /// Fill the tensor with the provided scalar value.
        /// </summary>
        /// <param name="value">A scalar value</param>
        /// <returns></returns>
        public TorchTensor fill_(TorchScalar value)
        {
            var res = THSTensor_fill_(handle, value.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern sbyte THSTensor_type(IntPtr handle);

        /// <summary>
        /// Gets the type of the tensor elements.
        /// </summary>
        public ScalarType Type => (ScalarType)THSTensor_type(handle);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        static extern string THSTensor_device_str(IntPtr handle);

        /// <summary>
        /// Gets a string representing the device where the tensor is stored.
        /// </summary>
        public torch.device device
        {
            get
            {
                var dev_type = device_type;
                if (dev_type == DeviceType.CPU) {
                    return new torch.device(DeviceType.CPU);
                }
                else {
                    return new torch.device(dev_type, device_index);
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
        public bool IsSparse
        {
            get
            {
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
        public static TorchTensor load(string location)
        {
            var res = THSTensor_load(location);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
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
        public TorchTensor with_requires_grad()
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
        public TorchTensor cpu()
        {
            var res = THSTensor_cpu(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cuda(IntPtr handle);

        /// <summary>
        /// Returns a copy of this object in CUDA memory.
        /// If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.
        /// </summary>
        /// <returns></returns>
        public TorchTensor cuda()
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
            var res = THSTensor_cuda(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
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
        public TorchTensor to_type(ScalarType type)
        {
            var res = THSTensor_to_type(handle, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Moves the tensor data to a specific device.
        /// </summary>
        /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
        /// <param name="deviceIndex">The optional device index.</param>
        /// <returns></returns>
        public TorchTensor to(DeviceType deviceType, int deviceIndex = -1)
        {
            torch.InitializeDeviceType(deviceType);
            var res = THSTensor_to_device(handle, (int)deviceType, deviceIndex);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Moves the tensor data and casts it to the given element type.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public TorchTensor to(ScalarType type, torch.device device)
        {
            torch.InitializeDevice(device);
            var res = THSTensor_to_type_and_device(handle, (sbyte)type, (int)device.Type, device.Index);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            //var res = THSTensor_to_type(handle, (sbyte)type);
            //if (res == IntPtr.Zero)
            //    Torch.CheckForErrors();

            //res = THSTensor_to_device(res, (int)device.Type, device.Index);
            //if (res == IntPtr.Zero)
            //    Torch.CheckForErrors();

            return new TorchTensor(res);
        }

        /// <summary>
        /// Cast the tensor to the given element type.
        /// </summary>
        /// <remarks>Alias for to_type</remarks>
        public TorchTensor to(ScalarType type) => to_type(type);

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="device">A string denoting the target device.</param>
        /// <returns></returns>
        public TorchTensor to(string device) => to(new torch.device(device));

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="device">The target device</param>
        /// <returns></returns>
        public TorchTensor to(torch.device device) => to(device.Type, device.Index);

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="other">The tensor serving as a template.</param>
        /// <returns></returns>
        public TorchTensor to(TorchTensor other) => to(other.device_type, other.device_index);

        
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
        public long[] shape
        {
            get
            {
                return size();
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_indices(IntPtr handle);

        /// <summary>
        /// Return the indices tensor of a sparse COO tensor.
        /// </summary>
        public TorchTensor SparseIndices {
            get {
                var res = THSTensor_indices(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_values(IntPtr handle);

        /// <summary>
        /// Return the values tensor of a sparse COO tensor.
        /// </summary>
        public TorchTensor SparseValues {
            get {
                var res = THSTensor_values(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_vander(IntPtr handle, long N, bool increasing);

        /// <summary>
        /// 
        /// </summary>
        public TorchTensor vander (long N = -1, bool increasing = false)
        {
            if (this.Dimensions != 1) throw new InvalidOperationException("Input argument for 'vander()' must be 1-D.");

            var res = THSTensor_vander(handle, (N == -1) ? this.size(0) : N, increasing);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
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
        public TorchTensor as_strided (long[] size, long[] strides, long storageOffset = 0L)
        {
            unsafe {
                fixed (long* psizes = size, pstrides = strides) {
                    var result = THSTensor_as_strided(handle, (IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, storageOffset);
                    if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(result);
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
        public TorchTensor to_dense()
        {
            var res = THSTensor_to_dense(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_clone(IntPtr handle);

        /// <summary>
        /// Returns a copy of the tensor input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor clone()
        {
            var res = THSTensor_clone(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, bool non_blocking);

        /// <summary>
        /// Copies the elements from src into self tensor and returns self.
        /// </summary>
        /// <returns></returns>
        /// <remarks>The src tensor must be broadcastable with the target 'this' tensor. It may be of a different data type or reside on a different device.</remarks>
        public TorchTensor copy_(TorchTensor source, bool nonBlocking = false)
        {
            var res = THSTensor_copy_(handle, source.Handle, nonBlocking);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_contiguous(IntPtr handle);

        /// <summary>
        /// Returns a contiguous in memory tensor containing the same data as the input tensor.
        /// If tensor is already in the specified memory format, this function returns the original tensor.
        /// </summary>
        /// <returns></returns>
        public TorchTensor contiguous()
        {
            var res = THSTensor_contiguous(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_grad(IntPtr handle);

        /// <summary>
        /// This attribute is None by default and becomes a Tensor the first time a call to backward() computes gradients for the tensor.
        /// The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.
        /// </summary>
        /// <returns></returns>
        public TorchTensor grad()
        {
            var res = THSTensor_grad(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_index(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_index_put_scalar_(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength, IntPtr value);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_index_put_(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength, IntPtr value);
        internal void EncodeIndices(TorchTensorIndex[] indices,
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
                if (idx.kind == TorchTensorIndex.Kind.Slice && idx.stopIndex.HasValue)
                    hasSliceEnd = true;
                if (idx.kind == TorchTensorIndex.Kind.Slice && idx.step.HasValue)
                    hasSliceStep = true;
                if (idx.kind == TorchTensorIndex.Kind.Tensor && (object?)idx.tensor != null)
                    hasTensor = true;
            }
            arrStops = hasSliceEnd ? new long[n] : null;
            arrSteps = hasSliceStep ? new long[n] : null;
            arrTensors = hasTensor ? new IntPtr[n] : null;
            arrKindAndStarts = new long[n];
            for (int i = 0; i < indices.Length; i++) {
                var idx = indices[i];
                arrKindAndStarts[i] =
                    (idx.kind == TorchTensorIndex.Kind.Null) ? long.MinValue:
                    (idx.kind == TorchTensorIndex.Kind.Bool && idx.startIndexOrBoolOrSingle == 0) ? long.MinValue+1 :
                    (idx.kind == TorchTensorIndex.Kind.Bool && idx.startIndexOrBoolOrSingle != 0) ? long.MinValue+2 :
                    (idx.kind == TorchTensorIndex.Kind.Ellipsis) ? long.MinValue+3 :
                    (idx.kind == TorchTensorIndex.Kind.None) ? long.MinValue+4 :
                    (idx.kind == TorchTensorIndex.Kind.Tensor) ? long.MinValue+5 :
                    (idx.kind == TorchTensorIndex.Kind.Slice && !idx.startIndexOrBoolOrSingle.HasValue) ? long.MinValue+6 :
                    (idx.kind == TorchTensorIndex.Kind.Single) ? idx.startIndexOrBoolOrSingle.GetValueOrDefault() :
                    idx.startIndexOrBoolOrSingle.GetValueOrDefault() + long.MinValue/2;
                if (arrStops != null && idx.kind == TorchTensorIndex.Kind.Slice)
                    arrStops[i] = (idx.stopIndex.HasValue ? idx.stopIndex.Value : long.MinValue);
                if (arrSteps != null && idx.kind == TorchTensorIndex.Kind.Slice)
                    arrSteps[i] = (idx.step.HasValue ? idx.step.Value : long.MinValue);
                if (arrTensors != null && idx.kind == TorchTensorIndex.Kind.Tensor)
                    arrTensors[i] = ((object?)idx.tensor == null ? IntPtr.Zero : idx.tensor.Handle);
            }

        }

        /// <summary>
        /// Index into the tensor using Python-like indexing expressions.
        /// </summary>
        [IndexerName("TensorItems")]
        public TorchTensor this[params TorchTensorIndex[] indices] {
            get { return index(indices); }
            set { index_put_(value, indices);  }
        }

        [IndexerName("TensorItems")]
        public TorchTensor this[params TorchTensor[] indices] {
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
        public TorchTensor this[long i1] {
            get {
                var res = THSTensor_get1(handle, i1);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
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
        public TorchTensor this[long i1, long i2] {
            get {
                var res = THSTensor_get2(handle, i1, i2);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
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
        public TorchTensor this[long i1, long i2, long i3] {
            get {
                var res = THSTensor_get3(handle, i1, i2, i3);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new TorchTensor(res);
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
        public TorchTensor this[long i1, long i2, long i3, long i4] {
            get {
                var res = THSTensor_get4(handle, i1, i2, i3, i4);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new TorchTensor(res);
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
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5] {
            get {
                var res = THSTensor_get5(handle, i1, i2, i3, i4, i5);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new TorchTensor(res);
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
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5, long i6] {
            get {
                var res = THSTensor_get6(handle, i1, i2, i3, i4, i5, i6);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new TorchTensor(res);
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
        public TorchTensor index(params TorchTensorIndex[] indices)
        {
            EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
            unsafe {
                fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                    fixed (IntPtr* ptrTensors = arrTensors) {
                        var res = THSTensor_index(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length);
                        if (res == IntPtr.Zero)
                            torch.CheckForErrors();
                        GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                        return new TorchTensor(res);
                    }
                }
            }

        }

        /// <summary>
        /// Index into the tensor using Python-like indexing expressions.
        /// </summary>
        /// <returns></returns>
        public TorchTensor index(params TorchTensor[] indices)
        {
            return index(indices.Select(t => TorchTensorIndex.Tensor(t)).ToArray());
        }

        /// <summary>
        /// Index into the tensor using Python-like indexing expressions and place a tensor at the index.
        /// </summary>
        /// <returns></returns>
        public TorchTensor index_put_(TorchTensor value, params TorchTensorIndex[] indices)
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
                        return new TorchTensor(res);
                    }
                }
            }
        }

        /// <summary>
        /// Index into the tensor using Python-like indexing expressions and place a tensor at the index.
        /// </summary>
        /// <returns></returns>
        public TorchTensor index_put_(TorchTensor value, params TorchTensor[] indices)
        {
            return index_put_(value, indices.Select(t => TorchTensorIndex.Tensor(t)).ToArray());
        }
        

        /// <summary>
        /// Index into the tensor using Python-like indexing expressions and place a scalar tensor at the index.
        /// </summary>
        /// <returns></returns>
        public TorchTensor index_put_(TorchScalar value, params TorchTensorIndex[] indices)
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
                        return new TorchTensor(res);
                    }
                }
            }
        }

        /// <summary>
        /// Index into the tensor using Python-like indexing expressions and place a scalar tensor at the index.
        /// </summary>
        /// <returns></returns>
        public TorchTensor index_put_(TorchScalar value, params TorchTensor[] indices)
        {
            return index_put_(value, indices.Select(t => TorchTensorIndex.Tensor(t)).ToArray());
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_index_select(IntPtr tensor, long dimension, IntPtr index);

        /// <summary>
        /// Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public TorchTensor index_select(long dimension, TorchTensor index)
        {
            var res = THSTensor_index_select(handle, dimension, index.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_take(IntPtr tensor, IntPtr index);

        /// <summary>
        /// Returns a new tensor with the elements of input at the given indices. The input tensor is treated as if it were viewed as a 1-D tensor.
        /// The result takes the same shape as the indices.
        /// </summary>
        /// <param name="index">The indices into tensor, an Int64 tensor.</param>
        /// <returns></returns>
        public TorchTensor take(TorchTensor index)
        {
            var res = THSTensor_take(handle, index.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
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
        public TorchTensor take_along_dim(TorchTensor indices)
        {
            var res = THSTensor_take_along_dim_dflt(handle, indices.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Selects values from input at the 1-dimensional indices from indices along the given dim.
        /// </summary>
        /// <param name="indices">The indices into input. Must have long dtype.</param>
        /// <returns></returns>
        /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
        public TorchTensor take_along_dim(IEnumerable<long> indices) => take_along_dim(Int64Tensor.from(indices.ToArray()));

        /// <summary>
        /// Selects values from input at the 1-dimensional indices from indices along the given dim.
        /// </summary>
        /// <param name="indices">The indices into input. Must have long dtype.</param>
        /// <param name="dimension">Dimension to select along.</param>
        /// <returns></returns>
        /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
        public TorchTensor take_along_dim(TorchTensor indices, long dimension)
        {
            var res = THSTensor_take_along_dim(handle, indices.Handle, dimension);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Selects values from input at the 1-dimensional indices from indices along the given dim.
        /// </summary>
        /// <param name="indices">The indices into input. Must have long dtype.</param>
        /// <param name="dim">Dimension to select along.</param>
        /// <returns></returns>
        /// <remarks>Functions that return indices along a dimension, like torch.argmax() and torch.argsort(), are designed to work with this function.</remarks>
        public TorchTensor take_along_dim(IEnumerable<long> indices, long dim) => take_along_dim(Int64Tensor.from(indices.ToArray()), dim);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_reshape(IntPtr tensor, IntPtr shape, int length);

        /// <summary>
        /// Returns a tensor with the same data and number of elements as self but with the specified shape.
        /// </summary>
        /// <param name="shape">The new tensor shape.</param>
        /// <returns></returns>
        public TorchTensor reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    var res = THSTensor_reshape(handle, (IntPtr)pshape, shape.Length);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_squeeze(IntPtr tensor, long dimension);

        /// <summary>
        /// Returns a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
        /// </summary>
        /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
        /// <returns></returns>
        public TorchTensor squeeze(long dim)
        {
            var res = THSTensor_squeeze(handle, dim);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_t(IntPtr tensor);

        /// <summary>
        /// Expects input to be 1- or 2-D tensor and transposes dimensions 0 and 1.
        /// </summary>
        /// <returns></returns>
        public TorchTensor t()
        {
            var res = THSTensor_t(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_transpose(IntPtr tensor, long dim1, long dim2);

        /// <summary>
        /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
        /// </summary>
        /// <param name="dim0"></param>
        /// <param name="dim1"></param>
        /// <returns></returns>
        public TorchTensor transpose(long dim0, long dim1)
        {
            var res = THSTensor_transpose(handle, dim0, dim1);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tril(IntPtr tensor, long diagonal);

        /// <summary>
        /// Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
        /// The lower triangular part of the matrix is defined as the elements on and below the diagonal.
        /// </summary>
        /// <param name="diagonal">The diagonal to consider</param>
        /// <returns></returns>
        public TorchTensor tril(long diagonal = 0)
        {
            var res = THSTensor_tril(handle, diagonal);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_triu(IntPtr tensor, long diagonal);

        /// <summary>
        /// Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
        /// The upper triangular part of the matrix is defined as the elements on and above the diagonal.
        /// </summary>
        /// <param name="diagonal">The diagonal to consider</param>
        /// <returns></returns>
        public TorchTensor triu(long diagonal = 0)
        {
            var res = THSTensor_triu(handle, diagonal);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }


        /// <summary>
        /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
        /// </summary>
        public TorchTensor swapdims(long dim0, long dim1) => transpose(dim0, dim1);

        /// <summary>
        /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
        /// </summary>
        public TorchTensor swapaxes(long dim0, long dim1) => transpose(dim0, dim1);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_transpose_(IntPtr tensor, long dim1, long dim2);

        /// <summary>
        /// Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
        /// Inplace version of transpose()
        /// </summary>
        /// <param name="dim0"></param>
        /// <param name="dim1"></param>
        /// <returns></returns>
        public TorchTensor transpose_(long dim0, long dim1)
        {
            return new TorchTensor(THSTensor_transpose_(handle, dim0, dim1));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_view(IntPtr tensor, IntPtr shape, int length);

        /// <summary>
        /// Returns a new tensor with the same data as the input tensor but of a different shape.
        /// </summary>
        /// <param name="shape">The shape of the view</param>
        /// <returns></returns>
        public TorchTensor view(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    var res = THSTensor_view(handle, (IntPtr)pshape, shape.Length);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_view_as_complex(IntPtr tensor);

        /// <summary>
        /// Returns a view of input as a complex tensor. 
        /// </summary>
        public TorchTensor view_as_complex()
        {
            var result = THSTensor_view_as_complex(handle);
            if (result == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_view_as_real(IntPtr tensor);

        /// <summary>
        /// Returns a view of input as a real tensor. 
        /// </summary>
        public TorchTensor view_as_real()
        {
            var result = THSTensor_view_as_real(handle);
            if (result == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_all(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor all()
        {
            var res = THSTensor_all(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_all_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="keepDim"></param>
        /// <returns></returns>
        public TorchTensor all(long dimension, bool keepDim = false)
        {
            var res = THSTensor_all_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_amax(IntPtr tensor, IntPtr dim, int dim_len, bool keep_dim);

        /// <summary>
        /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
        /// </summary>
        /// <param name="dims">The dimension or dimensions to reduce.</param>
        /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
        /// <returns></returns>
        public TorchTensor amax(long[] dims, bool keepDim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSTensor_amax(handle, (IntPtr)pdims, dims.Length, keepDim);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
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
        public TorchTensor amin(long[] dims, bool keepDim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSTensor_amin(handle, (IntPtr)pdims, dims.Length, keepDim);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_any(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor any()
        {
            var res = THSTensor_any(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_any_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="keepDim"></param>
        /// <returns></returns>
        public TorchTensor any(long dimension, bool keepDim = false)
        {
            var res = THSTensor_any_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_argmax(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor argmax()
        {
            var res = THSTensor_argmax(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_argmax_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="keepDim"></param>
        /// <returns></returns>
        public TorchTensor argmax(long dimension, bool keepDim = false)
        {
            var res = THSTensor_argmax_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_argmin(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor argmin()
        {
            var res = THSTensor_argmin(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_argmin_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="keepDim"></param>
        /// <returns></returns>
        public TorchTensor argmin(long dimension, bool keepDim = false)
        {
            var res = THSTensor_argmin_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_argsort(IntPtr tensor, long dimension, bool descending);

        /// <summary>
        /// Returns the indices that sort a tensor along a given dimension in ascending order by value.
        /// </summary>
        /// <param name="dimension">The dimension to sort along</param>
        /// <param name="descending">Controls the sorting order (ascending or descending)</param>
        /// <returns></returns>
        public TorchTensor argsort(long dimension = -1, bool descending = false)
        {
            var res = THSTensor_argsort(handle, dimension, descending);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_deg2rad(IntPtr tensor);

        /// <summary>
        /// Convert each element from degrees to radians.
        /// </summary>
        /// <returns></returns>
        public TorchTensor deg2rad()
        {
            var res = THSTensor_deg2rad(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rad2deg(IntPtr tensor);

        /// <summary>
        /// Convert each element from radians to degrees.
        /// </summary>
        /// <returns></returns>
        public TorchTensor rad2deg()
        {
            var res = THSTensor_rad2deg(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_copysign(IntPtr tensor, IntPtr other);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor copysign(TorchTensor other)
        {
            var res = THSTensor_copysign(handle, other.handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_count_nonzero(IntPtr tensor, IntPtr dim, int dim_len);

        public TorchTensor count_nonzero(long[]? dims = null)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSTensor_count_nonzero(handle, (IntPtr)pdims, dims is null ? 0 : dims.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
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
        public TorchTensor tile(long[] reps)
        {
            unsafe {
                fixed (long* pdims = reps) {
                    var res = THSTensor_tile(handle, (IntPtr)pdims, reps.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_digamma(IntPtr tensor);

        /// <summary>
        /// Computes the logarithmic derivative of the gamma function on input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor digamma()
        {
            var res = THSTensor_digamma(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_digamma_(IntPtr tensor);

        /// <summary>
        /// Computes the logarithmic derivative of the gamma function on input, in place.
        /// </summary>
        /// <returns></returns>
        public TorchTensor digamma_()
        {
            var res = THSTensor_digamma_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lgamma(IntPtr tensor);

        /// <summary>
        /// Computes the logarithm of the gamma function on input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor lgamma()
        {
            var res = THSTensor_lgamma(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lgamma_(IntPtr tensor);

        /// <summary>
        /// Computes the logarithm of the gamma function on input, in place.
        /// </summary>
        /// <returns></returns>
        public TorchTensor lgamma_()
        {
            var res = THSTensor_lgamma_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mvlgamma(IntPtr tensor, long p);

        /// <summary>
        /// Computes the multivariate log-gamma function) with dimension pp element-wise
        /// </summary>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public TorchTensor mvlgamma(long p)
        {
            var res = THSTensor_mvlgamma(handle, p);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mvlgamma_(IntPtr tensor, long p);

        /// <summary>
        /// Computes the multivariate log-gamma function) with dimension pp element-wise, in place.
        /// </summary>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public TorchTensor mvlgamma_(long p)
        {
            var res = THSTensor_mvlgamma_(handle, p);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_complex(IntPtr real, IntPtr imag);

        /// <summary>
        /// onstructs a complex tensor with its real part equal to real and its imaginary part equal to imag.
        /// </summary>
        static public TorchTensor complex(TorchTensor real, TorchTensor imag)
        {
            var res = THSTensor_complex(real.Handle, imag.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_polar(IntPtr abs, IntPtr angle);

        /// <summary>
        /// Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value 'abs' and angle 'angle'.
        /// </summary>
        static public TorchTensor polar(TorchTensor abs, TorchTensor angle)
        {
            var res = THSTensor_polar(abs.Handle, angle.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_polygamma(IntPtr tensor, long p);

        public TorchTensor polygamma(long p)
        {
            var res = THSTensor_polygamma(handle, p);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_polygamma_(IntPtr tensor, long p);

        public TorchTensor polygamma_(long p)
        {
            var res = THSTensor_polygamma_(handle, p);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_positive(IntPtr tensor);

        /// <summary>
        /// Returns input. Throws a runtime error if input is a bool tensor. 
        /// </summary>
        /// <returns></returns>
        public TorchTensor positive()
        {
            if (this.Type == ScalarType.Bool) throw new ArgumentException("Boolean tensor");
            var res = THSTensor_positive(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_softplus(IntPtr tensor);

        public TorchTensor softplus()
        {
            var res = THSTensor_softplus(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ravel(IntPtr tensor);

        public TorchTensor ravel()
        {
            var res = THSTensor_ravel(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu(IntPtr tensor);

        public TorchTensor relu()
        {
            var res = THSTensor_relu(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu_(IntPtr tensor);

        public TorchTensor relu_()
        {
            var res = THSTensor_relu_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu6(IntPtr tensor);

        public TorchTensor relu6()
        {
            var res = THSTensor_relu6(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu6_(IntPtr tensor);

        public TorchTensor relu6_()
        {
            var res = THSTensor_relu6_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_celu(IntPtr tensor);

        public TorchTensor celu()
        {
            var res = THSTensor_celu(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_celu_(IntPtr tensor);

        public TorchTensor celu_()
        {
            var res = THSTensor_celu_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_elu(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        public TorchTensor elu(TorchScalar alpha, TorchScalar scale, TorchScalar input_scale)
        {
            var res = THSTensor_elu(handle, alpha.Handle, scale.Handle, input_scale.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_elu_(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        public TorchTensor elu_(TorchScalar alpha, TorchScalar scale, TorchScalar input_scale)
        {
            var res = THSTensor_elu_(handle, alpha.Handle, scale.Handle, input_scale.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gelu(IntPtr tensor);

        public TorchTensor gelu()
        {
            var res = THSTensor_gelu(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardsigmoid(IntPtr tensor);

        public TorchTensor hardsigmoid()
        {
            var res = THSTensor_hardsigmoid(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardsigmoid_(IntPtr tensor);

        public TorchTensor hardsigmoid_()
        {
            var res = THSTensor_hardsigmoid_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardswish(IntPtr tensor);

        public TorchTensor hardswish()
        {
            var res = THSTensor_hardswish(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardswish_(IntPtr tensor);

        public TorchTensor hardswish_()
        {
            var res = THSTensor_hardswish_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardtanh(IntPtr tensor, IntPtr min, IntPtr max);

        public TorchTensor hardtanh(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_hardtanh(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardtanh_(IntPtr tensor, IntPtr min, IntPtr max);

        public TorchTensor hardtanh_(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_hardtanh_(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_heaviside(IntPtr tensor, IntPtr other);

        public TorchTensor heaviside(TorchTensor other)
        {
            var res = THSTensor_heaviside(handle, other.handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_igamma(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Computes the regularized lower incomplete gamma function
        /// </summary>
        /// <param name="other">The second non-negative input tensor</param>
        /// <returns></returns>
        public TorchTensor igamma(TorchTensor other)
        {
            var res = THSTensor_igamma(handle, other.handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_igammac(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Computes the regularized upper incomplete gamma function.
        /// </summary>
        /// <param name="other">The second non-negative input tensor</param>
        /// <returns></returns>
        public TorchTensor igammac(TorchTensor other)
        {
            var res = THSTensor_igammac(handle, other.handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_i0(IntPtr tensor);

        /// <summary>
        /// Computes the zeroth order modified Bessel function of the first kind for each element of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor i0()
        {
            var res = THSTensor_i0(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
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
        public TorchTensor isclose(TorchTensor other, double rtol = 1e-05, double atol = 1e-08, bool nanEqual = false)
        {
            var res = THSTensor_isclose(handle, other.Handle, rtol, atol, nanEqual);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_isinf(IntPtr tensor);

        public TorchTensor isinf()
        {
            var res = THSTensor_isinf(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_isfinite(IntPtr tensor);

        public TorchTensor isfinite()
        {
            var res = THSTensor_isfinite(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_isposinf(IntPtr tensor);

        public TorchTensor isposinf()
        {
            var res = THSTensor_isposinf(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_isneginf(IntPtr tensor);

        public TorchTensor isneginf()
        {
            var res = THSTensor_isneginf(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_isreal(IntPtr tensor);

        public TorchTensor isreal()
        {
            var res = THSTensor_isreal(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_leaky_relu(IntPtr tensor, IntPtr negative_slope);

        public TorchTensor leaky_relu(TorchScalar negative_slope)
        {
            var res = THSTensor_leaky_relu(handle, negative_slope.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_leaky_relu_(IntPtr tensor, IntPtr negative_slope);

        public TorchTensor leaky_relu_(TorchScalar negative_slope)
        {
            var res = THSTensor_leaky_relu_(handle, negative_slope.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_selu(IntPtr tensor);

        public TorchTensor selu()
        {
            var res = THSTensor_selu(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_selu_(IntPtr tensor);

        public TorchTensor selu_()
        {
            var res = THSTensor_selu_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_silu(IntPtr tensor);

        public TorchTensor silu()
        {
            var res = THSTensor_silu(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_silu_(IntPtr tensor);

        public TorchTensor silu_()
        {
            var res = THSTensor_silu_(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log_sigmoid(IntPtr tensor);

        public TorchTensor log_sigmoid()
        {
            var res = THSTensor_log_sigmoid(handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lerp(IntPtr tensor, IntPtr end, IntPtr weight);

        public TorchTensor lerp(TorchTensor end, TorchTensor weight)
        {
            var res = THSTensor_lerp(handle, end.Handle, weight.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lerp_(IntPtr tensor, IntPtr end, IntPtr weight);

        public TorchTensor lerp_(TorchTensor end, TorchTensor weight)
        {
            var res = THSTensor_lerp_(handle, end.Handle, weight.Handle);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
            float alpha);

        public TorchTensor baddbmm(TorchTensor batch2, TorchTensor mat, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_baddbmm(handle, batch2.Handle, mat.Handle, beta, alpha);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        public TorchTensor bmm(TorchTensor batch2)
        {
            var res = THSTensor_bmm(handle, batch2.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
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
        public TorchTensor bucketize(TorchTensor boundaries, bool outInt32 = false, bool right = false)
        {
            var res = THSTensor_bucketize(handle, boundaries.Handle, outInt32, right );
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bincount(IntPtr tensor, IntPtr weights, long minlength);

        /// <summary>
        /// Count the frequency of each value in an array of non-negative ints.
        /// </summary>
        public TorchTensor bincount(TorchTensor? weights, long minlength = 0)
        {
            var weightsHandle = (weights is null ? IntPtr.Zero : weights.Handle);
            var res = THSTensor_bincount(handle, weightsHandle, minlength);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        public TorchTensor @bool() => this.to_type(ScalarType.Bool);

        public TorchTensor @byte() => this.to_type(ScalarType.Byte);

        public TorchTensor @char() => this.to_type(ScalarType.Int8);

        public TorchTensor @int() => this.to_type(ScalarType.Int32);

        public TorchTensor @long() => this.to_type(ScalarType.Int64);

        public TorchTensor @float() => this.to_type(ScalarType.Float32);

        public TorchTensor @double() => this.to_type(ScalarType.Float64);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor clamp(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor clip(TorchScalar min, TorchScalar max) => clamp(min, max);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor clamp_(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp_(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

        public TorchTensor clamp_max(TorchScalar max)
        {
            var res = THSTensor_clamp_max(handle, max.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

        public TorchTensor clamp_max_(TorchScalar max)
        {
            var res = THSTensor_clamp_max_(handle, max.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

        public TorchTensor clamp_min(TorchScalar min)
        {
            var res = THSTensor_clamp_min(handle, min.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

        public TorchTensor clamp_min_(TorchScalar min)
        {
            var res = THSTensor_clamp_min_(handle, min.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_cummax(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public (TorchTensor values, TorchTensor indexes) cummax(long dimension)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_cummax(handle, pa.CreateArray, dimension);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_cummin(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public (TorchTensor values, TorchTensor indexes) cummin(long dimension)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_cummin(handle, pa.CreateArray, dimension);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cumsum(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

        public TorchTensor cumsum(long dimension, ScalarType? type = null)
        {
            var res = THSTensor_cumsum(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cumprod(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

        public TorchTensor cumprod(long dimension, ScalarType? type = null)
        {
            var res = THSTensor_cumprod(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_diff(IntPtr tensor, long n, long dim, IntPtr prepend, IntPtr append);

        public TorchTensor diff(long n=1, long dim=-1, TorchTensor? prepend = null, TorchTensor? append = null)
        {
            if (n != 1) throw new NotImplementedException("Tensor.diff with n != 1");
            var res = THSTensor_diff(handle, n, dim, (prepend is TorchTensor) ? (IntPtr)prepend.handle : IntPtr.Zero, (append is TorchTensor) ? (IntPtr)append.handle : IntPtr.Zero);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_diag(IntPtr tensor, long dimension);

        public TorchTensor diag(long dimension = 0)
        {
            var res = THSTensor_diag(handle, dimension);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_diagflat(IntPtr tensor, long offset);

        public TorchTensor diagflat(long offset = 0)
        {
            var res = THSTensor_diagflat(handle, offset);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_diagonal(IntPtr tensor, long offset, long dim1, long dim2);

        public TorchTensor diagonal(long offset = 0, long dim1 = 0, long dim2 = 0)
        {
            var res = THSTensor_diagonal(handle, offset, dim1, dim2);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erf(IntPtr tensor);

        public TorchTensor erf()
        {
            var res = THSTensor_erf(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erf_(IntPtr tensor);

        public TorchTensor erf_()
        {
            var res = THSTensor_erf_(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfc(IntPtr tensor);

        public TorchTensor erfc()
        {
            var res = THSTensor_erfc(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfc_(IntPtr tensor);

        public TorchTensor erfc_()
        {
            var res = THSTensor_erfc_(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfinv(IntPtr tensor);

        public TorchTensor erfinv()
        {
            var res = THSTensor_erfinv(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfinv_(IntPtr tensor);

        public TorchTensor erfinv_()
        {
            var res = THSTensor_erfinv_(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq(IntPtr tensor, IntPtr trg);

        public TorchTensor eq(TorchTensor target)
        {
            var res = THSTensor_eq(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor equal(TorchTensor target) => eq(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq_(IntPtr tensor, IntPtr trg);

        public TorchTensor eq_(TorchTensor target)
        {
            var res = THSTensor_eq_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor eq(TorchScalar target)
        {
            var res = THSTensor_eq_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor eq_(TorchScalar target)
        {
            var res = THSTensor_eq_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern bool THSTensor_equal(IntPtr tensor, IntPtr trg);

        public bool Equals(TorchTensor target)
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
        public bool allclose(TorchTensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
        {
            var res = THSTensor_allclose(handle, target.Handle, rtol, atol, equal_nan);
            torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge(IntPtr tensor, IntPtr trg);

        public TorchTensor ge(TorchTensor target)
        {
            var res = THSTensor_ge(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor greater_equal(TorchTensor target) => ge(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge_(IntPtr tensor, IntPtr trg);

        public TorchTensor ge_(TorchTensor target)
        {
            var res = THSTensor_ge_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor ge(TorchScalar target)
        {
            var res = THSTensor_ge_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor ge_(TorchScalar target)
        {
            var res = THSTensor_ge_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt(IntPtr tensor, IntPtr trg);

        public TorchTensor gt(TorchTensor target)
        {
            var res = THSTensor_gt(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor greater(TorchTensor target) => gt(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt_(IntPtr tensor, IntPtr trg);

        public TorchTensor gt_(TorchTensor target)
        {
            var res = THSTensor_gt_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor gt(TorchScalar target)
        {
            var res = THSTensor_gt_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor gt_(TorchScalar target)
        {
            var res = THSTensor_gt_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_kron(IntPtr tensor, IntPtr other);

        public TorchTensor kron(TorchTensor other)
        {
            var res = THSTensor_kron(handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lcm(IntPtr tensor, IntPtr other);

        public TorchTensor lcm(TorchTensor other)
        {
            var res = THSTensor_lcm(handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lcm_(IntPtr tensor, IntPtr other);

        public TorchTensor lcm_(TorchTensor other)
        {
            var res = THSTensor_lcm_(handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ldexp(IntPtr right, IntPtr left);

        public TorchTensor ldexp(TorchTensor other)
        {
            var res = THSTensor_ldexp(handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le(IntPtr tensor, IntPtr trg);

        public TorchTensor le(TorchTensor target)
        {
            var res = THSTensor_le(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor less_equal(TorchTensor target) => le(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le_(IntPtr tensor, IntPtr trg);

        public TorchTensor le_(TorchTensor target)
        {
            var res = THSTensor_le_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor le(TorchScalar target)
        {
            var res = THSTensor_le_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor le_(TorchScalar target)
        {
            var res = THSTensor_le_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt(IntPtr tensor, IntPtr trg);

        public TorchTensor lt(TorchTensor target)
        {
            var res = THSTensor_lt(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor less(TorchTensor target) => lt(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt_(IntPtr tensor, IntPtr trg);

        public TorchTensor lt_(TorchTensor target)
        {
            var res = THSTensor_lt_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor lt(TorchScalar target)
        {
            var res = THSTensor_lt_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor lt_(TorchScalar target)
        {
            var res = THSTensor_lt_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_masked_fill(IntPtr tensor, IntPtr mask, IntPtr value);

        public TorchTensor masked_fill(TorchTensor mask, TorchScalar value)
        {
            var res = THSTensor_masked_fill(handle, mask.Handle, value.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_masked_scatter(IntPtr tensor, IntPtr mask, IntPtr value);

        public TorchTensor masked_scatter(TorchTensor mask, TorchTensor value)
        {
            var res = THSTensor_masked_scatter(handle, mask.Handle, value.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_masked_scatter_(IntPtr tensor, IntPtr mask, IntPtr value);

        public TorchTensor masked_scatter_(TorchTensor mask, TorchTensor value)
        {
            var res = THSTensor_masked_scatter_(handle, mask.Handle, value.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_masked_select(IntPtr tensor, IntPtr mask);

        public TorchTensor masked_select(TorchTensor mask)
        {
            if (mask.Type != ScalarType.Bool) throw new ArgumentException("The mask tensor must be Boolean.");
            var res = THSTensor_masked_select(handle, mask.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_topk(IntPtr tensor, AllocatePinnedArray allocator, int k,
            long dimension, bool largest, bool sorted);

        public (TorchTensor values, TorchTensor indexes) topk(int k, int dimension = -1, bool largest = true, bool sorted = true)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_topk(handle, pa.CreateArray, k, dimension, largest, sorted);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }


        [DllImport("LibTorchSharp")]
        static extern void THSTensor_unbind(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        /// <summary>
        /// Removes a tensor dimension.
        /// </summary>
        /// <param name="dimension">The dimension to remove.</param>
        /// <returns>An array of all slices along a given dimension, already without it.</returns>
        public TorchTensor[] unbind(int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_unbind(handle, pa.CreateArray, dimension);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

        /// <summary>
        /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
        /// </summary>
        /// <param name="size">The size of a single chunk</param>
        /// <param name="dimension">The dimension along which to split the tensor.</param>
        /// <returns></returns>
        public TorchTensor[] split(long size, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_split_with_size(handle, pa.CreateArray, size, dimension);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

        /// <summary>
        /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
        /// </summary>
        /// <param name="sizes">A list of sizes for each chunk</param>
        /// <param name="dimension">The dimension along which to split the tensor.</param>
        /// <returns></returns>
        public TorchTensor[] split(long[] sizes, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                unsafe
                {
                    fixed (long* psizes = sizes)
                    {
                        THSTensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                        torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_tensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

        public TorchTensor[] tensor_split(long size, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_tensor_split_with_size(handle, pa.CreateArray, size, dimension);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_tensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

        public TorchTensor[] tensor_split(long[] sizes, int dimension = 0)
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

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_tensor_split_with_tensor_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr indices, long dimension);

        public TorchTensor[] tensor_split(TorchTensor indices, int dimension = 0)
        {
            if (indices.Type != ScalarType.Int64) throw new ArgumentException("Tensor indices should be Int64 in 'tensor_split");
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_tensor_split_with_tensor_sizes(handle, pa.CreateArray, indices.Handle, dimension);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_vsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

        /// <summary>
        /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to sizes.
        /// </summary>
        /// <param name="size">The size of each chunk</param>
        /// <returns></returns>
        public TorchTensor[] vsplit(long size)
        {
            if (this.shape[0] % size != 0) throw new ArgumentException("The first dimension must be evenly divisible by the size");
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_vsplit_with_size(handle, pa.CreateArray, size);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_vsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

        /// <summary>
        /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to sizes.
        /// </summary>
        /// <param name="sizes">A list of split points</param>
        /// <returns></returns>
        public TorchTensor[] vsplit(params long[] sizes)
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

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        /// <summary>
        /// Splits input, a tensor with one or more dimensions, into multiple tensors vertically according to indices.
        /// </summary>
        /// <param name="indices">A list of split points</param>
        /// <returns></returns>
        public TorchTensor[] vsplit(TorchTensor indices) => tensor_split(indices, 0);


        [DllImport("LibTorchSharp")]
        static extern void THSTensor_hsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

        /// <summary>
        /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to sizes.
        /// </summary>
        /// <param name="size">The size of each chunk</param>
        /// <returns></returns>
        public TorchTensor[] hsplit(long size)
        {
            if (this.shape[1] % size != 0) throw new ArgumentException("The second dimension must be evenly divisible by the size");
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_hsplit_with_size(handle, pa.CreateArray, size);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_hsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

        /// <summary>
        /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to sizes.
        /// </summary>
        /// <param name="sizes">A list of split points</param>
        /// <returns></returns>
        public TorchTensor[] hsplit(params long[] sizes)
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

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        /// <summary>
        /// Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices.
        /// </summary>
        /// <param name="indices">A list of split points</param>
        /// <returns></returns>
        public TorchTensor[] hsplit(TorchTensor indices) => tensor_split(indices, 1);

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_dsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

        /// <summary>
        /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
        /// </summary>
        /// <param name="size">The size of each chunk</param>
        /// <returns></returns>
        public TorchTensor[] dsplit(long size)
        {
            if (this.shape[2] % size != 0) throw new ArgumentException("The third dimension must be evenly divisible by the size");
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_dsplit_with_size(handle, pa.CreateArray, size);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_dsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

        /// <summary>
        /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
        /// </summary>
        /// <param name="sizes">A list of split points</param>
        public TorchTensor[] dsplit(params long[] sizes)
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

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        /// <summary>
        /// Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
        /// </summary>
        /// <param name="indices">A list of split points</param>
        public TorchTensor[] dsplit(TorchTensor indices) => tensor_split(indices, 2);


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_chunk(IntPtr tensor, AllocatePinnedArray allocator, long chunks, long dim);

        /// <summary>
        /// Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
        /// </summary>
        /// <param name="chunks">The number of chunks to return</param>
        /// <param name="dim">Dimension along which to split the tensor</param>
        /// <returns></returns>
        /// <remarks>The last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.</remarks>
        public TorchTensor[] chunk(long chunks, long dim = 0L)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_chunk(handle, pa.CreateArray, chunks, dim);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max(IntPtr tensor);

        public TorchTensor max()
        {
            var res = THSTensor_max(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_elementwise(IntPtr tensor, IntPtr other);

        public TorchTensor max(TorchTensor other)
        {
            var res = THSTensor_max_elementwise(handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_maximum(IntPtr tensor, IntPtr other);

        public TorchTensor maximum(TorchTensor other)
        {
            var res = THSTensor_maximum(handle, other.handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dimension,
            bool keep_dim);

        public (TorchTensor values, TorchTensor indexes) max(long dimension, bool keepDim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_max_along_dimension(handle, pa.CreateArray, dimension, keepDim);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mean(IntPtr tensor);


        public TorchTensor mean()
        {
            var res = THSTensor_mean(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
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
        TorchTensor quantile(TorchTensor q, long dim = -1, bool keepdim = false)
        {
            var res = THSTensor_quantile(handle, q.handle, dim, keepdim);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_nanquantile(IntPtr tensor, IntPtr q, long dim, bool keepdim);

        /// <summary>
        /// This is a variant of torch.quantile() that “ignores” NaN values, computing the quantiles q as if NaN values in input did not exist.
        /// If all values in a reduced row are NaN then the quantiles for that reduction will be NaN.
        /// </summary>
        /// <seealso cref="TorchTensor.quantile(TorchTensor, long, bool)"/>
        /// <param name="q">1D tensor of quantile values in the range [0, 1]</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
        /// <returns></returns>
        TorchTensor nanquantile(TorchTensor q, long dim = -1, bool keepdim = false)
        {
            var res = THSTensor_nanquantile(handle, q.handle, dim, keepdim);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
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

        public (TorchTensor values, TorchTensor indices) mode(long dim = -1L, bool keepdim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_mode(handle, pa.CreateArray, dim, keepdim);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (values: new TorchTensor(ptrArray[0]), indices: new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        public TorchTensor mean(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe {
                fixed (long* pdims = dimensions) {
                    var res = THSTensor_mean_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_median(IntPtr tensor);

        public TorchTensor median()
        {
            var res = THSTensor_median(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_min(IntPtr tensor);

        public TorchTensor min()
        {
            var res = THSTensor_min(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_minimum(IntPtr tensor, IntPtr other);

        public TorchTensor min(TorchTensor other)
        {
            var res = THSTensor_minimum(handle, other.handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_min_elementwise(IntPtr tensor, IntPtr other);

        public TorchTensor minimum(TorchTensor other)
        {
            var res = THSTensor_min_elementwise(handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_min_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dimension,
            bool keep_dim);

        public (TorchTensor values, TorchTensor indexes) min(long dimension, bool keepDim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_min_along_dimension(handle, pa.CreateArray, dimension, keepDim);
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_msort(IntPtr tensor);

        public TorchTensor msort()
        {
            var res = THSTensor_msort(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sort(IntPtr tensor, long dim, bool descending, bool stable, out IntPtr indices);

        public (TorchTensor Values, TorchTensor Indices) sort(long dim = -1, bool descending = false, bool stable = false)
        {
            var res = THSTensor_sort(handle, dim, descending, stable, out var indices);
            if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
            return (new TorchTensor(res), new TorchTensor(indices));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne(IntPtr tensor, IntPtr trg);

        public TorchTensor ne(TorchTensor target)
        {
            var res = THSTensor_ne(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor not_equal(TorchTensor target) => ne(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne_(IntPtr tensor, IntPtr trg);

        public TorchTensor ne_(TorchTensor target)
        {
            var res = THSTensor_ne_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor not_equal_(TorchTensor target) => ne_(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor ne(TorchScalar target)
        {
            var res = THSTensor_ne_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor ne_(TorchScalar target)
        {
            var res = THSTensor_ne_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_dist(IntPtr tensor, IntPtr other, float p);

        public TorchTensor dist(TorchTensor other, float p = 2.0f)
        {
            var res = THSTensor_dist(handle, other.Handle, p);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_norm(IntPtr tensor, float p);

        public TorchTensor norm(float p = 2.0f)
        {
            var res = THSTensor_norm(handle, p);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_norm_along_dimension(IntPtr tensor, int dimension, bool keepdim, float p);

        public TorchTensor norm(int dimension, bool keepdim = false, float p = 2.0f)
        {
            var res = THSTensor_norm_along_dimension(handle, dimension, keepdim, p);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_outer(IntPtr input, IntPtr vec2);

        public TorchTensor outer(TorchTensor vec2)
        {
            var res = THSTensor_outer(handle, vec2.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_inner(IntPtr input, IntPtr vec2);

        public TorchTensor inner(TorchTensor vec2)
        {
            var res = THSTensor_inner(handle, vec2.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_inverse(IntPtr tensor);

        public TorchTensor inverse()
        {
            var res = THSTensor_inverse(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_prelu(IntPtr tensor, IntPtr trg);

        public TorchTensor prelu(TorchTensor target)
        {
            var res = THSTensor_prelu(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmax(IntPtr tensor, IntPtr trg);

        public TorchTensor fmax(TorchTensor target)
        {
            var res = THSTensor_fmax(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmin(IntPtr tensor, IntPtr trg);

        public TorchTensor fmin(TorchTensor target)
        {
            var res = THSTensor_fmin(handle, target.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_renorm(IntPtr tensor, float p, long dim, float maxnorm);

        public TorchTensor renorm(TorchScalar scalar, float p, long dim, float maxnorm)
        {
            var res = THSTensor_renorm(handle, p, dim, maxnorm);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sigmoid(IntPtr tensor);

        public TorchTensor sigmoid()
        {
            var res = THSTensor_sigmoid(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sigmoid_(IntPtr tensor);

        public TorchTensor sigmoid_()
        {
            var res = THSTensor_sigmoid_(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_std(IntPtr tensor);

        public TorchTensor std()
        {
            var res = THSTensor_std(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_std_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool unbiased, bool keepdim);

        public TorchTensor std(long[] dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe {
                fixed (long* pdims = dimensions) {
                    var res = THSTensor_std_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, unbiased, keepDimension);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sum(IntPtr tensor, bool has_type, sbyte scalar_type);

        /// <summary>
        /// Returns the sum of all elements in the :attr:`input` tensor.
        /// </summary>
        public TorchTensor sum(ScalarType? type = null)
        {
            var res = THSTensor_sum(handle, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sum_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        /// <summary>
        ///  Returns the sum of each row of the input tensor in the given dimensions.
        /// </summary>
        public TorchTensor sum(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe
            {
                fixed (long* pdims = dimensions)
                {
                    var res = THSTensor_sum_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_expand(IntPtr tensor, IntPtr psizes, int length, bool isImplicit);

        /// <summary>
        ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
        /// </summary>
        public TorchTensor expand(long[] sizes, bool isImplicit = false)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    var res = THSTensor_expand(handle, (IntPtr)psizes, sizes.Length, isImplicit);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        /// <summary>
        ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
        /// </summary>
        public TorchTensor expand(params long[] sizes)
        {
            return expand(sizes, false);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_repeat(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        /// Repeats this tensor along the specified dimensions.
        /// </summary>
        public TorchTensor repeat(params long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_repeat(handle, (IntPtr)psizes, sizes.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_broadcast_to(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        /// Broadcasts input to the shape shape. Equivalent to calling input.expand(shape). 
        /// </summary>
        public TorchTensor broadcast_to(params long[] shape)
        {
            unsafe {
                fixed (long* psizes = shape) {
                    var res = THSTensor_broadcast_to(handle, (IntPtr)psizes, shape.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_movedim(IntPtr tensor, IntPtr src, int src_len, IntPtr dst, int dst_len);

        public TorchTensor movedim(long[] source, long[] destination)
        {
            unsafe {
                fixed (long* psource = source, pdest = destination) {
                    var res = THSTensor_movedim(handle, (IntPtr)psource, source.Length, (IntPtr)pdest, destination.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public TorchTensor moveaxis(long[] source, long[] destination) => movedim(source, destination);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public TorchTensor randn_out(params long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_randn_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public TorchTensor rand_out(params long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_rand_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint_out(long high, IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public TorchTensor randint_out(long high, long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_randint_out(high, (IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1) .
        /// </summary>
        public TorchTensor rand_like(ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_rand_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_rand_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1. 
        /// </summary>
        public TorchTensor randn_like(ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_randn_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_randn_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint_like(IntPtr input, long low, long high, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly in the range [low,high).
        /// </summary>
        public TorchTensor randint_like(long low, long high, ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_randint_like(handle, low, high, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_randint_like(handle, low, high, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm_out(long n, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        public TorchTensor randperm_out(long n)
        {
            var res = THSTensor_randperm_out(n, handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bernoulli(IntPtr tensor, double p);

        public TorchTensor bernoulli(double p)
        {
            var res = THSTensor_bernoulli(handle, p);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_multinomial(IntPtr tensor, double num_samples, bool replacement);

        public TorchTensor multinomial(double num_samples, bool replacement = false)
        {
            var res = THSTensor_multinomial(handle, num_samples, replacement);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bernoulli_0(IntPtr tensor, double p, IntPtr gen);

        public TorchTensor bernoulli_(double p = 0.5, torch.Generator? generator = null)
        {
            var res = THSTensor_bernoulli_0(handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bernoulli_1(IntPtr tensor, IntPtr p_tensor, IntPtr gen);

        public TorchTensor bernoulli_(TorchTensor p, torch.Generator? generator = null)
        {
            var res = THSTensor_bernoulli_1(handle, p.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_cauchy_(IntPtr tensor, double median, double sigma, IntPtr gen);

        public TorchTensor cauchy_(double median = 0.0, double sigma = 1.0, torch.Generator? generator = null)
        {
            var res = THSTensor_cauchy_(handle, median, sigma, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_exponential_(IntPtr tensor, double lambda, IntPtr gen);

        public TorchTensor exponential_(double lambda = 1.0, torch.Generator? generator = null)
        {
            var res = THSTensor_exponential_(handle, lambda, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_geometric_(IntPtr tensor, double p, IntPtr gen);

        public TorchTensor geometric_(double p, torch.Generator? generator = null)
        {
            var res = THSTensor_geometric_(handle, p, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

        public TorchTensor normal_(double mean = 0.0, double stddev = 1.0, torch.Generator? generator = null)
        {
            var res = THSTensor_normal_(handle, mean, stddev, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_log_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

        public TorchTensor log_normal_(double mean = 0.0, double stddev = 1.0, torch.Generator? generator = null)
        {
            var res = THSTensor_log_normal_(handle, mean, stddev, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_random_(IntPtr tensor, double low, double high, IntPtr gen);

        public TorchTensor random_(double from, double to, torch.Generator? generator = null)
        {
            var res = THSTensor_random_(handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_uniform_(IntPtr tensor, double low, double high, IntPtr gen);

        public TorchTensor uniform_(double from, double to, torch.Generator? generator = null)
        {
            var res = THSTensor_uniform_(handle, from, to, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange_out(IntPtr start, IntPtr strp, IntPtr step, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with with values from interval [start, end) and
		/// common difference step, starting from start.
        /// </summary>
        public TorchTensor arange_out(TorchScalar start, TorchScalar stop, TorchScalar step)
        {
            var res = THSTensor_arange_out(start.Handle, stop.Handle, step.Handle, handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_permute(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        ///  Returns a view of the original tensor with its dimensions permuted.
        /// </summary>
        /// <param name="permutation">The desired ordering of dimensions</param>
        public TorchTensor permute(long[] permutation)
        {
            unsafe {
                fixed (long* pPermutation = permutation) {
                    var res = THSTensor_permute(handle, (IntPtr)pPermutation, permutation.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 1
        /// </summary>
        public TorchTensor ones(params long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_ones_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 0
        /// </summary>
        public TorchTensor zeros(params long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_zeros_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Returns a tensor filled with the scalar value 0, with the same size as input.
        /// </summary>
        public TorchTensor zeros_like(ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_zeros_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_zeros_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Returns a tensor filled with the scalar value 1, with the same size as input.
        /// </summary>
        public TorchTensor ones_like(ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_ones_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_ones_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values uninitialized
        /// </summary>
        public TorchTensor empty(params long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_empty_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns an uninitialized tensor with the same size as input.
        /// </summary>
        public TorchTensor empty_like(ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_empty_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_empty_like(handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full_out(IntPtr psizes, int length, IntPtr value, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values uninitialized
        /// </summary>
        public TorchTensor full(long[] sizes, TorchScalar value)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_full_out((IntPtr)psizes, sizes.Length, value.Handle, handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full_like(IntPtr input, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Returns a tensor with the same size as input filled with 'value.'
        /// </summary>
        public TorchTensor full_like(TorchScalar value, ScalarType? dtype = null, torch.device? device = null, bool requiresGrad = false)
        {
            dtype = (dtype is null) ? this.Type : dtype;
            device = (device is null) ? this.device : device;

            var result = THSTensor_full_like(handle, value.Handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            if (result == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                result = THSTensor_full_like(handle, value.Handle, (sbyte)dtype, (int)device.Type, device.Index, requiresGrad);
            }
            if (result == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(result);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye_out(long rows, long columns, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor into a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        public TorchTensor eye(long rows, long columns)
        {
            var res = THSTensor_eye_out(rows, columns, handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_scatter(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

        /// <summary>
        ///  Writes all values from the tensor src into self at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public TorchTensor scatter(long dimension, TorchTensor index, TorchTensor src)
        {
            var res = THSTensor_scatter(handle, dimension, index.Handle, src.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_gather(IntPtr tensor, long dimension, IntPtr index);

        /// <summary>
        /// Gathers values along an axis specified by dim.
        /// </summary>
        public TorchTensor gather(long dimension, TorchTensor index)
        {
            var res = THSTensor_gather(handle, dimension, index.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_flip(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        ///  Reverse the order of a n-D tensor along given axis in dims.
        /// </summary>
        public TorchTensor flip(params long[] sizes)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    var res = THSTensor_flip(handle, (IntPtr)psizes, sizes.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_fliplr(IntPtr tensor);

        /// <summary>
        /// Flip tensor in the left/right direction, returning a new tensor.
        /// </summary>
        public TorchTensor fliplr()
        {
            var res = THSTensor_fliplr(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_flipud(IntPtr tensor);

        /// <summary>
        /// Flip tensor in the up/down direction, returning a new tensor.
        /// </summary>
        public TorchTensor flipud()
        {
            var res = THSTensor_flipud(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_nanmedian(IntPtr tensor);

        /// <summary>
        /// Returns the median of the values in input, ignoring NaN values.
        /// </summary>
        public TorchTensor nanmedian()
        {
            var res = THSTensor_nanmedian(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_nansum(IntPtr tensor);

        /// <summary>
        /// Returns the sum of all elements in the input tensor, treating NaN as zero.
        /// </summary>
        public TorchTensor nansum()
        {
            var res = THSTensor_nansum(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_nan_to_num(IntPtr tensor, IntPtr nan, IntPtr posinf, IntPtr neginf);

        /// <summary>
        /// Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
        /// By default, NaN`s are replaced with zero, positive infinity is replaced with the greatest finite value representable by input’s dtype,
        /// and negative infinity is replaced with the least finite value representable by input’s dtype.
        /// </summary>
        public TorchTensor nan_to_num(double? nan = null, double? posinf = null, double? neginf = null)
        {
            var _nan = nan.HasValue ? new double[] { nan.Value } : null;
            var _posinf = posinf.HasValue ? new double[] { posinf.Value } : null;
            var _neginf = neginf.HasValue ? new double[] { neginf.Value } : null;
            unsafe {
                fixed (double* pnan = _nan, pposinf = _posinf, pneginf = _neginf) {
                    var res =
                        THSTensor_nan_to_num(handle, (IntPtr)pnan, (IntPtr)pposinf, (IntPtr)pneginf);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_nextafter(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Return the next floating-point value after input towards other, elementwise.
        /// </summary>
        public TorchTensor nextafter(TorchTensor other)
        {
            var res = THSTensor_nextafter(handle, other.handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_narrow(IntPtr tensor, long dimension, long start, long length);

        /// <summary>
        ///  Returns a new tensor that is a narrowed version of the input along one dimension. The
        /// dimension is input from start to start + length. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor narrow(long dimension, long start, long length)
        {
            var res = THSTensor_narrow(handle, dimension, start, length);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_nonzero(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor nonzero()
        {
            var res = THSTensor_nonzero(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public IList<TorchTensor> nonzero_as_list()
        {
            var res = THSTensor_nonzero(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }

            var t = new TorchTensor(res);
            return t.chunk(t.shape[1], dim: 1);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_slice(IntPtr tensor, long dimension, long start, long length, long step);

        /// <summary>
        /// Returns a new tensor that is a sliced version of the input along one dimension. The
        /// dimension is input from start to finish-1. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor slice(long dimension, long start, long finish, long step)
        {
            var res = THSTensor_slice(handle, dimension, start, finish, step);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_unsqueeze(IntPtr tensor, long dimension);

        public TorchTensor unsqueeze(long dimension)
        {
            var res = THSTensor_unsqueeze(handle, dimension);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_where(IntPtr condition, IntPtr x, IntPtr y);

        public TorchTensor where(TorchTensor condition, TorchTensor other)
        {
            if (condition.Type != ScalarType.Bool) throw new ArgumentException("The condition to 'where' must be a boolean tensor.");

            var res = THSTensor_where(condition.Handle, this.Handle, other.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_einsum([MarshalAs(UnmanagedType.LPStr)] string location, IntPtr tensors, int len);

        public static TorchTensor einsum(string equation, params TorchTensor[] tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_einsum(equation, tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }


        // Operators overloading

        public static TorchTensor operator ==(TorchTensor left, TorchTensor right)
        {
            return left.eq(right);
        }

        public static TorchTensor operator ==(TorchTensor left, TorchScalar right)
        {
            return left.eq(right);
        }

        public static TorchTensor operator ==(TorchScalar left, TorchTensor right)
        {
            return right.eq(left);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchTensor right)
        {
            return left.ne(right);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchScalar right)
        {
            return left.ne(right);
        }

        public static TorchTensor operator !=(TorchScalar left, TorchTensor right)
        {
            return right.ne(left);
        }

        public static TorchTensor operator <(TorchTensor left, TorchTensor right)
        {
            return left.lt(right);
        }

        public static TorchTensor operator <(TorchTensor left, TorchScalar right)
        {
            return left.lt(right);
        }

        public static TorchTensor operator <(TorchScalar left, TorchTensor right)
        {
            return right.gt(left);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchTensor right)
        {
            return left.le(right);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchScalar right)
        {
            return left.le(right);
        }

        public static TorchTensor operator <=(TorchScalar left, TorchTensor right)
        {
            return right.ge(left);
        }

        public static TorchTensor operator >(TorchTensor left, TorchTensor right)
        {
            return left.gt(right);
        }

        public static TorchTensor operator >(TorchTensor left, TorchScalar right)
        {
            return left.gt(right);
        }

        public static TorchTensor operator >(TorchScalar left, TorchTensor right)
        {
            return right.lt(left);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchTensor right)
        {
            return left.ge(right);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchScalar right)
        {
            return left.ge(right);
        }

        public static TorchTensor operator >=(TorchScalar left, TorchTensor right)
        {
            return right.le(left);
        }

        // Specifically added to make F# look good.
        public static TorchTensor op_MinusMinusGreater(TorchTensor t, torch.nn.Module m) => m.forward(t);

        /// <summary>
        ///   Get a string representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            if (Handle == IntPtr.Zero) return "";

            var n = Dimensions;
            if (n == 0)
                return "[]";

            var sb = new StringBuilder("[");
            for (var i = 0; i < n; i++)
            {
                sb.Append(size(i));
                if (i + 1 < n)
                    sb.Append("x");
            }

            sb.Append("]");
            sb.Append($", device = {device}");
            return sb.ToString();
        }

        public static explicit operator float (TorchTensor value) => value.ToSingle();
        public static explicit operator double (TorchTensor value) => value.ToDouble();
        public static explicit operator sbyte (TorchTensor value) => value.ToSByte();
        public static explicit operator byte (TorchTensor value) => value.ToByte();
        public static explicit operator short (TorchTensor value) => value.ToInt16();
        public static explicit operator int (TorchTensor value) => value.ToInt32();
        public static explicit operator long (TorchTensor value) => value.ToInt64();
        public static explicit operator bool (TorchTensor value) => value.ToBoolean();

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_block_diag(IntPtr tensor, int len);

        public static TorchTensor block_diag(params TorchTensor[] tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_block_diag(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_atleast_1d(IntPtr tensor);

        public TorchTensor atleast_1d()
        {
            var res = THSTensor_atleast_1d(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_atleast_2d(IntPtr tensor);

        public TorchTensor atleast_2d()
        {
            var res = THSTensor_atleast_2d(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_atleast_3d(IntPtr tensor);

        public TorchTensor atleast_3d()
        {
            var res = THSTensor_atleast_3d(handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }

    /// <summary>
    /// Type used to represent the variety of indexing capabilities that are
    /// available in Pyton, and therefore to PyTorch.
    /// </summary>
    public struct TorchTensorIndex
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
        internal TorchTensor? tensor;
        static public TorchTensorIndex Slice(long? start = null, long? stop = null, long? step = null)
        {
            return new TorchTensorIndex() { startIndexOrBoolOrSingle = start, step = step, stopIndex = stop, kind = Kind.Slice };
        }
        static public TorchTensorIndex Bool(bool value) => new TorchTensorIndex() { startIndexOrBoolOrSingle = (value ? 1 : 0), kind = Kind.Bool };
        static public TorchTensorIndex Single(long? index) => new TorchTensorIndex() { startIndexOrBoolOrSingle = index, kind = Kind.Single };
        static public TorchTensorIndex Tensor(TorchTensor tensor) => new TorchTensorIndex() { tensor = tensor, kind = Kind.Tensor };
        static public TorchTensorIndex Ellipsis => new TorchTensorIndex() { kind = Kind.Ellipsis };
        static public TorchTensorIndex None => new TorchTensorIndex() { kind = Kind.None };
        static public TorchTensorIndex Null => new TorchTensorIndex() { kind = Kind.Null };
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
}