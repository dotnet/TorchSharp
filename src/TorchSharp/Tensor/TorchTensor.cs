// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable
namespace TorchSharp.Tensor
{
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

    public sealed class TorchTensor : IDisposable
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

        [DllImport("LibTorchSharp")]
        static extern long THSTensor_element_size(IntPtr handle);

        [DllImport("LibTorchSharp")]
        static extern long THSTensor_numel(IntPtr handle);

        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumberOfElements => THSTensor_numel(handle);

        /// <summary>
        ///  Get the size of each element in the tensor.
        /// </summary>
        public long ElementSize => THSTensor_element_size(handle);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_data(IntPtr handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<T> Data<T>()
        {
            if (NumberOfElements > int.MaxValue)
            {
                throw new ArgumentException("Span only supports up to int.MaxValue elements.");
            }
            unsafe
            {
                var res = THSTensor_data(handle);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                // NOTE: there is no safety here.
                return new Span<T>((void*)res, (int)NumberOfElements);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T DataItem<T>()
        {
            if (NumberOfElements != 1) throw new ArgumentException("Number of elements in the tensor must be 1");

            return Data<T>()[0];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public double ReadCpuDouble(long i) => Data<double>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public float ReadCpuSingle(long i) => Data<float>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public int ReadCpuInt32(long i) => Data<int>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public long ReadCpuInt64(long i) => Data<long>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public byte ReadCpuByte(long i) => Data<byte>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public sbyte ReadCpuSByte(long i) => Data<sbyte>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public short ReadCpuInt16(long i) => Data<short>()[(int)i];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public bool ReadCpuBool(long i) => Data<bool>()[(int)i];

        [DllImport("LibTorchSharp")]
        static extern float THSTensor_data_idx_float16(IntPtr handle, long i);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i"></param>
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
        /// 
        /// </summary>
        /// <param name="i"></param>
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
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchScalar ToScalar()
        {
            var res = THSTensor_item(Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchScalar(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor fill_(TorchScalar value)
        {
            var res = THSTensor_fill_(handle, value.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_get1(IntPtr handle, long i1);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set1(IntPtr handle, long i1, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i1"></param>
        /// <returns></returns>
        [IndexerName("TensorItems")]
        public TorchTensor this[long i1]
        {
            get
            {
                var res = THSTensor_get1(handle, i1);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set1(handle, i1, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i2"></param>
        /// <returns></returns>
        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2]
        {
            get
            {
                var res = THSTensor_get2(handle, i1, i2);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set2(handle, i1, i2, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i2"></param>
        /// <param name="i3"></param>
        /// <returns></returns>
        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3]
        {
            get
            {
                var res = THSTensor_get3(handle, i1, i2, i3);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set3(handle, i1, i2, i3, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i2"></param>
        /// <param name="i3"></param>
        /// <param name="i4"></param>
        /// <returns></returns>
        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4]
        {
            get
            {
                var res = THSTensor_get4(handle, i1, i2, i3, i4);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set4(handle, i1, i2, i3, i4, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_get5(IntPtr handle, long i1, long i2, long i3, long i4, long i5);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set5(IntPtr handle, long i1, long i2, long i3, long i4, long i5, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i2"></param>
        /// <param name="i3"></param>
        /// <param name="i4"></param>
        /// <param name="i5"></param>
        /// <returns></returns>
        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5] {
            get {
                var res = THSTensor_get5(handle, i1, i2, i3, i4, i5);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set5(handle, i1, i2, i3, i4, i5, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_get6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i2"></param>
        /// <param name="i3"></param>
        /// <param name="i4"></param>
        /// <param name="i5"></param>
        /// <param name="i6"></param>
        /// <returns></returns>
        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5, long i6] {
            get {
                var res = THSTensor_get6(handle, i1, i2, i3, i4, i5, i6);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set6(handle, i1, i2, i3, i4, i5, i6, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }
        [DllImport("LibTorchSharp")]
        static extern sbyte THSTensor_type(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public ScalarType Type => (ScalarType)THSTensor_type(handle);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        static extern string THSTensor_device_str(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public string device
        {
            get
            {
                var res = THSTensor_device_str(handle);
                if (res == null)
                    Torch.CheckForErrors();
                return res!;
            }
        }


        [DllImport("LibTorchSharp")]
        static extern int THSTensor_device_index(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public int device_index {
            get {
                var res = THSTensor_device_index(handle);
                Torch.CheckForErrors();
                return res;
            }
        }


        [DllImport("LibTorchSharp")]
        static extern int THSTensor_device_type(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public DeviceType device_type {
            get {
                var res = THSTensor_device_type(handle);
                Torch.CheckForErrors();
                return (DeviceType)res;
            }
        }


        [DllImport("LibTorchSharp")]
        static extern bool THSTensor_is_sparse(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public bool IsSparse
        {
            get
            {
                var res = THSTensor_is_sparse(handle);
                Torch.CheckForErrors();
                return res;
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public TorchTensor to_type(ScalarType type)
        {
            var res = THSTensor_to_type(handle, (sbyte)type);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_load([MarshalAs(UnmanagedType.LPStr)] string location);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="location"></param>
        /// <returns></returns>
        public static TorchTensor load(string location)
        {
            var res = THSTensor_load(location);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_save(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string location);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="location"></param>
        public void save(string location)
        {
            THSTensor_save(handle, location);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        static extern bool THSTensor_requires_grad(IntPtr handle);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_set_requires_grad(IntPtr handle, bool requires_grad);

        /// <summary>
        /// 
        /// </summary>
        public bool requires_grad {
            get { return THSTensor_requires_grad(handle); }
            set {
                var res = THSTensor_set_requires_grad(handle, value);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public TorchTensor with_requires_grad()
        {
            this.requires_grad = true;
            return this;
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cpu(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor cpu()
        {
            var res = THSTensor_cpu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cuda(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor cuda()
        {
            Torch.InitializeDeviceType(DeviceType.CUDA);
            var res = THSTensor_cuda(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_to_device(IntPtr handle, int device_type, int device_index);

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="deviceType">The device type, e.g. 'CPU' or 'CUDA'.</param>
        /// <param name="deviceIndex">The optional device index.</param>
        /// <returns></returns>
        public TorchTensor to(DeviceType deviceType, int deviceIndex = -1)
        {
            Torch.InitializeDeviceType(deviceType);
            var res = THSTensor_to_device(handle, (int)deviceType, deviceIndex);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="device">A string denoting the target device.</param>
        /// <returns></returns>
        public TorchTensor to(string device)
        {
            return to(new Device(device));
        }

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="device">The target device</param>
        /// <returns></returns>
        public TorchTensor to(Device device)
        {
            return to(device.Type, device.Index);
        }

        /// <summary>
        /// Moves the tensor data.
        /// </summary>
        /// <param name="other">The tensor serving as a template.</param>
        /// <returns></returns>
        public TorchTensor to(TorchTensor other)
        {
            return to(other.device_type, other.device_index);
        }

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
            Torch.CheckForErrors();
            return res;
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
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
                var dims = new long[Dimensions];
                for (var i = 0; i < dims.Length; i++)
                    dims[i] = size(i);

                return dims;
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_indices(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public TorchTensor SparseIndices {
            get {
                var res = THSTensor_indices(handle);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_values(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public TorchTensor SparseValues {
            get {
                var res = THSTensor_values(handle);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        static extern long THSTensor_stride(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride(int dim)
        {
            var res = THSTensor_stride(handle, dim);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_backward(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        public void backward()
        {
            THSTensor_backward(handle);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_to_dense(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor to_dense()
        {
            var res = THSTensor_to_dense(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_clone(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor clone()
        {
            var res = THSTensor_clone(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_contiguous(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor contiguous()
        {
            var res = THSTensor_contiguous(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_grad(IntPtr handle);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor grad()
        {
            var res = THSTensor_grad(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
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
        /// 
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public TorchTensor index(TorchTensorIndex[] indices)
        {
            EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
            unsafe {
                fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                    fixed (IntPtr* ptrTensors = arrTensors) {
                        var res = THSTensor_index(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length);
                        if (res == IntPtr.Zero)
                            Torch.CheckForErrors();
                        GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                        return new TorchTensor(res);
                    }
                }
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor index_put_(TorchTensorIndex[] indices, TorchTensor value)
        {
            EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
            unsafe {
                fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                    fixed (IntPtr* ptrTensors = arrTensors) {
                        var res = THSTensor_index_put_(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
                        if (res == IntPtr.Zero)
                            Torch.CheckForErrors();
                        GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                        GC.KeepAlive(value); 
                        return new TorchTensor(res);
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor index_put_(TorchTensorIndex[] indices, TorchScalar value)
        {
            EncodeIndices(indices, out var arrKindAndStarts, out var arrStops, out var arrSteps, out var arrTensors);
            unsafe {
                fixed (long* ptrKindAndStarts = arrKindAndStarts, ptrStops = arrStops, ptrSteps = arrSteps) {
                    fixed (IntPtr* ptrTensors = arrTensors) {
                        var res = THSTensor_index_put_scalar_(handle, (IntPtr)ptrKindAndStarts, (IntPtr)ptrStops, (IntPtr)ptrSteps, (IntPtr)ptrTensors, indices.Length, value.Handle);
                        if (res == IntPtr.Zero)
                            Torch.CheckForErrors();
                        GC.KeepAlive(indices); // don't release or finalize Tensor indices whose handles have been put into ptrTensors
                        GC.KeepAlive(value);
                        return new TorchTensor(res);
                    }
                }
            }
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_index_select(IntPtr tensor, long dimension, IntPtr index);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public TorchTensor index_select(long dimension, TorchTensor index)
        {
            var res = THSTensor_index_select(handle, dimension, index.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_reshape(IntPtr tensor, IntPtr shape, int length);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public TorchTensor reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    var res = THSTensor_reshape(handle, (IntPtr)pshape, shape.Length);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_squeeze(IntPtr tensor, long dimension);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        public TorchTensor squeeze(long dimension)
        {
            var res = THSTensor_squeeze(handle, dimension);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_t(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor t()
        {
            var res = THSTensor_t(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_transpose(IntPtr tensor, long dim1, long dim2);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension1"></param>
        /// <param name="dimension2"></param>
        /// <returns></returns>
        public TorchTensor transpose(long dimension1, long dimension2)
        {
            var res = THSTensor_transpose(handle, dimension1, dimension2);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_transpose_(IntPtr tensor, long dim1, long dim2);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimension1"></param>
        /// <param name="dimension2"></param>
        /// <returns></returns>
        public TorchTensor transpose_(long dimension1, long dimension2)
        {
            return new TorchTensor(THSTensor_transpose_(handle, dimension1, dimension2));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_view(IntPtr tensor, IntPtr shape, int length);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public TorchTensor view(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    var res = THSTensor_view(handle, (IntPtr)pshape, shape.Length);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor add(TorchTensor target)
        {
            return add(target, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add(TorchTensor target, TorchScalar alpha)
        {
            var res = THSTensor_add(handle, target.Handle, alpha.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add_scalar(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor add(TorchScalar scalar)
        {
            return add(scalar, 1);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add(TorchScalar scalar, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_scalar(handle, scalar.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchTensor target)
        {
            return add_(target, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchTensor target, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_(handle, target.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add_scalar_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchScalar scalar)
        {
            return add_(scalar, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchScalar scalar, TorchScalar alpha)
        {
            var res = THSTensor_add_scalar_(handle, scalar.Handle, alpha.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="batch1"></param>
        /// <param name="batch2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addbmm(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addbmm_(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="batch1"></param>
        /// <param name="batch2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addbmm_(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm_(handle, batch1.Handle, batch2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcdiv(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcdiv(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcdiv_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcdiv_(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcmul(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcmul(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcmul_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcmul_(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="mat2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmm(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmm_(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="mat2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmm_(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm_(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmv(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="vec2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmv(TorchTensor mat1, TorchTensor vec2, float beta, float alpha)
        {
            var res = THSTensor_addmv(handle, mat1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmv_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="vec2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmv_(TorchTensor mat1, TorchTensor vec2, float beta, float alpha)
        {
            var res = THSTensor_addmv_(handle, mat1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
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
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
            return new TorchTensor(res);
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
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cos(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor cos()
        {
            var res = THSTensor_cos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cos_(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor cos_()
        {
            var res = THSTensor_cos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sin(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor sin()
        {
            var res = THSTensor_sin(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sin_(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor sin_()
        {
            var res = THSTensor_sin_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tan(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor tan()
        {
            var res = THSTensor_tan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tan_(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor tan_()
        {
            var res = THSTensor_tan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_angle(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor angle()
        {
            return new TorchTensor(THSTensor_angle(handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_asin(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor asin()
        {
            return new TorchTensor(THSTensor_asin(handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_asin_(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor asin_()
        {
            var res = THSTensor_asin_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_acos(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor acos()
        {
            var res = THSTensor_acos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_acos_(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor acos_()
        {
            var res = THSTensor_acos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor atan()
        {
            var res = THSTensor_atan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan_(IntPtr tensor);

        public TorchTensor atan_()
        {
            var res = THSTensor_atan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan2(IntPtr tensor, IntPtr other);

        public TorchTensor atan2(TorchTensor other)
        {
            var res = THSTensor_atan2(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan2_(IntPtr tensor, IntPtr other);

        public TorchTensor atan2_(TorchTensor other)
        {
            var res = THSTensor_atan2_(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinh(IntPtr tensor);

        public TorchTensor Sinh()
        {
            var res = THSTensor_sinh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinh_(IntPtr tensor);

        public TorchTensor Sinh_()
        {
            var res = THSTensor_sinh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cosh(IntPtr tensor);

        public TorchTensor cosh()
        {
            var res = THSTensor_cosh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cosh_(IntPtr tensor);

        public TorchTensor cosh_()
        {
            var res = THSTensor_cosh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tanh(IntPtr tensor);

        public TorchTensor tanh()
        {
            var res = THSTensor_tanh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tanh_(IntPtr tensor);

        public TorchTensor tanh_()
        {
            var res = THSTensor_tanh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_floor(IntPtr tensor);

        public TorchTensor floor()
        {
            var res = THSTensor_floor(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_floor_(IntPtr tensor);

        public TorchTensor floor_()
        {
            var res = THSTensor_floor_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_digamma(IntPtr tensor);

        public TorchTensor digamma()
        {
            var res = THSTensor_digamma(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_digamma_(IntPtr tensor);

        public TorchTensor digamma_()
        {
            var res = THSTensor_digamma_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lgamma(IntPtr tensor);

        public TorchTensor lgamma()
        {
            var res = THSTensor_lgamma(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lgamma_(IntPtr tensor);

        public TorchTensor lgamma_()
        {
            var res = THSTensor_lgamma_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mvlgamma(IntPtr tensor, long p);

        public TorchTensor mvlgamma(long p)
        {
            var res = THSTensor_mvlgamma(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mvlgamma_(IntPtr tensor, long p);

        public TorchTensor mvlgamma_(long p)
        {
            var res = THSTensor_mvlgamma_(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_polygamma(IntPtr tensor, long p);

        public TorchTensor polygamma(long p)
        {
            var res = THSTensor_polygamma(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_polygamma_(IntPtr tensor, long p);

        public TorchTensor polygamma_(long p)
        {
            var res = THSTensor_polygamma_(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ceil(IntPtr tensor);

        public TorchTensor ceil()
        {
            var res = THSTensor_ceil(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ceil_(IntPtr tensor);

        public TorchTensor ceil_()
        {
            var res = THSTensor_ceil_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sign(IntPtr tensor);

        public TorchTensor sign()
        {
            var res = THSTensor_sign(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sign_(IntPtr tensor);

        public TorchTensor sign_()
        {
            var res = THSTensor_sign_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_softplus(IntPtr tensor);

        public TorchTensor softplus()
        {
            var res = THSTensor_softplus(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu(IntPtr tensor);

        public TorchTensor relu()
        {
            var res = THSTensor_relu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu_(IntPtr tensor);

        public TorchTensor relu_()
        {
            var res = THSTensor_relu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu6(IntPtr tensor);

        public TorchTensor relu6()
        {
            var res = THSTensor_relu6(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_relu6_(IntPtr tensor);

        public TorchTensor relu6_()
        {
            var res = THSTensor_relu6_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_celu(IntPtr tensor);

        public TorchTensor celu()
        {
            var res = THSTensor_celu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_celu_(IntPtr tensor);

        public TorchTensor celu_()
        {
            var res = THSTensor_celu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_elu(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        public TorchTensor elu(TorchScalar alpha, TorchScalar scale, TorchScalar input_scale)
        {
            var res = THSTensor_elu(handle, alpha.Handle, scale.Handle, input_scale.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_elu_(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        public TorchTensor elu_(TorchScalar alpha, TorchScalar scale, TorchScalar input_scale)
        {
            var res = THSTensor_elu_(handle, alpha.Handle, scale.Handle, input_scale.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gelu(IntPtr tensor);

        public TorchTensor gelu()
        {
            var res = THSTensor_gelu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardsigmoid(IntPtr tensor);

        public TorchTensor hardsigmoid()
        {
            var res = THSTensor_hardsigmoid(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardsigmoid_(IntPtr tensor);

        public TorchTensor hardsigmoid_()
        {
            var res = THSTensor_hardsigmoid_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardswish(IntPtr tensor);

        public TorchTensor hardswish()
        {
            var res = THSTensor_hardswish(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardswish_(IntPtr tensor);

        public TorchTensor hardswish_()
        {
            var res = THSTensor_hardswish_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardtanh(IntPtr tensor, IntPtr min, IntPtr max);

        public TorchTensor hardtanh(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_hardtanh(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hardtanh_(IntPtr tensor, IntPtr min, IntPtr max);

        public TorchTensor hardtanh_(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_hardtanh_(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_leaky_relu(IntPtr tensor, IntPtr negative_slope);

        public TorchTensor leaky_relu(TorchScalar negative_slope)
        {
            var res = THSTensor_leaky_relu(handle, negative_slope.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_leaky_relu_(IntPtr tensor, IntPtr negative_slope);

        public TorchTensor leaky_relu_(TorchScalar negative_slope)
        {
            var res = THSTensor_leaky_relu_(handle, negative_slope.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_round(IntPtr tensor);

        public TorchTensor round()
        {
            var res = THSTensor_round(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_round_(IntPtr tensor);

        public TorchTensor round_()
        {
            var res = THSTensor_round_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_selu(IntPtr tensor);

        public TorchTensor selu()
        {
            var res = THSTensor_selu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_selu_(IntPtr tensor);

        public TorchTensor selu_()
        {
            var res = THSTensor_selu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_silu(IntPtr tensor);

        public TorchTensor silu()
        {
            var res = THSTensor_silu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_silu_(IntPtr tensor);

        public TorchTensor silu_()
        {
            var res = THSTensor_silu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_abs(IntPtr tensor);

        public TorchTensor abs()
        {
            var res = THSTensor_abs(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_abs_(IntPtr tensor);

        public TorchTensor abs_()
        {
            var res = THSTensor_abs_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log_sigmoid(IntPtr tensor);

        public TorchTensor log_sigmoid()
        {
            var res = THSTensor_log_sigmoid(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

        public TorchTensor logcumsumexp(long dim)
        {
            var res = THSTensor_logcumsumexp(handle, dim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log10(IntPtr tensor);

        public TorchTensor log10()
        {
            var res = THSTensor_log10(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log10_(IntPtr tensor);

        public TorchTensor log10_()
        {
            var res = THSTensor_log10_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lerp(IntPtr tensor, IntPtr end, IntPtr weight);

        public TorchTensor lerp(TorchTensor end, TorchTensor weight)
        {
            var res = THSTensor_lerp(handle, end.Handle, weight.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lerp_(IntPtr tensor, IntPtr end, IntPtr weight);

        public TorchTensor lerp_(TorchTensor end, TorchTensor weight)
        {
            var res = THSTensor_lerp_(handle, end.Handle, weight.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log1p(IntPtr tensor);

        public TorchTensor log1p()
        {
            var res = THSTensor_log1p(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log1p_(IntPtr tensor);

        public TorchTensor log1p_()
        {
            var res = THSTensor_log1p_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sqrt(IntPtr tensor);

        public TorchTensor sqrt()
        {
            var res = THSTensor_sqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sqrt_(IntPtr tensor);

        public TorchTensor sqrt_()
        {
            var res = THSTensor_sqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rsqrt(IntPtr tensor);

        public TorchTensor rsqrt()
        {
            var res = THSTensor_rsqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rsqrt_(IntPtr tensor);

        public TorchTensor rsqrt_()
        {
            var res = THSTensor_rsqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public static TorchTensor operator -(TorchTensor tensor)
        {
            return tensor.neg();
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_neg(IntPtr tensor);

        public TorchTensor neg()
        {
            var res = THSTensor_neg(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_neg_(IntPtr tensor);

        public TorchTensor neg_()
        {
            var res = THSTensor_neg_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
            float alpha);

        public TorchTensor baddbmm(TorchTensor batch2, TorchTensor mat, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_baddbmm(handle, batch2.Handle, mat.Handle, beta, alpha);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        public TorchTensor bmm(TorchTensor batch2)
        {
            var res = THSTensor_bmm(handle, batch2.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_and(TorchTensor other)
        {
            var res = THSTensor_bitwise_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_and_(TorchTensor other)
        {
            var res = THSTensor_bitwise_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

        public TorchTensor bitwise_not()
        {
            var res = THSTensor_bitwise_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

        public TorchTensor bitwise_not_(TorchTensor other)
        {
            var res = THSTensor_bitwise_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_or(TorchTensor other)
        {
            var res = THSTensor_bitwise_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_or_(TorchTensor other)
        {
            var res = THSTensor_bitwise_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_xor(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseXor_(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

        public TorchTensor logical_and(TorchTensor other)
        {
            var res = THSTensor_logical_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

        public TorchTensor logical_and_(TorchTensor other)
        {
            var res = THSTensor_logical_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_not(IntPtr tensor);

        public TorchTensor logical_not()
        {
            var res = THSTensor_logical_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

        public TorchTensor logical_not_(TorchTensor other)
        {
            var res = THSTensor_logical_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

        public TorchTensor logical_or(TorchTensor other)
        {
            var res = THSTensor_logical_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

        public TorchTensor logical_or_(TorchTensor other)
        {
            var res = THSTensor_logical_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

        public TorchTensor logical_xor(TorchTensor other)
        {
            var res = THSTensor_logical_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor logical_xor_(TorchTensor other)
        {
            var res = THSTensor_logical_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky(IntPtr input, bool upper);

        public TorchTensor cholesky(bool upper = false)
        {
            var res = THSTensor_cholesky(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, bool upper);

        public TorchTensor cholesky_inverse(bool upper = false)
        {
            var res = THSTensor_cholesky_inverse(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, bool upper);

        public TorchTensor CholeskySolve(TorchTensor input2, bool upper = false)
        {
            var res = THSTensor_cholesky_solve(handle, input2.Handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor clamp(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor clamp_(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp_(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

        public TorchTensor clamp_max(TorchScalar max)
        {
            var res = THSTensor_clamp_max(handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

        public TorchTensor clamp_max_(TorchScalar max)
        {
            var res = THSTensor_clamp_max_(handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

        public TorchTensor clamp_min(TorchScalar min)
        {
            var res = THSTensor_clamp_min(handle, min.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

        public TorchTensor clamp_min_(TorchScalar min)
        {
            var res = THSTensor_clamp_min_(handle, min.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public TorchTensor cross(TorchScalar other, long dim)
        {
            var res = THSTensor_cross(handle, other.Handle, dim);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_cummax(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public (TorchTensor values, TorchTensor indexes) cummax(long dimension)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_cummax(handle, pa.CreateArray, dimension);
                Torch.CheckForErrors();
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
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cumsum(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

        public TorchTensor cumsum(long dimension, ScalarType? type = null)
        {
            var res = THSTensor_cumsum(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cumprod(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

        public TorchTensor cumprod(long dimension, ScalarType? type = null)
        {
            var res = THSTensor_cumprod(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg);

        public TorchTensor div(TorchTensor target)
        {
            var res = THSTensor_div(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor div(TorchScalar target)
        {
            var res = THSTensor_div_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg);

        public TorchTensor div_(TorchTensor target)
        {
            var res = THSTensor_div_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor div_(TorchScalar target)
        {
            var res = THSTensor_div_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erf(IntPtr tensor);

        public TorchTensor erf()
        {
            var res = THSTensor_erf(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erf_(IntPtr tensor);

        public TorchTensor erf_()
        {
            var res = THSTensor_erf_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfc(IntPtr tensor);

        public TorchTensor erfc()
        {
            var res = THSTensor_erfc(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfc_(IntPtr tensor);

        public TorchTensor erfc_()
        {
            var res = THSTensor_erfc_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfinv(IntPtr tensor);

        public TorchTensor erfinv()
        {
            var res = THSTensor_erfinv(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_erfinv_(IntPtr tensor);

        public TorchTensor erfinv_()
        {
            var res = THSTensor_erfinv_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq(IntPtr tensor, IntPtr trg);

        public TorchTensor eq(TorchTensor target)
        {
            var res = THSTensor_eq(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq_(IntPtr tensor, IntPtr trg);

        public TorchTensor eq_(TorchTensor target)
        {
            var res = THSTensor_eq_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor eq(TorchScalar target)
        {
            var res = THSTensor_eq_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_eq_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor eq_(TorchScalar target)
        {
            var res = THSTensor_eq_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern bool THSTensor_equal(IntPtr tensor, IntPtr trg);

        public bool Equals(TorchTensor target)
        {
            var res = THSTensor_equal(handle, target.Handle);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        static extern bool THSTensor_allclose(IntPtr tensor, IntPtr trg, double rtol, double atol, bool equal_nan);

        public bool allclose(TorchTensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
        {
            var res = THSTensor_allclose(handle, target.Handle, rtol, atol, equal_nan);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exp(IntPtr tensor);

        public TorchTensor exp()
        {
            var res = THSTensor_exp(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exp_(IntPtr tensor);

        public TorchTensor exp_()
        {
            var res = THSTensor_exp_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_expm1(IntPtr tensor);

        public TorchTensor expm1()
        {
            var res = THSTensor_expm1(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_expm1_(IntPtr tensor);

        public TorchTensor expm1_()
        {
            var res = THSTensor_expm1_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor fft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_fft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ifft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor ifft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_ifft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_irfft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor irfft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_irfft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rfft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor rfft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_rfft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_frac(IntPtr tensor);

        public TorchTensor frac()
        {
            var res = THSTensor_frac(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_frac_(IntPtr tensor);

        public TorchTensor frac_()
        {
            var res = THSTensor_frac_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gcd(IntPtr tensor, IntPtr other);

        public TorchTensor gcd(TorchTensor other)
        {
            var res = THSTensor_gcd(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gcd_(IntPtr tensor, IntPtr other);

        public TorchTensor gcd_(TorchTensor other)
        {
            var res = THSTensor_gcd_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge(IntPtr tensor, IntPtr trg);

        public TorchTensor ge(TorchTensor target)
        {
            var res = THSTensor_ge(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge_(IntPtr tensor, IntPtr trg);

        public TorchTensor ge_(TorchTensor target)
        {
            var res = THSTensor_ge_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor ge(TorchScalar target)
        {
            var res = THSTensor_ge_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ge_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor ge_(TorchScalar target)
        {
            var res = THSTensor_ge_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt(IntPtr tensor, IntPtr trg);

        public TorchTensor gt(TorchTensor target)
        {
            var res = THSTensor_gt(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt_(IntPtr tensor, IntPtr trg);

        public TorchTensor gt_(TorchTensor target)
        {
            var res = THSTensor_gt_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor gt(TorchScalar target)
        {
            var res = THSTensor_gt_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gt_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor gt_(TorchScalar target)
        {
            var res = THSTensor_gt_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lcm(IntPtr tensor, IntPtr other);

        public TorchTensor lcm(TorchTensor other)
        {
            var res = THSTensor_lcm(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lcm_(IntPtr tensor, IntPtr other);

        public TorchTensor lcm_(TorchTensor other)
        {
            var res = THSTensor_lcm_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le(IntPtr tensor, IntPtr trg);

        public TorchTensor le(TorchTensor target)
        {
            var res = THSTensor_le(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le_(IntPtr tensor, IntPtr trg);

        public TorchTensor le_(TorchTensor target)
        {
            var res = THSTensor_le_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor le(TorchScalar target)
        {
            var res = THSTensor_le_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_le_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor le_(TorchScalar target)
        {
            var res = THSTensor_le_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log(IntPtr tensor);

        public TorchTensor log()
        {
            var res = THSTensor_log(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log_(IntPtr tensor);

        public TorchTensor log_()
        {
            var res = THSTensor_log_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt(IntPtr tensor, IntPtr trg);

        public TorchTensor lt(TorchTensor target)
        {
            var res = THSTensor_lt(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt_(IntPtr tensor, IntPtr trg);

        public TorchTensor lt_(TorchTensor target)
        {
            var res = THSTensor_lt_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor lt(TorchScalar target)
        {
            var res = THSTensor_lt_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lt_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor lt_(TorchScalar target)
        {
            var res = THSTensor_lt_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_matmul(IntPtr tensor, IntPtr target);

        public TorchTensor matmul(TorchTensor target)
        {
            var res = THSTensor_matmul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_unbind(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public TorchTensor[] unbind(int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_unbind(handle, pa.CreateArray, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }


        [DllImport("LibTorchSharp")]
        static extern void THSTensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

        public TorchTensor[] split_with_size(long size, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_split_with_size(handle, pa.CreateArray, size, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

        public TorchTensor[] split_with_sizes(long[] sizes, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                unsafe
                {
                    fixed (long* psizes = sizes)
                    {
                        THSTensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max(IntPtr tensor);

        public TorchTensor max()
        {
            var res = THSTensor_max(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_elementwise(IntPtr tensor, IntPtr other);

        public TorchTensor max(TorchTensor other)
        {
            var res = THSTensor_max_elementwise(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mean(IntPtr tensor);

        public TorchTensor mean()
        {
            var res = THSTensor_mean(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        public TorchTensor mean(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe {
                fixed (long* pdims = dimensions) {
                    var res = THSTensor_mean_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_median(IntPtr tensor);

        public TorchTensor median()
        {
            var res = THSTensor_median(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_min(IntPtr tensor);

        public TorchTensor min()
        {
            var res = THSTensor_min(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_min_elementwise(IntPtr tensor, IntPtr other);

        public TorchTensor min(TorchTensor other)
        {
            var res = THSTensor_min_elementwise(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mm(IntPtr tensor, IntPtr target);

        public TorchTensor mm(TorchTensor target)
        {
            var res = THSTensor_mm(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul(IntPtr tensor, IntPtr target);

        public TorchTensor mul(TorchTensor target)
        {
            var res = THSTensor_mul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor mul(TorchScalar scalar)
        {
            var res = THSTensor_mul_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

        public TorchTensor mul_(TorchTensor target)
        {
            var res = THSTensor_mul_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

        public TorchTensor mul_(TorchScalar target)
        {
            var res = THSTensor_mul_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne(IntPtr tensor, IntPtr trg);

        public TorchTensor ne(TorchTensor target)
        {
            var res = THSTensor_ne(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne_(IntPtr tensor, IntPtr trg);

        public TorchTensor ne_(TorchTensor target)
        {
            var res = THSTensor_ne_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor ne(TorchScalar target)
        {
            var res = THSTensor_ne_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ne_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor ne_(TorchScalar target)
        {
            var res = THSTensor_ne_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_dist(IntPtr tensor, IntPtr other, float p);

        public TorchTensor dist(TorchTensor other, float p = 2.0f)
        {
            var res = THSTensor_dist(handle, other.Handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_norm(IntPtr tensor, float p);

        public TorchTensor norm(float p = 2.0f)
        {
            var res = THSTensor_norm(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_norm_along_dimension(IntPtr tensor, int dimension, bool keepdim, float p);

        public TorchTensor norm(int dimension, bool keepdim = false, float p = 2.0f)
        {
            var res = THSTensor_norm_along_dimension(handle, dimension, keepdim, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

        public TorchTensor pow(TorchTensor exponent)
        {
            var res = THSTensor_pow(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

        public TorchTensor pow_(TorchTensor exponent)
        {
            var res = THSTensor_pow_(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor pow(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor pow_(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_prelu(IntPtr tensor, IntPtr trg);

        public TorchTensor prelu(TorchTensor target)
        {
            var res = THSTensor_prelu(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

        public TorchTensor remainder(TorchTensor target)
        {
            var res = THSTensor_remainder(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

        public TorchTensor remainder_(TorchTensor target)
        {
            var res = THSTensor_remainder_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor remainder(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor remainder_(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

        public TorchTensor fmod(TorchTensor target)
        {
            var res = THSTensor_fmod(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

        public TorchTensor fmod_(TorchTensor target)
        {
            var res = THSTensor_fmod_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor fmod(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor fmod_(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_renorm(IntPtr tensor, float p, long dim, float maxnorm);

        public TorchTensor renorm(TorchScalar scalar, float p, long dim, float maxnorm)
        {
            var res = THSTensor_renorm(handle, p, dim, maxnorm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sigmoid(IntPtr tensor);

        public TorchTensor sigmoid()
        {
            var res = THSTensor_sigmoid(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sigmoid_(IntPtr tensor);

        public TorchTensor sigmoid_()
        {
            var res = THSTensor_sigmoid_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

        public TorchTensor sub(TorchTensor target)
        {
            var res = THSTensor_sub(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor sub(TorchScalar target)
        {
            var res = THSTensor_sub_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

        public TorchTensor sub_(TorchTensor target)
        {
            var res = THSTensor_sub_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor sub_(TorchScalar target)
        {
            var res = THSTensor_sub_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sum(IntPtr tensor, bool has_type, sbyte scalar_type);

        /// <summary>
        /// Returns the sum of all elements in the :attr:`input` tensor.
        /// </summary>
        public TorchTensor sum(ScalarType? type = null)
        {
            var res = THSTensor_sum(handle, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bernoulli(IntPtr tensor, double p);

        public TorchTensor bernoulli(double p)
        {
            var res = THSTensor_bernoulli(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bernoulli_(IntPtr tensor, double p);

        public TorchTensor bernoulli_(double p)
        {
            var res = THSTensor_bernoulli_(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cauchy_(IntPtr tensor, double median, double sigma);

        public TorchTensor cauchy_(double median = 0.0, double sigma = 1.0)
        {
            var res = THSTensor_cauchy_(handle, median, sigma);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exponential_(IntPtr tensor, double lambd);

        public TorchTensor exponential_(double lambd = 1.0)
        {
            var res = THSTensor_exponential_(handle, lambd);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_geometric_(IntPtr tensor, double p);

        public TorchTensor geometric_(double p)
        {
            var res = THSTensor_geometric_(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log_normal_(IntPtr tensor, double mean, double std);

        public TorchTensor log_normal_(double mean = 1.0, double std = 2.0)
        {
            var res = THSTensor_log_normal_(handle, mean, std);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_normal_(IntPtr tensor, double mean, double std);

        public TorchTensor normal_(double mean = 0.0, double std = 1.0)
        {
            var res = THSTensor_normal_(handle, mean, std);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_uniform_(IntPtr tensor, double from, double to);

        public TorchTensor uniform_(double from = 0.0, double to = 1.0)
        {
            var res = THSTensor_uniform_(handle, from, to);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_multinomial(IntPtr tensor, double num_samples, bool replacement);

        public TorchTensor multinomial(double num_samples, bool replacement = false)
        {
            var res = THSTensor_multinomial(handle, num_samples, replacement);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
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
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public TorchTensor randn_out(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_randn_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public TorchTensor rand_out(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_rand_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm_out(long n, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        public TorchTensor randperm_out(long n)
        {
            var res = THSTensor_randperm_out(n, handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_permute(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 1
        /// </summary>
        public TorchTensor permute(long[] permutation)
        {
            unsafe {
                fixed (long* pPermutation = permutation) {
                    var res = THSTensor_permute(handle, (IntPtr)pPermutation, permutation.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 1
        /// </summary>
        public TorchTensor ones_out(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_ones_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 0
        /// </summary>
        public TorchTensor zeros_out(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_zeros_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_flip(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        ///  Reverse the order of a n-D tensor along given axis in dims.
        /// </summary>
        public TorchTensor flip(long[] sizes)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    var res = THSTensor_flip(handle, (IntPtr)psizes, sizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_slice(IntPtr tensor, long dimension, long start, long length, long step);

        /// <summary>
        ///  Returns a new tensor that is a sliced version of the input along one dimension. The
        /// dimension is input from start to finish-1. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor slice(long dimension, long start, long finish, long step)
        {
            var res = THSTensor_slice(handle, dimension, start, finish, step);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_einsum([MarshalAs(UnmanagedType.LPStr)] string location, IntPtr tensors, int len);

        public static TorchTensor einsum(string equation, params TorchTensor[] tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_einsum(equation, tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor conv1d(TorchTensor weight, TorchTensor? bias = null, 
            long? stride = null,
            long? padding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddingArray = new long[] { padding ?? 0 }; 
            var dilationArray = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = paddingArray, pdilation = dilationArray)
                {
                    var res =
                        THSTensor_conv1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddingArray.Length,
                            (IntPtr)pdilation, dilationArray.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor conv2d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor conv3d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length, 
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv_transpose1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor conv_transpose1d(TorchTensor weight, TorchTensor? bias = null,
            long? stride = null,
            long? padding = null,
            long? outputPadding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var outputPaddings = new long[] { outputPadding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = paddings, poutputPadding = outputPaddings, pdilation = dilations)
                {
                    var res =
                        THSTensor_conv_transpose1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)poutputPadding, outputPaddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv_transpose2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor conv_transpose2d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv_transpose2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv_transpose3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor conv_transpose3d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv_transpose3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor max_pool1d(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            unsafe
            {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations)
                {
                    var res =
                        THSTensor_max_pool1d(handle, 
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_pool1d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) max_pool1d_with_indices(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                        THSTensor_max_pool1d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor max_pool2d(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe
            {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_max_pool2d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_pool2d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) max_pool2d_with_indices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_max_pool2d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor max_pool3d(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe
            {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_max_pool3d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_pool3d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) max_pool3d_with_indices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_max_pool3d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_maxunpool2d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength);

        public TorchTensor maxunpool2d(TorchTensor indices, long[] outputSize)
        {
            unsafe {
                fixed (long* poutputSize = outputSize) {
                    var res = THSTensor_maxunpool2d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_maxunpool3d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength, IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength);

        public TorchTensor maxunpool3d(TorchTensor indices, long[] outputSize, long[] strides, long[] padding)
        {
            unsafe {
                fixed (long* poutputSize = outputSize, pstrides = strides, ppadding = padding) {
                    var res = THSTensor_maxunpool3d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length,
                        (IntPtr)pstrides, strides.Length,
                        (IntPtr)ppadding, padding.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        public TorchTensor avg_pool1d(long kernelSize, long? stride = null,
            long? padding = null, bool ceil_mode = false, bool count_include_pad = true)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool1d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        public TorchTensor avg_pool2d(long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool2d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        public TorchTensor avg_pool3d(long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool3d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool2d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad,
                long divisorOverride);

        public TorchTensor avg_pool2d_backward(TorchTensor originalInput,
            long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true,
            long divisorOverride = 0)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool2d_backward(handle, originalInput.Handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad,
                            divisorOverride);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool3d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad,
                long divisorOverride);

        public TorchTensor avg_pool3d_backward(TorchTensor originalInput,
            long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true,
            long divisorOverride = 0)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool3d_backward(handle, originalInput.Handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad,
                            divisorOverride);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        public TorchTensor adaptive_avg_pool1d(long outputSize)
        {
            var outputSizes = new long[] { outputSize };
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool1d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        public TorchTensor adaptive_avg_pool2d(long[] outputSizes)
        {
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool2d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool3d(IntPtr input, IntPtr outputSize, int outputSizeLength);

        public TorchTensor adaptive_avg_pool3d(long[] outputSizes)
        {
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool3d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool3d_backward(IntPtr gradOutput, IntPtr originalInput);

        public TorchTensor adaptive_avg_pool3d_backward(TorchTensor originalInput)
        {
            var res = THSTensor_adaptive_avg_pool3d_backward(handle, originalInput.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest1d(long? outputSize, double? scaleFactor)
        {
            var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
            var outputSizesLength = outputSize.HasValue ? 1 : 0;
            var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
            var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest1d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest1d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest1d_backward(long? outputSize, long inputSize, double? scaleFactor)
        {
            var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
            var outputSizesLength = outputSize.HasValue ? 1 : 0;
            var inputSizes = new long[] { inputSize };
            var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
            var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest1d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest2d(long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ?  0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest2d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest2d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest2d_backward(long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest2d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest3d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest3d_backward(long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest3d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest3d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest3d(long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest3d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
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

        public static TorchTensor operator +(TorchTensor left, TorchTensor right)
        {
            return left.add(right);
        }

        public static TorchTensor operator +(TorchTensor left, TorchScalar right)
        {
            return left.add(right);
        }

        public static TorchTensor operator +(TorchScalar left, TorchTensor right)
        {
            return right.add(left);
        }

        public static TorchTensor operator *(TorchTensor left, TorchTensor right)
        {
            return left.mul(right);
        }

        public static TorchTensor operator *(TorchTensor left, TorchScalar right)
        {
            return left.mul(right);
        }

        public static TorchTensor operator *(TorchScalar left, TorchTensor right)
        {
            return right.mul(left);
        }

        public static TorchTensor operator -(TorchTensor left, TorchTensor right)
        {
            return left.sub(right);
        }

        public static TorchTensor operator -(TorchTensor left, TorchScalar right)
        {
            return left.sub(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchTensor right)
        {
            return left.div(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchScalar right)
        {
            return left.div(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchTensor right)
        {
            return left.remainder(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchScalar right)
        {
            return left.remainder(right);
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
    }

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

    public static class TensorExtensionMethods
    {
        internal static bool IsIntegral(this TorchTensor tensor)
        {
            switch(tensor.Type) {
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

        public static TorchTensor ToTorchTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requiresGrad = false)
        {
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.from(array as byte[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(sbyte):
                    {
                        return Int8Tensor.from(array as sbyte[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return Int16Tensor.from(array as short[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return Int32Tensor.from(array as int[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return Int64Tensor.from(array as long[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return Float64Tensor.from(array as double[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return Float32Tensor.from(array as float[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(bool):
                    {
                        return BoolTensor.from(array as bool[], dimensions, requiresGrad);
                    }
                //case bool _ when typeof(T) == typeof(System.Numerics.Complex):
                //    {
                //        return ComplexFloat64Tensor.from(array as System.Numerics.Complex[], dimensions, requiresGrad);
                //    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static TorchTensor ToTorchTensor<T>(this T scalar, Device? device = null, bool requiresGrad = false) where T : struct
        {
            if (requiresGrad && typeof(T) != typeof(float) && typeof(T) != typeof(double))
            {
                throw new ArgumentException(nameof(requiresGrad), "Only floating point types support gradients.");
            }

            if (typeof(T) == typeof(byte))
                return ByteTensor.from((byte)(object)scalar, device, requiresGrad);
            if (typeof(T) == typeof(sbyte))
                return Int8Tensor.from((sbyte)(object)scalar, device, requiresGrad);
            if (typeof(T) == typeof(short))
                return Int16Tensor.from((short)(object)scalar, device, requiresGrad);
            if (typeof(T) == typeof(int))
                return Int32Tensor.from((int)(object)scalar, device, requiresGrad);
            if (typeof(T) == typeof(long))
                return Int64Tensor.from((long)(object)scalar, device, requiresGrad);
            if (typeof(T) == typeof(double))
                return Float64Tensor.from((double)(object)scalar, device, requiresGrad);
            if (typeof(T) == typeof(float))
                return Float32Tensor.from((float)(object)scalar, device, requiresGrad);
            throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_cat(IntPtr tensor, int len, long dim);
        public static TorchTensor cat(this TorchTensor[] tensors, long dimension)
        {
            if (tensors.Length == 0) {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Length == 1) {
                return tensors[0];
            }

            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                return new TorchTensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr tensor, int len, long dim);

        public static TorchTensor stack(this TorchTensor[] tensors, long dimension)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_stack(tensorsRef, parray.Array.Length, dimension);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        public static float ToSingle(this TorchTensor value) => value.ToScalar().ToSingle();
        public static double ToDouble(this TorchTensor value) => value.ToScalar().ToDouble();
        public static sbyte ToSByte(this TorchTensor value) => value.ToScalar().ToSByte();
        public static byte ToByte(this TorchTensor value) => value.ToScalar().ToByte();
        public static short ToInt16(this TorchTensor value) => value.ToScalar().ToInt16();
        public static int ToInt32(this TorchTensor value) => value.ToScalar().ToInt32();
        public static long ToInt64(this TorchTensor value) => value.ToScalar().ToInt64();
        public static bool ToBoolean(this TorchTensor value) => value.ToScalar().ToBoolean();
    }
}