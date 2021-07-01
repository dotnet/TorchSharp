// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.Utils.LEB128Codec;

#nullable enable
namespace TorchSharp.Tensor
{

    /// <summary>
    /// A few extensions to the TorchTensor type.
    /// </summary>
    public static class TensorExtensionMethods
    {
        internal static bool IsIntegral(this TorchTensor tensor)
        {
            return IsIntegral(tensor.Type);
        }

        internal static bool IsIntegral(ScalarType type)
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

        /// <summary>
        /// Save the tensor in a .NET-specific format.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="writer"></param>
        public static void Save(this TorchTensor tensor, System.IO.BinaryWriter writer)
        {
            // First, write the type
            writer.Encode((int)tensor.Type); // 4 bytes
            // Then, the shape.
            writer.Encode(tensor.shape.Length); // 4 bytes
            foreach (var s in tensor.shape) writer.Encode(s); // n * 8 bytes
            // Then, the data
            writer.Write(tensor.Bytes()); // ElementSize * NumberofElements
        }

        /// <summary>
        /// Load the tensor using a .NET-specific format.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="reader"></param>
        public static void Load(this TorchTensor tensor, System.IO.BinaryReader reader)
        {
            // First, read the type
            var type = (ScalarType)reader.Decode();

            if (type != tensor.Type)
                throw new ArgumentException("Mismatched tensor data types while loading.");

            // Then, the shape
            var shLen = reader.Decode();
            long[] loadedShape = new long[shLen];

            long totalSize = 1;
            for (int i = 0; i < shLen; ++i) {
                loadedShape[i] = reader.Decode();
                totalSize *= loadedShape[i];
            }

            if (!loadedShape.SequenceEqual(tensor.shape))
                throw new ArgumentException("Mismatched tensor shape while loading.");

            //
            // TODO: Fix this so that you can read large tensors. Right now, they are limited to 2GB
            //
            if (totalSize > int.MaxValue)
                throw new NotImplementedException("Loading tensors larger than 2GB");

            tensor.SetBytes(reader.ReadBytes((int)(totalSize * tensor.ElementSize)));
        }

        /// <summary>
        /// Creating a tensor form an array of data.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rawArray"></param>
        /// <param name="dimensions"></param>
        /// <param name="doCopy"></param>
        /// <param name="requiresGrad"></param>
        /// <returns></returns>
        public static TorchTensor ToTorchTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requiresGrad = false)
        {
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            switch (true) {
            case bool _ when typeof(T) == typeof(byte): {
                    return ByteTensor.from(array as byte[], dimensions, requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(sbyte): {
                    return Int8Tensor.from(array as sbyte[], dimensions, requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(short): {
                    return Int16Tensor.from(array as short[], dimensions, requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(int): {
                    return Int32Tensor.from(array as int[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(long): {
                    return Int64Tensor.from(array as long[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(double): {
                    return Float64Tensor.from(array as double[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(float): {
                    return Float32Tensor.from(array as float[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(bool): {
                    return BoolTensor.from(array as bool[], dimensions, requiresGrad);
                }
            //case bool _ when typeof(T) == typeof(System.Numerics.Complex):
            //    {
            //        return ComplexFloat64Tensor.from(array as System.Numerics.Complex[], dimensions, requiresGrad);
            //    }
            default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        /// <summary>
        /// Creating a tensor from a scalar value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="scalar"></param>
        /// <param name="device"></param>
        /// <param name="requiresGrad"></param>
        /// <returns></returns>
        public static TorchTensor ToTorchTensor<T>(this T scalar, Device? device = null, bool requiresGrad = false) where T : struct
        {
            if (requiresGrad && typeof(T) != typeof(float) && typeof(T) != typeof(double)) {
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

        /// <summary>
        /// Concatenates the given sequence of seq tensors in the given dimension.
        /// </summary>
        /// <param name="tensors"></param>
        /// <param name="dimension"></param>
        /// <returns></returns>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static TorchTensor cat(this IList<TorchTensor> tensors, long dimension)
        {
            if (tensors.Count == 0) {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Count == 1) {
                return tensors[0];
            }

            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                return new TorchTensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr tensor, int len, long dim);

        /// <summary>
        /// Concatenates a sequence of tensors along a new dimension.
        /// </summary>
        /// <returns></returns>
        /// <remarks>All tensors need to be of the same size.</remarks>
        public static TorchTensor stack(this IList<TorchTensor> tensors, long dimension)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_stack(tensorsRef, parray.Array.Length, dimension);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hstack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence horizontally (column wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static TorchTensor hstack(this IList<TorchTensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_hstack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_vstack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static TorchTensor vstack(this IList<TorchTensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_vstack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        /// <summary>
        /// Creates a new tensor by horizontally stacking the tensors in tensors.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="len"></param>
        /// <returns></returns>
        /// <remarks>Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1) column before being stacked horizontally.</remarks>
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_column_stack(IntPtr tensor, int len);

        public static TorchTensor column_stack(this IList<TorchTensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_column_stack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_row_stack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static TorchTensor row_stack(this IList<TorchTensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_row_stack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_dstack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence depthwise (along third axis).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        /// <remarks>This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by torch.atleast_3d().</remarks>
        public static TorchTensor dstack(this IList<TorchTensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_dstack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static double THSTensor_clip_grad_norm_(IntPtr tensor, int len, double max_norm, double norm_type);

        /// <summary>
        /// Clips gradient norm of an iterable of parameters.
        /// The norm is computed over all gradients together, as if they were concatenated into a single vector.
        /// Gradients are modified in-place.
        /// </summary>
        /// <param name="tensors"></param>
        /// <param name="max_norm"></param>
        /// <param name="norm_type"></param>
        /// <returns></returns>
        public static double clip_grad_norm(this IList<TorchTensor> tensors, double max_norm, double norm_type = 2.0)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                return THSTensor_clip_grad_norm_(tensorsRef, parray.Array.Length, max_norm, norm_type);
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

        public static (float Real, float Imaginary) ToComplex32(this TorchTensor value) => value.ToScalar().ToComplexFloat32();
        public static System.Numerics.Complex ToComplex64(this TorchTensor value) => value.ToScalar().ToComplexFloat64();

        // Vision-related operations

        public static TorchTensor crop(this TorchTensor image, int top, int left, int height, int width)
        {
            var dims = image.Dimensions;
            var hoffset = dims - 2;
            long h = image.shape[hoffset], w = image.shape[hoffset + 1];

            var right = left + width;
            var bottom = top + height;

            if (left < 0 || top < 0 || right > w || bottom > h) {

                var slice = image.index(TorchTensorIndex.Ellipsis, TorchTensorIndex.Slice(Math.Max(top, 0), bottom), TorchTensorIndex.Slice(Math.Max(left, 0), right));

                // Note: according to the documentation, it should be LTRB, but that generates the wrong result. Here, we use LRTB.
                var padding_ltrb = new long[] { Math.Max(-left, 0), Math.Max(right - w, 0), Math.Max(-top, 0), Math.Max(bottom - h, 0) };

                return nn.functional.Pad(slice, padding_ltrb);
            }

            return image.index(TorchTensorIndex.Ellipsis, TorchTensorIndex.Slice(top, bottom), TorchTensorIndex.Slice(left, right));
        }
    }
}
