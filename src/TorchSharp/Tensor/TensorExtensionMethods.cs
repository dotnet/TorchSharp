// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.Utils.LEB128Codec;

#nullable enable
namespace TorchSharp
{
    using static torch;

    /// <summary>
    /// A few extensions to the Tensor type.
    /// </summary>
    public static class TensorExtensionMethods
    {
        internal static bool IsIntegral(this Tensor tensor)
        {
            return IsIntegral(tensor.dtype);
        }

        internal static bool IsIntegral(this ScalarType type)
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

        internal static bool IsFloatingPoint(this ScalarType type)
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

        /// <summary>
        /// Save the tensor in a .NET-specific format.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="writer"></param>
        public static void Save(this Tensor tensor, System.IO.BinaryWriter writer)
        {
            // First, write the type
            writer.Encode((int)tensor.dtype); // 4 bytes
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
        public static void Load(this Tensor tensor, System.IO.BinaryReader reader)
        {
            // First, read the type
            var type = (ScalarType)reader.Decode();

            if (type != tensor.dtype)
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
        public static Tensor ToTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requiresGrad = false)
        {
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            switch (true) {
            case bool _ when typeof(T) == typeof(byte): {
                    return torch.tensor(array as byte[], dimensions, requiresGrad: requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(sbyte): {
                    return torch.tensor(array as sbyte[], dimensions, requiresGrad: requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(short): {
                    return torch.tensor(array as short[], dimensions, requiresGrad: requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(int): {
                    return torch.tensor(array as int[], dimensions, requiresGrad: requiresGrad);
                }
            case bool _ when typeof(T) == typeof(long): {
                    return torch.tensor(array as long[], dimensions, requiresGrad: requiresGrad);
                }
            case bool _ when typeof(T) == typeof(double): {
                    return torch.tensor(array as double[], dimensions, requiresGrad: requiresGrad);
                }
            case bool _ when typeof(T) == typeof(float): {
                    return torch.tensor(array as float[], dimensions, requiresGrad: requiresGrad);
                }
            case bool _ when typeof(T) == typeof(bool): {
                    return torch.tensor(array as bool[], dimensions, requiresGrad: requiresGrad);
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
        public static Tensor ToTensor<T>(this T scalar, torch.Device? device = null, bool requiresGrad = false) where T : struct
        {
            if (requiresGrad && typeof(T) != typeof(float) && typeof(T) != typeof(double)) {
                throw new ArgumentException(nameof(requiresGrad), "Only floating point types support gradients.");
            }

            if (typeof(T) == typeof(byte))
                return torch.tensor((byte)(object)scalar, uint8, device, requiresGrad);
            if (typeof(T) == typeof(sbyte))
                return torch.tensor((sbyte)(object)scalar, int8, device, requiresGrad);
            if (typeof(T) == typeof(short))
                return torch.tensor((short)(object)scalar, int16, device, requiresGrad);
            if (typeof(T) == typeof(int))
                return torch.tensor((int)(object)scalar, int32, device, requiresGrad);
            if (typeof(T) == typeof(long))
                return torch.tensor((long)(object)scalar, int64, device, requiresGrad);
            if (typeof(T) == typeof(float))
                return torch.tensor((float)(object)scalar, float32, device, requiresGrad);
            if (typeof(T) == typeof(double))
                return torch.tensor((double)(object)scalar, float64, device, requiresGrad);
            throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
        }

        public static (float Real,float Imaginary) ToComplexFloat32(this Tensor value) => value.ToScalar().ToComplexFloat32();
        public static System.Numerics.Complex ToComplexFloat64(this Tensor value) => value.ToScalar().ToComplexFloat64();
        public static float ToSingle(this Tensor value) => value.ToScalar().ToSingle();
        public static double ToDouble(this Tensor value) => value.ToScalar().ToDouble();
        public static sbyte ToSByte(this Tensor value) => value.ToScalar().ToSByte();
        public static byte ToByte(this Tensor value) => value.ToScalar().ToByte();
        public static short ToInt16(this Tensor value) => value.ToScalar().ToInt16();
        public static int ToInt32(this Tensor value) => value.ToScalar().ToInt32();
        public static long ToInt64(this Tensor value) => value.ToScalar().ToInt64();
        public static bool ToBoolean(this Tensor value) => value.ToScalar().ToBoolean();

        public static (float Real, float Imaginary) ToComplex32(this Tensor value) => value.ToScalar().ToComplexFloat32();
        public static System.Numerics.Complex ToComplex64(this Tensor value) => value.ToScalar().ToComplexFloat64();

        /// <summary>
        /// Multiply the dimensions of a tensor shape to provide a complete size.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns></returns>
        public static long TotalSize(this IEnumerable<long> shape)
        {
            long result = 1;
            foreach (var sz in shape) {
                result *= sz;
            }
            return result;
        }

        // Vision-related operations

        public static Tensor crop(this Tensor image, int top, int left, int height, int width)
        {
            var dims = image.Dimensions;
            var hoffset = dims - 2;
            long h = image.shape[hoffset], w = image.shape[hoffset + 1];

            var right = left + width;
            var bottom = top + height;

            if (left < 0 || top < 0 || right > w || bottom > h) {

                var slice = image.index(TensorIndex.Ellipsis, TensorIndex.Slice(Math.Max(top, 0), bottom), TensorIndex.Slice(Math.Max(left, 0), right));

                var padding_ltrb = new long[] { Math.Max(-left, 0), Math.Max(-top, 0), Math.Max(right - w, 0), Math.Max(bottom - h, 0) };

                return TorchSharp.torch.nn.functional.pad(slice, padding_ltrb);
            }

            return image.index(TensorIndex.Ellipsis, TensorIndex.Slice(top, bottom), TensorIndex.Slice(left, right));
        }
    }
}
