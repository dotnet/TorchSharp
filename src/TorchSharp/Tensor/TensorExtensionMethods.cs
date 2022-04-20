// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.Utils.LEB128Codec;
using System.Runtime.CompilerServices;

#nullable enable
namespace TorchSharp
{
    using static torch;

    public enum TensorStringStyle
    {
        Metadata,
        Julia,
        Numpy
    }

    /// <summary>
    /// A few extensions to the Tensor type.
    /// </summary>
    public static class TensorExtensionMethods
    {
        /// <summary>
        /// Convert to a parameter.
        /// </summary>
        /// <param name="tensor">A tensor.</param>
        /// <returns></returns>
        public static Modules.Parameter AsParameter(this Tensor tensor)
        {
            return new Modules.Parameter(tensor);
        }

        /// <summary>
        /// Get a string representation of the tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="style">
        /// The style to use -- either 'metadata,' 'julia,' or 'numpy'
        /// </param>
        /// <param name="fltFormat">The format string to use for floating point values.</param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <returns></returns>
        /// <remarks>
        /// This method does exactly the same as ToString(bool, string, int), but is shorter,
        /// looks more like Python 'str' and doesn't require a style argument in order
        /// to discriminate.
        ///
        /// Primarily intended for use in interactive notebooks.
        /// </remarks>
        public static string str(this Tensor tensor, TensorStringStyle style = TensorStringStyle.Julia, string fltFormat = "g5", int width = 100, string newLine = "\n")
        {
            return tensor.ToString(style, fltFormat, width, newLine: newLine);
        }

        /// <summary>
        /// Uses Console.WriteLine to print a tensor expression on stdout. This is intended for
        /// interactive notebook use, primarily.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="style">
        /// The style to use -- either 'metadata,' 'julia,' or 'numpy'
        /// </param>
        /// <param name="fltFormat">The format string to use for floating point values.</param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <returns></returns>
        public static Tensor print(this Tensor t, TensorStringStyle style = TensorStringStyle.Julia, string fltFormat = "g5", int width = 100, string newLine = "\n")
        {
            Console.WriteLine(t.str(style, fltFormat, width, newLine: newLine));
            return t;
        }

        /// <summary>
        /// Indicates whether the element type of a given tensor is integral.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns></returns>
        internal static bool IsIntegral(this Tensor tensor)
        {
            return IsIntegral(tensor.dtype);
        }

        /// <summary>
        /// Indicates whether a given element type is integral.
        /// </summary>
        /// <param name="type">The input type.</param>
        /// <returns></returns>
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

        /// <summary>
        /// Indicates whether a given element type is real.
        /// </summary>
        /// <param name="type">The input type.</param>
        /// <returns></returns>
        /// <remarks>
        /// Complex numbers are not real, and thus will return 'false'
        /// </remarks>
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
        /// Indicates whether a given element type is complex.
        /// </summary>
        /// <param name="type">The input type.</param>
        /// <returns></returns>
        /// <remarks>
        /// Complex numbers are not real, and thus will return 'false'
        /// </remarks>
        internal static bool IsComplex(this ScalarType type)
        {
            switch (type) {
            case ScalarType.ComplexFloat32:
            case ScalarType.ComplexFloat64:
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
            bool copied = false;

            if (tensor.device_type != DeviceType.CPU) {
                tensor = tensor.to(torch.CPU);
                copied = true;
            }

            // First, write the type
            writer.Encode((int)tensor.dtype); // 4 bytes
                                                // Then, the shape.
            writer.Encode(tensor.shape.Length); // 4 bytes
            foreach (var s in tensor.shape) writer.Encode(s); // n * 8 bytes
                                                                // Then, the data
#if NETSTANDARD2_0_OR_GREATER
            // TODO: NETSTANDARD2_0_OR_GREATER Try to optimize to avoid the allocation
            writer.Write(tensor.bytes.ToArray()); // ElementSize * NumberOfElements
#else
            writer.Write(tensor.bytes); // ElementSize * NumberOfElements
#endif // NETSTANDARD2_0_OR_GREATER

            if (copied) tensor.Dispose();
        }

        /// <summary>
        /// Load the tensor using a .NET-specific format.
        /// </summary>
        /// <param name="tensor">The tensor into which to load serialized data.</param>
        /// <param name="reader">A BinaryReader instance</param>
        public static void Load(this Tensor tensor, System.IO.BinaryReader reader)
        {
            // First, read the type
            var type = (ScalarType)reader.Decode();

            if (type != tensor.dtype)
                throw new ArgumentException("Mismatched tensor data types while loading. Make sure that the model you are loading into is exactly the same as the origin.");

            // Then, the shape
            var shLen = reader.Decode();
            long[] loadedShape = new long[shLen];

            long totalSize = 1;
            for (int i = 0; i < shLen; ++i) {
                loadedShape[i] = reader.Decode();
                totalSize *= loadedShape[i];
            }

            if (!loadedShape.SequenceEqual(tensor.shape))
                throw new ArgumentException("Mismatched tensor shape while loading. Make sure that the model you are loading into is exactly the same as the origin.");

            //
            // TODO: Fix this so that you can read large tensors. Right now, they are limited to 2GB
            //
            if (totalSize > int.MaxValue)
                throw new NotImplementedException("Loading tensors larger than 2GB");

            tensor.bytes = reader.ReadBytes((int)(totalSize * tensor.ElementSize));
        }

        /// <summary>
        /// Creating a tensor form an array of data.
        /// </summary>
        /// <typeparam name="T">The .NET element type.</typeparam>
        /// <param name="rawArray">Input data.</param>
        /// <param name="dimensions">The shape of the tensor that is created.</param>
        /// <param name="doCopy">Perform a copy rather than using the array as backing storage for the tensor.</param>
        /// <param name="requiresGrad">If true, the tensor must track its gradients.</param>
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
        /// <typeparam name="T">The .NET element type.</typeparam>
        /// <param name="scalar">Scalar input value</param>
        /// <param name="device">The device to place the tensor on.</param>
        /// <param name="requiresGrad">If true, the tensor must track its gradients.</param>
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

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static (float Real,float Imaginary) ToComplexFloat32(this Tensor value) => value.ToScalar().ToComplexFloat32();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static System.Numerics.Complex ToComplexFloat64(this Tensor value) => value.ToScalar().ToComplexFloat64();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static float ToSingle(this Tensor value) => value.ToScalar().ToSingle();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static double ToDouble(this Tensor value) => value.ToScalar().ToDouble();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static sbyte ToSByte(this Tensor value) => value.ToScalar().ToSByte();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static byte ToByte(this Tensor value) => value.ToScalar().ToByte();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static short ToInt16(this Tensor value) => value.ToScalar().ToInt16();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static int ToInt32(this Tensor value) => value.ToScalar().ToInt32();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static long ToInt64(this Tensor value) => value.ToScalar().ToInt64();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static bool ToBoolean(this Tensor value) => value.ToScalar().ToBoolean();


        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static (float Real, float Imaginary) ToComplex32(this Tensor value) => value.ToScalar().ToComplexFloat32();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
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

        /// <summary>
        /// Crop the given image tensor at specified location and output size.
        /// If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an
        /// arbitrary number of leading dimensions.
        /// If image size is smaller than output size along any edge, image is padded with 0 and then cropped.
        /// </summary>
        /// <param name="image">The input tensor.</param>
        /// <param name="top">Vertical component of the top left corner of the crop box.</param>
        /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
        /// <param name="height">Height of the crop box.</param>
        /// <param name="width">Width of the crop box.</param>
        /// <returns></returns>
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
