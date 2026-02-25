// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using static TorchSharp.Utils.LEB128Codec;

namespace TorchSharp
{
    using static Google.Protobuf.Reflection.SourceCodeInfo.Types;
    using static torch;

    public enum TensorStringStyle
    {
        Metadata,   // Print only shape, dtype, and device
        Julia,      // Print tensor in the style of Julia
        Numpy,      // Print tensor in the style of NumPy
        CSharp,     // Print tensor in the style of CSharp
        Default,    // Pick up the style to use from torch.TensorStringStyle
    }

    public static partial class torch
    {
        /// <summary>
        /// The default string formatting style used by ToString(), print(), and str()
        /// </summary>
        public static TensorStringStyle TensorStringStyle {
            get {
                return _style;
            }
            set {

                if (value == TensorStringStyle.Default)
                    throw new ArgumentException("The style cannot be set to 'Default'");
                _style = value;
            }
        }

        /// <summary>
        /// Set options for printing.
        /// </summary>
        /// <param name="precision">Number of digits of precision for floating point output.</param>
        /// <param name="linewidth">The number of characters per line for the purpose of inserting line breaks (default = 100).</param>
        /// <param name="newLine">The string to use to represent new-lines. Starts out as 'Environment.NewLine'</param>
        /// <param name="sci_mode">Enable scientific notation.</param>
        /// <param name="maxRows">The maximum number of rows prrinted.</param>
        /// <param name="maxColumns">The maximum number of columns prrinted.</param>
        public static void set_printoptions(
            int precision,
            int? linewidth = null,
            string? newLine = null,
            bool sci_mode = false,
            int? maxRows = null,
            int? maxColumns = null)
        {
            torch.floatFormat = sci_mode ? $"E{precision}" : $"F{precision}";
            if (newLine is not null)
                torch.newLine = newLine;
            if (linewidth.HasValue)
                torch.lineWidth = linewidth.Value;
            if (maxRows.HasValue)
                torch.maxRows = maxRows.Value;
            if (maxColumns.HasValue)
                torch.maxColumns = maxColumns.Value;
        }

        /// <summary>
        /// Set options for printing.
        /// </summary>
        /// <param name="style">The default string formatting style used by ToString(), print(), and str()</param>
        /// <param name="floatFormat">
        /// The format string to use for floating point values.
        /// See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings
        /// </param>
        /// <param name="linewidth">The number of characters per line for the purpose of inserting line breaks (default = 100).</param>
        /// <param name="newLine">The string to use to represent new-lines. Starts out as 'Environment.NewLine'</param>
        /// <param name="maxRows">The maximum number of rows prrinted.</param>
        /// <param name="maxColumns">The maximum number of columns prrinted.</param>
        public static void set_printoptions(
            TensorStringStyle? style = null,
            string? floatFormat = null,
            int? linewidth = null,
            string? newLine = null,
            int? maxRows = null,
            int? maxColumns = null)
        {
            if (style.HasValue)
                torch._style = style.Value;
            if (floatFormat is not null)
                torch.floatFormat = floatFormat;
            if (newLine is not null)
                torch.newLine = newLine;
            if (linewidth.HasValue)
                torch.lineWidth = linewidth.Value;
            if (maxRows.HasValue)
                torch.maxRows = maxRows.Value;
            if (maxColumns.HasValue)
                torch.maxColumns = maxColumns.Value;
        }

        public const TensorStringStyle julia = TensorStringStyle.Julia;
        public const TensorStringStyle numpy = TensorStringStyle.Numpy;
        public const TensorStringStyle csharp = TensorStringStyle.CSharp;

        private static TensorStringStyle _style = TensorStringStyle.Julia;

        internal static string floatFormat = "g5";
        internal static string newLine = Environment.NewLine;
        internal static int lineWidth = 100;
        internal static int maxRows = 6;
        internal static int maxColumns = 6;
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
        /// <param name="fltFormat">
        /// The format string to use for floating point values.
        /// See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings
        /// </param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
        /// <param name="style">
        /// The style to use -- either 'default,' 'metadata,' 'julia,' or 'numpy'
        /// </param>
        /// <remarks>
        /// This method does exactly the same as ToString(bool, string, int), but is shorter,
        /// looks more like Python 'str' and doesn't require a style argument in order
        /// to discriminate.
        ///
        /// Primarily intended for use in interactive notebooks.
        /// </remarks>
        public static string str(this Tensor tensor, string? fltFormat = null, int? width = null, string? newLine = null, CultureInfo? cultureInfo = null, TensorStringStyle style = TensorStringStyle.Default)
        {
            return tensor.ToString(style, fltFormat, width, cultureInfo, newLine);
        }

        /// <summary>
        /// Get a Julia-style string representation of the tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="fltFormat">
        /// The format string to use for floating point values.
        /// See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings
        /// </param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
        /// <returns></returns>
        /// <remarks>
        /// This method does exactly the same as str(TensorStringStyle.Julia, ...) but is shorter,
        /// looks more like Python 'str' and doesn't require a style argument in order
        /// to discriminate.
        ///
        /// Primarily intended for use in interactive notebooks.
        /// </remarks>
        public static string jlstr(this Tensor tensor, string? fltFormat = null, int? width = null, string? newLine = null, CultureInfo? cultureInfo = null)
        {
            return tensor.ToString(TensorStringStyle.Julia, fltFormat, width, cultureInfo, newLine);
        }

        /// <summary>
        /// Get a metadata string representation of the tensor.
        /// Creating metadata string will ignore fltFormat, etc. so this method will not accept them as parameter.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns></returns>
        /// <remarks>
        /// This method does exactly the same as str(TensorStringStyle.Metadata, ...) but is shorter,
        /// looks more like Python 'str' and doesn't require a style argument in order
        /// to discriminate.
        ///
        /// Primarily intended for use in interactive notebooks.
        /// </remarks>
        public static string metastr(this Tensor tensor)
        {
            return tensor.ToString(TensorStringStyle.Metadata);
        }

        /// <summary>
        /// Get a numpy-style string representation of the tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="fltFormat">
        /// The format string to use for floating point values.
        /// See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings
        /// </param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
        /// <returns></returns>
        /// <remarks>
        /// This method does exactly the same as str(TensorStringStyle.Numpy, ...) but is shorter,
        /// looks more like Python 'str()' and doesn't require a style argument in order
        /// to discriminate.
        ///
        /// Primarily intended for use in interactive notebooks.
        /// </remarks>
        public static string npstr(this Tensor tensor, string? fltFormat = "g5", int? width = 100, string? newLine = null, CultureInfo? cultureInfo = null)
        {
            return tensor.ToString(TensorStringStyle.Numpy, fltFormat, width, cultureInfo, newLine);
        }

        /// <summary>
        /// Get a C#-style string representation of the tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="fltFormat">
        /// The format string to use for floating point values.
        /// See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings
        /// </param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
        /// <returns></returns>
        /// <remarks>
        /// This method does exactly the same as str(TensorStringStyle.CSharp, ...) but is shorter,
        /// looks more like Python 'str()' and doesn't require a style argument in order
        /// to discriminate.
        ///
        /// Primarily intended for use in interactive notebooks.
        /// </remarks>
        public static string cstr(this Tensor tensor, string? fltFormat = "g5", int? width = 100, string? newLine = null, CultureInfo? cultureInfo = null)
        {
            return tensor.ToString(TensorStringStyle.CSharp, fltFormat, width, cultureInfo, newLine);
        }

        /// <summary>
        /// Uses Console.WriteLine to print a tensor expression on stdout. This is intended for
        /// interactive notebook use, primarily.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="fltFormat">
        /// The format string to use for floating point values.
        /// See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings
        /// </param>
        /// <param name="width">The width of each line of the output string.</param>
        /// <param name="newLine">The newline string to use, defaults to system default.</param>
        /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
        /// <param name="style">
        /// The style to use -- either 'default,' 'metadata,' 'julia,' or 'numpy'
        /// </param>
        /// <returns></returns>
        public static Tensor print(this Tensor t, string? fltFormat = "g5", int? width = 100, string? newLine = null, CultureInfo? cultureInfo = null, TensorStringStyle style = TensorStringStyle.Default)
        {
            Console.WriteLine(t.ToString(style, fltFormat, width, cultureInfo, newLine));
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

        public static ReadOnlySpan<int> IntShape(this Tensor tensor)
        {
            var shape = tensor.shape;
            var int_shape = new int[shape.Length];
            for (var i = 0; i < shape.Length; ++i) int_shape[i] = (int)shape[i];
            return int_shape;
        }

        /// <summary>
        /// Get the size of each element of this ScalarType.
        /// </summary>
        /// <param name="type">The input type.</param>
        /// <returns>The element size</returns>
        /// <exception cref="NotImplementedException">Invalid ScalarType</exception>
        public static int ElementSize(this ScalarType type)
        {
            return type switch {
                ScalarType.Byte => 1,
                ScalarType.Int8 => 1,
                ScalarType.Int16 => 2,
                ScalarType.Int32 => 4,
                ScalarType.Int64 => 8,
                ScalarType.Float16 => 2,
                ScalarType.Float32 => 4,
                ScalarType.Float64 => 8,
                ScalarType.ComplexFloat32 => 8,
                ScalarType.ComplexFloat64 => 16,
                ScalarType.Bool => 1,
                ScalarType.QInt8 => 1,
                ScalarType.QUInt8 => 1,
                ScalarType.QInt32 => 4,
                ScalarType.BFloat16 => 2,
                _ => throw new NotImplementedException()
            };
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
        /// Indicates whether a given element type is quantized.
        /// </summary>
        /// <param name="type">The input type.</param>
        /// <returns></returns>
        internal static bool IsQuantized(this ScalarType type)
        {
            switch (type) {
            case ScalarType.QInt8:
            case ScalarType.QUInt8:
            case ScalarType.QInt32:
                return true;
            default:
                return false;
            }
        }

        /// <summary>
        /// Save the tensor in a .NET-specific format.
        /// </summary>
        /// <param name="tensor">The tensor to save</param>
        /// <param name="writer">A BinaryWriter instance.</param>
        public static void Save(this Tensor tensor, System.IO.BinaryWriter writer)
        {
            bool copied = false;

            if (tensor.device_type != DeviceType.CPU) {
                tensor = tensor.to(CPU);
                copied = true;
            }

            // First, write the type
            writer.Encode((int)tensor.dtype); // 4 bytes
                                              // Then, the shape.
            writer.Encode(tensor.shape.Length); // 4 bytes
            foreach (var s in tensor.shape) writer.Encode(s); // n * 8 bytes

            // Then, the data
            tensor.WriteBytesToStream(writer.BaseStream);

            if (copied) tensor.Dispose();
        }

        /// <summary>
        /// Save the tensor in a .NET-specific format.
        /// </summary>
        /// <param name="tensor">The tensor to save</param>
        /// <param name="stream">A stream opened for writing binary data.</param>
        public static void Save(this Tensor tensor, System.IO.Stream stream)
        {
            using var writer = new System.IO.BinaryWriter(stream);
            tensor.Save(writer);
            stream.Flush();
        }

        /// <summary>
        /// Save the tensor in a .NET-specific format.
        /// </summary>
        /// <param name="tensor">The tensor to save</param>
        /// <param name="location">A file path.</param>
        public static void Save(this Tensor tensor, string location)
        {
            tensor.Save(System.IO.File.Create(location));
        }

        /// <summary>
        /// Load the tensor using a .NET-specific format.
        /// </summary>
        /// <param name="tensor">The tensor into which to load serialized data.</param>
        /// <param name="reader">A BinaryReader instance</param>
        /// <param name="skip">If true, the data will be read from the stream, but not copied to the target tensor.</param>
        /// <remarks>
        /// Using a skip list only prevents tensors in the target module from being modified, it
        /// does not alter the logic related to checking for matching tensor element types.
        /// </remarks>
        public static void Load(this Tensor tensor, System.IO.BinaryReader reader, bool skip = false)
        {
            // First, read the type
            var type = (ScalarType)reader.Decode();

            if (type != tensor.dtype)
                throw new ArgumentException("Mismatched tensor data types while loading. Make sure that the model you are loading into is exactly the same as the origin.");

            // Then, the shape
            var shLen = reader.Decode();
            long[] loadedShape = new long[shLen];
            for (int i = 0; i < shLen; ++i) {
                loadedShape[i] = reader.Decode();
            }

            if (!skip && !loadedShape.SequenceEqual(tensor.shape))
                // We only care about this if the bytes will be written to the tensor.
                throw new ArgumentException("Mismatched tensor shape while loading. Make sure that the model you are loading into is exactly the same as the origin.");

            if (!skip) {
                using var d = torch.no_grad(); // so that we can perform operations on leaf tensors

                var device = tensor.device;
                if (device.type != DeviceType.CPU) {
                    using var temp = torch.zeros_like(tensor, device: torch.CPU);
                    temp.ReadBytesFromStream(reader.BaseStream);

                    tensor.copy_(temp);
                } else {
                    tensor.ReadBytesFromStream(reader.BaseStream);
                }                
            } else {
                reader.BaseStream.Seek(tensor.NumberOfElements * tensor.ElementSize, System.IO.SeekOrigin.Current);
            }
        }

        /// <summary>
        /// Load the tensor using a .NET-specific format.
        /// </summary>
        /// <param name="reader">A BinaryReader instance</param>
        /// <param name="skip">If true, the data will be read from the stream, but not stored in the return tensor.</param>
        /// <remarks>
        /// Using skip returns an empty tensor with the dimensions of the tensor that was loaded.
        /// </remarks>
        public static Tensor Load(System.IO.BinaryReader reader, bool skip = false)
        {
            Tensor? t = null;
            Load(ref t, reader, skip);
            return t!;
        }

        /// <summary>
        /// Load the tensor using a .NET-specific format.
        /// </summary>
        /// <param name="tensor">The tensor into which to load serialized data. If null, will create a new tensor.</param>
        /// <param name="reader">A BinaryReader instance</param>
        /// <param name="skip">If true, the data will be read from the stream, but not stored in the return tensor.</param>
        /// <remarks>
        /// Using skip returns an empty tensor with the dimensions of the tensor that was loaded.
        /// </remarks>
        /// <exception cref="NotImplementedException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public static void Load(ref Tensor? tensor, System.IO.BinaryReader reader, bool skip = false)
        {
            // First, read the type
            var type = (ScalarType)reader.Decode();

            // Then, the shape
            var shLen = reader.Decode();
            long[] loadedShape = new long[shLen];

            for (int i = 0; i < shLen; ++i) {
                loadedShape[i] = reader.Decode();
            }

            if (tensor is null) {
                // If the tensor doesn't exist, initialize by zeros unless
                // it's going to be loaded from the stream.
                tensor = skip
                    ? torch.zeros(loadedShape, dtype: type)
                    : torch.empty(loadedShape, dtype: type);
            } else if (!skip && !loadedShape.SequenceEqual(tensor.shape)) {
                // We only care about this if the bytes will be written to the tensor.
                throw new ArgumentException("Mismatched tensor shape while loading. Make sure that the model you are loading into is exactly the same as the origin.");
            }

            if (!skip) {
                using var d = torch.no_grad(); // so that we can perform operations on leaf tensors

                var device = tensor.device;
                if (device.type != DeviceType.CPU) {
                    using var temp = torch.zeros_like(tensor, device: torch.CPU);
                    temp.ReadBytesFromStream(reader.BaseStream);
                    tensor.copy_(temp);
                } else {
                    tensor.ReadBytesFromStream(reader.BaseStream);
                }
            } else {
                reader.BaseStream.Seek(tensor.NumberOfElements * tensor.ElementSize, System.IO.SeekOrigin.Current);
            }
        }

        /// <summary>
        /// Creating a tensor form an array of data.
        /// </summary>
        /// <typeparam name="T">The .NET element type.</typeparam>
        /// <param name="rawArray">Input data.</param>
        /// <param name="dimensions">The shape of the tensor that is created.</param>
        /// <param name="doCopy">Perform a copy rather than using the array as backing storage for the tensor.</param>
        /// <param name="requires_grad">If true, the tensor must track its gradients.</param>
        /// <returns></returns>
        public static Tensor ToTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requires_grad = false)
        {
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            return true switch {
                true when typeof(T) == typeof(byte) => tensor((array as byte[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(sbyte) => tensor((array as sbyte[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(short) => tensor((array as short[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(int) => tensor((array as int[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(long) => tensor((array as long[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(double) => tensor((array as double[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(float) => tensor((array as float[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(bool) => tensor((array as bool[])!, dimensions,
                    requires_grad: requires_grad),
                true when typeof(T) == typeof(BFloat16) => tensor((array as BFloat16[])!, dimensions,
                    requires_grad: requires_grad),
                _ => throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.")
            };
        }

        /// <summary>
        /// Creating a tensor from a scalar value.
        /// </summary>
        /// <typeparam name="T">The .NET element type.</typeparam>
        /// <param name="scalar">Scalar input value</param>
        /// <param name="device">The device to place the tensor on.</param>
        /// <param name="requires_grad">If true, the tensor must track its gradients.</param>
        /// <returns></returns>
        public static Tensor ToTensor<T>(this T scalar, Device? device = null, bool requires_grad = false) where T : struct
        {
            if (requires_grad && typeof(T) != typeof(float) && typeof(T) != typeof(double)
                && typeof(T) != typeof(BFloat16)) {
                throw new ArgumentException(nameof(requires_grad), "Only floating point types support gradients.");
            }

            if (typeof(T) == typeof(byte))
                return tensor((byte)(object)scalar, uint8, device, requires_grad);
            if (typeof(T) == typeof(sbyte))
                return tensor((sbyte)(object)scalar, int8, device, requires_grad);
            if (typeof(T) == typeof(short))
                return tensor((short)(object)scalar, int16, device, requires_grad);
            if (typeof(T) == typeof(int))
                return tensor((int)(object)scalar, int32, device, requires_grad);
            if (typeof(T) == typeof(long))
                return tensor((long)(object)scalar, int64, device, requires_grad);
            if (typeof(T) == typeof(float))
                return tensor((float)(object)scalar, float32, device, requires_grad);
            if (typeof(T) == typeof(double))
                return tensor((double)(object)scalar, float64, device, requires_grad);
            if (typeof(T) == typeof(BFloat16))
                return tensor(new BFloat16[] { (BFloat16)(object)scalar }, bfloat16, device, requires_grad);
            throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
        }

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static (float Real, float Imaginary) ToComplexFloat32(this Tensor value) => value.ToScalar().ToComplexFloat32();

        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static System.Numerics.Complex ToComplexFloat64(this Tensor value) => value.ToScalar().ToComplexFloat64();

#if NET6_0_OR_GREATER
        /// <summary>
        /// Explicitly convert a singleton tensor to a .NET scalar value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static Half ToHalf(this Tensor value) => value.ToScalar().ToHalf();
#endif

        /// <summary>
        /// Explicitly convert a singleton tensor to a BFloat16 value.
        /// </summary>
        /// <param name="value">The input tensor</param>
        public static BFloat16 ToBFloat16(this Tensor value) => value.ToScalar().ToBFloat16();

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

        /// <summary>
        /// Checks if calling `Tensor.to()` with the parameters will result in actually copying the tensor.
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <param name="device">The device to move to</param>
        /// <returns>True if the tensor will be copied</returns>
        internal static bool toWillCopy(this Tensor tensor, Device device)
        {
            return tensor.toWillCopy(device.type, device.index);
        }

        /// <summary>
        /// Checks if calling `Tensor.to()` with the parameters will result in actually copying the tensor.
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <param name="deviceType">The device type to move to</param>
        /// <param name="deviceIndex">The device index to move to</param>
        /// <returns>True if the tensor will be copied</returns>
        internal static bool toWillCopy(this Tensor tensor, DeviceType deviceType, int deviceIndex)
        {
            return tensor.device_index != deviceIndex || tensor.device_type != deviceType;
        }

        /// <summary>
        /// Checks if calling `Tensor.to()` with the parameters will result in actually copying the tensor.
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <param name="dtype">The dtype to move to</param>
        /// <returns>True if the tensor will be copied</returns>
        internal static bool toWillCopy(this Tensor tensor, ScalarType dtype)
        {
            return tensor.dtype != dtype;
        }

        /// <summary>
        /// Checks if calling `Tensor.to()` with the parameters will result in actually copying the tensor.
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <param name="dtype">The dtype to move to</param>
        /// <param name="device">The device to move to</param>
        /// <returns>True if the tensor will be copied</returns>
        internal static bool toWillCopy(this Tensor tensor, ScalarType dtype, Device device)
        {
            return tensor.toWillCopy(dtype, device.type, device.index);
        }

        /// <summary>
        /// Checks if calling `Tensor.to()` with the parameters will result in actually copying the tensor.
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <param name="dtype">The dtype to move to</param>
        /// <param name="deviceType">The device type to move to</param>
        /// <param name="deviceIndex">The device index to move to</param>
        /// <returns>True if the tensor will be copied</returns>
        internal static bool toWillCopy(this Tensor tensor, ScalarType dtype, DeviceType deviceType, int deviceIndex)
        {
            return tensor.device_index != deviceIndex || tensor.device_type != deviceType || tensor.dtype != dtype;
        }


    }
}
