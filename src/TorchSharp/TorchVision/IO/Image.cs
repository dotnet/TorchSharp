// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class io
        {
            /// <summary>
            /// <cref>Imager</cref> to be used when a <cref>torchvision.io</cref> image method's <c>imager</c> is unspecified.
            /// </summary>
            public static Imager DefaultImager { get; set; } = new NotImplementedImager();

            /// <summary>
            /// Abstract class providing a generic way to decode and encode images as <cref>Tensor</cref>s.
            /// Used by <cref>torchvision.io</cref> image methods.
            /// </summary>
            public abstract class Imager
            {
                /// <summary>
                /// Decode the contents of an image file and returns the result as a <cref>Tensor</cref>.
                /// </summary>
                /// <param name="stream">Stream to read from.</param>
                /// <param name="mode">Image read mode.</param>
                /// <remarks>
                /// The image format is detected from image file contents.
                /// </remarks>
                /// <returns>
                /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
                /// </returns>
                public abstract Tensor DecodeImage(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED);

                /// <summary>
                /// Decode the contents of a byte array and returns the result as a <cref>Tensor</cref>.
                /// </summary>
                /// <param name="data">byte array to decode from.</param>
                /// <param name="mode">Image read mode.</param>
                /// <remarks>
                /// The image format is detected from image file contents.
                /// </remarks>
                /// <returns>
                /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
                /// </returns>
                public abstract Tensor DecodeImage(byte[] data, ImageReadMode mode = ImageReadMode.UNCHANGED);

                /// <summary>
                /// Asynchronously decode the contents of an image file and returns the result as a <cref>Tensor</cref>.
                /// </summary>
                /// <param name="stream">Stream to read from.</param>
                /// <param name="mode">Image read mode.</param>
                /// <param name="cancellationToken">Cancellation token.</param>
                /// <remarks>
                /// The image format is detected from image file contents.
                /// </remarks>
                /// <returns>
                /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
                /// </returns>
                public virtual Task<Tensor> DecodeImageAsync(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED, CancellationToken cancellationToken = default)
                {
                    // Default implementation -- just do things synchronously.
                    var t = DecodeImage(stream, mode);
                    return Task<Tensor>.FromResult(t);
                }

                /// <summary>
                /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a stream.
                /// </summary>
                /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
                /// <param name="format">Image format.</param>
                /// <param name="stream">Stream to write to.</param>
                public abstract void EncodeImage(Tensor image, ImageFormat format, Stream stream);

                /// <summary>
                /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a stream.
                /// </summary>
                /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
                /// <param name="format">Image format.</param>
                /// <returns>The encoded image as a byte array.</returns>
                public abstract byte[] EncodeImage(Tensor image, ImageFormat format);

                /// <summary>
                /// Asynchronously encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a stream.
                /// </summary>
                /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
                /// <param name="format">Image format.</param>
                /// <param name="stream">Stream to write to.</param>
                /// <param name="cancellationToken">Cancellation token.</param>
                public virtual Task EncodeImageAsync(Tensor image, ImageFormat format, Stream stream, CancellationToken cancellationToken = default)
                {
                    // Default implementation -- just do things synchronously.
                    EncodeImage(image, format, stream);
                    return Task.CompletedTask;
                }
            }

            /// <summary>
            /// Support for various modes while reading images. Affects the returned <cref>Tensor</cref>'s <c>color_channels</c>.
            /// </summary>
            public enum ImageReadMode
            {
                /// <summary>
                /// Read as is. Returned <cref>Tensor</cref>'s color_channels depend on the <cref>ImageFormat</cref>.
                /// </summary>
                UNCHANGED,
                /// <summary>
                /// Read as grayscale. Return <cref>Tensor</cref> with <c>color_channels = 1 </c>.
                /// </summary>
                GRAY,
                /// <summary>
                /// Read as grayscale with transparency. Return <cref>Tensor</cref> with <c>color_channels = 2 </c>.
                /// </summary>
                GRAY_ALPHA,
                /// <summary>
                /// Read as RGB. Return <cref>Tensor</cref> with <c>color_channels = 3 </c>.
                /// </summary>
                RGB,
                /// <summary>
                /// Read as RGB with transparency. Return <cref>Tensor</cref> with <c>color_channels = 4 </c>.
                /// </summary>
                RGB_ALPHA
            }

            /// <summary>
            /// Reads an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="filename">Path to the image.</param>
            /// <param name="mode">Image read mode.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            /// <returns>
            /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
            /// </returns>
            public static Tensor read_image(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
            {
                using (FileStream stream = File.Open(filename, FileMode.Open))
                    return (imager ?? DefaultImager).DecodeImage(stream, mode);
            }

            /// <summary>
            /// Reads an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="stream">Stream to read from.</param>
            /// <param name="mode">Image read mode.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            /// <returns>
            /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
            /// </returns>
            public static Tensor read_image(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
            {
                return (imager ?? DefaultImager).DecodeImage(stream, mode);
            }

            /// <summary>
            /// Asynchronously reads an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="filename">Path to the image.</param>
            /// <param name="mode">Read mode.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            /// <returns>
            /// A task that represents the asynchronous read operation.
            /// The value of the TResult parameter is a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
            /// </returns>
            public static async Task<Tensor> read_image_async(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
            {

                using (FileStream stream = File.Open(filename, FileMode.Open))
                    return await (imager ?? DefaultImager).DecodeImageAsync(stream, mode);
            }

            /// <summary>
            /// Asynchronously reads an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="stream">Stream to read from.</param>
            /// <param name="mode">Read mode.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            /// <returns>
            /// A task that represents the asynchronous read operation.
            /// The value of the TResult parameter is a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
            /// </returns>
            public static async Task<Tensor> read_image_async(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
            {
                return await (imager ?? DefaultImager).DecodeImageAsync(stream, mode);
            }

            /// <summary>
            /// Write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="filename">Path to the file.</param>
            /// <param name="format">Image format.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static void write_image(Tensor image, string filename, ImageFormat format, Imager imager = null)
            {
                if (image.dtype != ScalarType.Byte) throw new System.ArgumentException("Image tensors must be 'byte' tensors when passed to 'write_image()'.");
                using (FileStream stream = File.Open(filename, FileMode.OpenOrCreate))
                    (imager ?? DefaultImager).EncodeImage(image, format, stream);
            }

            /// <summary>
            /// Write a PNG image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="filename">Path to the file.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static void write_png(Tensor image, string filename, Imager imager = null)
            {
                write_image(image, filename, ImageFormat.Png, imager);
            }

            /// <summary>
            /// Write a JPEG image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="filename">Path to the file.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static void write_jpeg(Tensor image, string filename, Imager imager = null)
            {
                write_image(image, filename, ImageFormat.Jpeg, imager);
            }

            /// <summary>
            /// Write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="stream">Stream to write to.</param>
            /// <param name="format">Image format.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static void write_image(Tensor image, Stream stream, ImageFormat format, Imager imager = null)
            {
                if (image.dtype != ScalarType.Byte) throw new System.ArgumentException("Image tensors must be 'byte' tensors when passed to 'write_image()'.");
                (imager ?? DefaultImager).EncodeImage(image, format, stream);
            }

            /// <summary>
            /// Write a PNG image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="stream">Stream to write to.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static void write_png(Tensor image, Stream stream, Imager imager = null)
            {
                write_image(image, stream, ImageFormat.Png, imager);
            }

            /// <summary>
            /// Write a JPEG image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="stream">Stream to write to.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static void write_jpeg(Tensor image, Stream stream, Imager imager = null)
            {
                write_image(image, stream, ImageFormat.Jpeg, imager);
            }

            /// <summary>
            /// Asynchronously write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="filename">Path to the file.</param>
            /// <param name="format">Image format.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static async Task write_image_async(Tensor image, string filename, ImageFormat format, Imager imager = null)
            {
                if (image.dtype != ScalarType.Byte) throw new System.ArgumentException("Image tensors must be 'byte' tensors when passed to 'write_image()'.");
                using (FileStream stream = File.Open(filename, FileMode.OpenOrCreate))
                    await (imager ?? DefaultImager).EncodeImageAsync(image, format, stream);
            }

            /// <summary>
            /// Asynchronously write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="stream">Stream to write to.</param>
            /// <param name="format">Image format.</param>
            /// <param name="cancellationToken">Cancellation token</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            public static async Task write_image_async(Tensor image, Stream stream, ImageFormat format, CancellationToken cancellationToken = default, Imager imager = null)
            {
                if (image.dtype != ScalarType.Byte) throw new System.ArgumentException("Image tensors must be 'byte' tensors when passed to 'write_image()'.");
                await (imager ?? DefaultImager).EncodeImageAsync(image, format, stream);
            }

            /// <summary>
            /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>
            /// into a image <cref>Tensor</cref> buffer.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="format">Image format.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            /// <returns>A one dimensional <c>uint8</c> <cref>Tensor</cref> that contains the raw bytes of <c>image</c> encoded in the provided format.</returns>
            public static Tensor encode_image(Tensor image, ImageFormat format, Imager imager = null)
            {
                if (image.dtype != ScalarType.Byte) throw new System.ArgumentException("Image tensors must be 'byte' tensors when passed to 'encode_image()'.");
                using (var stream = new MemoryStream()) {
                    (imager ?? DefaultImager).EncodeImage(image, format, stream);
                    return stream.ToArray();
                }
            }

            /// <summary>
            /// Decodes an image <cref>Tensor</cref> buffer into a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
            /// </summary>
            /// <param name="image">A one dimensional <c>uint8</c> <cref>Tensor</cref> that contains the raw bytes of an image.</param>
            /// <param name="mode">Decode mode.</param>
            /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
            /// <returns><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</returns>
            public static Tensor decode_image(Tensor image, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
            {
                if (image.dtype != ScalarType.Byte) throw new System.ArgumentException("Image tensors must be 'byte' tensors when passed to 'decode_image()'.");
                using (var stream = new MemoryStream()) {
                    (imager ?? DefaultImager).DecodeImage(stream, mode);
                    return stream.ToArray();
                }
            }
        }
    }
}
