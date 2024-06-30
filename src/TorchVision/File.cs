using System.IO;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        partial class io
        {
            /// <summary>
            /// Reads a file into a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="filename">Path to the file.</param>
            /// <returns>
            /// One dimensional <cref>Tensor</cref> containing the bytes of the file.
            /// </returns>
            public static Tensor read_file(string filename)
            {
                return File.ReadAllBytes(filename);
            }

            /// <summary>
            /// Asynchronously reads a file into a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="filename">Path to the file.</param>
            /// <returns>
            /// A task that represents the asynchronous read operation.
            /// The value of the TResult parameter is a one dimensional <cref>Tensor</cref> containing the bytes of the file.
            /// </returns>
            public static async Task<Tensor> read_file_async(string filename)
            {
                byte[] data;

                using (FileStream stream = File.Open(filename, FileMode.Open)) {
                    data = new byte[stream.Length];
                    await stream.ReadAsync(data, 0, data.Length);
                }

                return data;
            }

            /// <summary>
            /// Writes a one dimensional <c>uint8</c> <cref>Tensor</cref> into a file.
            /// </summary>
            /// <param name="filename">Path to the file.</param>
            /// <param name="data">One dimensional <c>uint8</c> <cref>Tensor</cref>.</param>
            public static void write_file(string filename, Tensor data)
            {
                using var stream = File.OpenWrite(filename);
                data.WriteBytesToStream(stream);
            }

            /// <summary>
            /// Asynchronously writes a one dimensional <c>uint8</c> <cref>Tensor</cref> into a file.
            /// </summary>
            /// <param name="filename">Path to the file.</param>
            /// <param name="data">One dimensional <c>uint8</c> <cref>Tensor</cref>.</param>
            public static async void write_file_async(string filename, Tensor data)
            {
                using (FileStream stream = File.Open(filename, FileMode.OpenOrCreate)) {
                    await stream.WriteAsync(data.bytes.ToArray(), 0, (int)data.shape[0]);
                }
            }
        }
    }
}