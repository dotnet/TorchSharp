using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    partial class io
    {
        public static Tensor read_file(string filename)
        {
            return File.ReadAllBytes(filename);
        }

        public static async Task<Tensor> read_file_async(string filename)
        {
            byte[] data;

            using (FileStream stream = File.Open(filename, FileMode.Open)) {
                data = new byte[stream.Length];
                await stream.ReadAsync(data, 0, data.Length);
            }

            return data;
        }

        public static void write_file(string filename, Tensor data)
        {
            File.WriteAllBytes(filename, data.bytes.ToArray());
        }

        public static async void write_file_async(string filename, Tensor data)
        {
            using (FileStream stream = File.Open(filename, FileMode.OpenOrCreate)) {
                await stream.WriteAsync(data.bytes.ToArray(), 0, (int)data.shape[0]);
            }
        }
    }
}
