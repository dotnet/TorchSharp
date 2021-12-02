using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {
        public interface IVideoer
        {
            Task<Tensor> ReadVideoAsync(string filename);
            Task WriteVideoAsync(Tensor video);
        }

        public static Tensor read_video(string filename)
        {
            throw new NotImplementedException();
        }

        public static async Task<Tensor> read_video_async(string filename)
        {
            return await Task.Run(() => {
                return ones(1);
            });
        }
    }
}
