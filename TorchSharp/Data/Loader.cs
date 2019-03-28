using System;
using System.Runtime.InteropServices;

namespace TorchSharp.Data
{
    public class Loader
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr Data_LoaderMNIST(string filename, long batchSize, bool isTrain);

        /// <summary>
        /// Create an iterator scanning the MNIST dataset.
        /// </summary>
        /// <param name="filename">The position of the MNIST dataset</param>
        /// <param name="batchSize">The required batch size</param>
        /// <param name="isTrain">Wheter the iterator is for training or testing</param>
        /// <returns></returns>
        static public DataIterator<int, int> MNIST(string filename, long batchSize, bool isTrain = true)
        {
            return new DataIterator<int, int>(Data_LoaderMNIST(filename, batchSize, isTrain));
        }
    }
}
