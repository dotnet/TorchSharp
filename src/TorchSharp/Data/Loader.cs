// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp.Data
{
    public class Loader
    {
        /// <summary>
        /// Create an iterator scanning the MNIST dataset.
        /// </summary>
        /// <param name="filename">The position of the MNIST dataset</param>
        /// <param name="batchSize">The required batch size</param>
        /// <param name="isTrain">Wheter the iterator is for training or testing</param>
        /// <returns></returns>
        public static DataIterator MNIST(string filename, long batchSize, bool isTrain = true)
        {
            return new DataIterator(THSData_loaderMNIST(filename, batchSize, isTrain));
        }

        /// <summary>
        /// Create an iterator scanning the CIFAR10 dataset.
        /// </summary>
        /// <param name="path">The position of the CIFAR10 dataset</param>
        /// <param name="batchSize">The required batch size</param>
        /// <param name="isTrain">Wheter the iterator is for training or testing</param>
        /// <returns></returns>
        public static DataIterator CIFAR10(string path, long batchSize, bool isTrain = true)
        {
            return new DataIterator(THSData_loaderCIFAR10(path, batchSize, isTrain));
        }
    }
}
