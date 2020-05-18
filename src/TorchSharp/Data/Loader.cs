// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.Data
{
    public class Loader
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSData_loaderMNIST([MarshalAs(UnmanagedType.LPStr)]string filename,
            long batchSize,
            bool isTrain);

        /// <summary>
        /// Create an iterator scanning the MNIST dataset.
        /// </summary>
        /// <param name="filename">The position of the MNIST dataset</param>
        /// <param name="batchSize">The required batch size</param>
        /// <param name="isTrain">Wheter the iterator is for training or testing</param>
        /// <returns></returns>
        static public DataIterator MNIST(string filename, long batchSize, bool isTrain = true)
        {
            return new DataIterator(THSData_loaderMNIST(filename, batchSize, isTrain));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSData_loaderCIFAR10([MarshalAs(UnmanagedType.LPStr)] string path,
            long batchSize, bool isTrain);

        /// <summary>
        /// Create an iterator scanning the CIFAR10 dataset.
        /// </summary>
        /// <param name="path">The position of the CIFAR10 dataset</param>
        /// <param name="batchSize">The required batch size</param>
        /// <param name="isTrain">Wheter the iterator is for training or testing</param>
        /// <returns></returns>
        static public DataIterator CIFAR10(string path, long batchSize, bool isTrain = true)
        {
            return new DataIterator(THSData_loaderCIFAR10(path, batchSize, isTrain));
        }
    }
}
