using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.Data
{
    public class Loader
    {
        [DllImport("LibTorchSharp")]
        extern static void Data_LoaderMNIST(
            string filename, 
            long batchSize, 
            bool isTrain,
            AllocatePinnedArray dataAllocator, 
            AllocatePinnedArray targetAllocator);

        static public IEnumerable<(ITorchTensor<float> data, ITorchTensor<float> target)> MNIST(string filename, long batchSize, bool isTrain = true)
        {
            IntPtr[] dataPtrArray;
            IntPtr[] targetPtrArray;

            using (var data = new PinnedArray<IntPtr>())
            using (var target = new PinnedArray<IntPtr>())
            {
                Data_LoaderMNIST(filename, batchSize, isTrain, data.CreateArray, target.CreateArray);
                dataPtrArray = data.Array;
                targetPtrArray = target.Array;
            }

            return dataPtrArray
                .Zip(
                    targetPtrArray, 
                    (d, t) => (
                        (ITorchTensor<float>)new FloatTensor(new FloatTensor.HType(d, true)), 
                        (ITorchTensor<float>)new FloatTensor(new FloatTensor.HType(t, true))));
        }
    }
}
