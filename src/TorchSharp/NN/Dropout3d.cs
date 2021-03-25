// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a Dropout3d module.
    /// </summary>
    public class Dropout3d : Module
    {
        internal Dropout3d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Dropout3d_forward (Module.HType module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Dropout3d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_Dropout3d_ctor(double probability, bool inPlace, out IntPtr pBoxedModule);

        /// <summary>
        /// Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 3D tensor \text{input}[i, j]input[i,j] ).
        /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
        /// </summary>
        /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
        /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
        /// <returns></returns>
        static public Dropout3d Dropout3d(double probability = 0.5, bool inPlace = false)
        {
            var handle = THSNN_Dropout3d_ctor(probability, inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Dropout3d(handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 3D tensor \text{input}[i, j]input[i,j] ).
        /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
        /// </summary>
        /// <param name="x">Input tensor</param>
        /// <param name="probability">Probability of an element to be zeroed. Default: 0.5</param>
        /// <param name="inPlace">If set to true, will do this operation in-place. Default: false</param>
        /// <returns></returns>
        static public TorchTensor Dropout3d(TorchTensor x, double probability = 0.5, bool inPlace = false)
        {
            using (var d = Modules.Dropout3d(probability, inPlace)) {
                return d.forward(x);
            }
        }
    }

}
