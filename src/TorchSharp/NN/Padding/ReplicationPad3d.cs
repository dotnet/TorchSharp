// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReplicationPad3d module.
    /// </summary>
    public class ReplicationPad3d : Module
    {
        internal ReplicationPad3d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ReplicationPad3d_forward (Module.HType module, IntPtr tensor);

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="tensor">Input tensor</param>
        /// <returns></returns>
        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_ReplicationPad3d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_ReplicationPad3d_ctor (long padding, out IntPtr pBoxedModule);

        /// <summary>
        /// Pads the input tensor using replication of the input boundary.
        /// </summary>
        /// <param name="padding">The size of the padding.</param>
        /// <returns></returns>
        static public ReplicationPad3d ReplicationPad3d(long padding)
        {
            var handle = THSNN_ReplicationPad3d_ctor(padding, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReplicationPad3d(handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Pads the input tensor using replication of the input boundary.
        /// </summary>
        /// <param name="x">Input tensor</param>
        /// <param name="padding">The size of the padding.</param>
        /// <returns></returns>
        static public TorchTensor ReplicationPad3d (TorchTensor x, long padding)
        {
            using (var d = Modules.ReplicationPad3d (padding)) {
                return d.forward (x);
            }
        }
    }

}
