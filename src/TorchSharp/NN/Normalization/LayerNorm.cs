// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a LayerNorm module.
    /// </summary>
    public class LayerNorm : nn.Module
    {
        internal LayerNorm (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_LayerNorm_forward (IntPtr module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_LayerNorm_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_LayerNorm_ctor (IntPtr norm_shape, long norm_shape_len, double eps, bool elementwise_affine, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization
        /// </summary>
        /// <returns></returns>
        static public LayerNorm LayerNorm (long[] normalizedShape, double eps = 1e-05, bool elementwiseAffine = true)
        {
            unsafe {
                fixed (long* pNormShape = normalizedShape) {
                    var handle = THSNN_LayerNorm_ctor((IntPtr)pNormShape, normalizedShape.Length, eps, elementwiseAffine, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new LayerNorm(handle, boxedHandle);
                }
            }
        }
    }

    public static partial class functional
    {
        /// <summary>
        /// Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization
        /// </summary>
        static public TorchTensor LayerNorm (TorchTensor x, long[] normalizedShape, double eps = 1e-05, bool elementwiseAffine = true)
        {
            using (var d =nn.LayerNorm (normalizedShape, eps, elementwiseAffine)) {
                return d.forward (x);
            }
        }
    }
}
