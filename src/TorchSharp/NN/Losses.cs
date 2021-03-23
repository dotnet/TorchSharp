// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable

namespace TorchSharp.NN
{
    public delegate TorchTensor Loss (TorchTensor source, TorchTensor target);

    /// <summary>
    /// Class maintaing the supported loss functions.
    /// </summary>
    public static partial class Functions
    {

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long ignore_index, bool hasII, long reduction);

        public static Loss cross_entropy_loss(TorchTensor? weigths = null, long? ignore_index = null, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var ii = ignore_index.HasValue ? ignore_index.Value : -100;
                var res = THSNN_cross_entropy (src.Handle, target.Handle, weigths?.Handle ?? IntPtr.Zero, ii, ignore_index.HasValue, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor (res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_binary_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        public static Loss binary_cross_entropy_loss(TorchTensor? weigths = null, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_binary_cross_entropy(src.Handle, target.Handle, weigths?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_binary_cross_entropy_with_logits(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction, IntPtr posWeights);

        public static Loss binary_cross_entropy_with_logits_loss(TorchTensor? weigths = null, Reduction reduction = Reduction.Mean, TorchTensor? posWeights = null)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_binary_cross_entropy_with_logits(src.Handle, target.Handle, weigths?.Handle ?? IntPtr.Zero, (long)reduction, posWeights?.Handle ?? IntPtr.Zero);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_mse_loss (IntPtr srct, IntPtr trgt, long reduction);

        public static Loss mse_loss (Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                    var res = THSNN_mse_loss (src.Handle, target.Handle, (long)reduction);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (res);
                };
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_l1_loss(IntPtr srct, IntPtr trgt, long reduction);

        public static Loss l1_loss(Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_l1_loss(src.Handle, target.Handle, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            };
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_nll_loss (IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        public static Loss nll_loss (TorchTensor? weigths = null, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_nll_loss (src.Handle, target.Handle, weigths?.Handle ?? IntPtr.Zero, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor (res);
            };
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_poisson_loss (IntPtr srct, IntPtr trgt, bool logInput, bool full, float eps, long reduction);

        public static Loss poisson_loss (bool logInput = true, bool full = false, float eps = 1e-8f, Reduction reduction = Reduction.Mean)
        {
            return (TorchTensor src, TorchTensor target) => {
                var res = THSNN_poisson_loss (src.Handle, target.Handle, logInput, full, eps, (long)reduction);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor (res);
            };
        }
    }
    public enum Reduction : long
    {
        None = 0,
        Mean = 1,
        Sum = 2
    }
}
