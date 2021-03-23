// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Embedding : Module
    {
        internal Embedding (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Embedding_forward (Module.HType module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            if (!tensor.IsIntegral()) throw new ArgumentException("Embedding input must be an integral tensor.");
            var res = THSNN_Embedding_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_Embedding_weight(Module.HType module);

        [DllImport("LibTorchSharp")]
        extern static void THSNN_Embedding_set_weight(Module.HType module, IntPtr tensor);

        public TorchTensor Weight {
            get {
                var res = THSNN_Embedding_weight(handle);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
            set {
                THSNN_Embedding_set_weight(handle, value.Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Embedding_from_pretrained(IntPtr embeddings, bool freeze, long padding_idx, bool hasPI, double max_norm, bool hasMN, double norm_type, bool scale_grad_by_freq, bool sparse, out IntPtr pBoxedModule);

        /// <summary>
        /// A simple lookup table that stores embeddings of a fixed dictionary and size.
        /// This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
        /// </summary>
        /// <param name="embeddings">FloatTensor containing weights for the Embedding in two dimensions. First dimension is being passed to Embedding as num_embeddings, second as embedding_dim.</param>
        /// <param name="freeze">If true (the default), the tensor does not get updated in the learning</param>
        /// <param name="padding_idx">If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.</param>
        /// <param name="max_norm">If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.</param>
        /// <param name="norm_type">The p of the p-norm to compute for the max_norm option. Default 2.</param>
        /// <param name="scale_grad_by_freq">If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default: false.</param>
        /// <param name="sparse">If true, gradient w.r.t. weight matrix will be a sparse tensor. Default: false</param>
        /// <returns></returns>
        public static Embedding from_pretrained(TorchTensor embeddings, bool freeze = true, long? padding_idx = null, double? max_norm = null, double norm_type = 2.0, bool scale_grad_by_freq = false, bool sparse = false)
        {
            var res = THSNN_Embedding_from_pretrained(embeddings.Handle, freeze,
                padding_idx.HasValue ? padding_idx.Value : -1, padding_idx.HasValue,
                max_norm.HasValue ? max_norm.Value : 0.0, max_norm.HasValue,
                norm_type, scale_grad_by_freq, sparse, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Embedding(res, boxedHandle);

        }

    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Embedding_ctor (long num_embeddings, long embedding_dims, long padding_idx, bool hasPI, double max_norm, bool hasMN, double norm_type, bool scale_grad_by_freq, bool sparse, out IntPtr pBoxedModule);

        /// <summary>
        /// A simple lookup table that stores embeddings of a fixed dictionary and size.
        /// This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
        /// </summary>
        /// <param name="num_embeddings">Size of the dictionary of embeddings, the vocabulary size.</param>
        /// <param name="embedding_dims">The size of each embedding vector</param>
        /// <param name="padding_idx">If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.</param>
        /// <param name="max_norm">If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.</param>
        /// <param name="norm_type">The p of the p-norm to compute for the max_norm option. Default 2.</param>
        /// <param name="scale_grad_by_freq">If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default: false.</param>
        /// <param name="sparse">If true, gradient w.r.t. weight matrix will be a sparse tensor. Default: false</param>
        /// <returns></returns>
        static public Embedding Embedding (long num_embeddings, long embedding_dims, long? padding_idx = null, double? max_norm = null, double norm_type = 2.0, bool scale_grad_by_freq = false, bool sparse = false)
        {
            var res = THSNN_Embedding_ctor (num_embeddings, embedding_dims,
                padding_idx.HasValue ? padding_idx.Value : -1, padding_idx.HasValue,
                max_norm.HasValue ? max_norm.Value : 0.0, max_norm.HasValue,
                norm_type, scale_grad_by_freq, sparse, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Embedding (res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// A simple lookup table that stores embeddings of a fixed dictionary and size.
        /// This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
        /// </summary>
        /// <param name="x">An input tensor of arbitrary shape.</param>
        /// <param name="num_embeddings">Size of the dictionary of embeddings, the vocabulary size.</param>
        /// <param name="embedding_dims">The size of each embedding vector</param>
        /// <param name="padding_idx">If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.</param>
        /// <param name="max_norm">If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.</param>
        /// <param name="norm_type">The p of the p-norm to compute for the max_norm option. Default 2.</param>
        /// <param name="scale_grad_by_freq">If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default: false.</param>
        /// <param name="sparse">If true, gradient w.r.t. weight matrix will be a sparse tensor. Default: false</param>
        /// <returns></returns>
        static public TorchTensor Embedding (TorchTensor x, long num_embeddings, long embedding_dims, long? padding_idx = null, double? max_norm = null, double norm_type = 2.0, bool scale_grad_by_freq = false, bool sparse = false)
        {
            using (var d = Modules.Embedding(num_embeddings, embedding_dims, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)) {
                return d.forward (x);
            }
        }
    }

}
