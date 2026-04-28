// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics.CodeAnalysis;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Embedding : torch.nn.Module<Tensor, Tensor>
        {
            internal Embedding(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor input)
            {
                var res = THSNN_Embedding_forward(handle, input.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DisallowNull]
            public Parameter? weight {
                get {
                    var res = THSNN_Embedding_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_Embedding_set_weight(handle, value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight", value);
                }
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
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
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            /// <remarks>Keep in mind that only a limited number of optimizers support sparse gradients: currently it’s optim.SGD (CUDA and CPU), optim.SparseAdam (CUDA and CPU) and optim.Adagrad (CPU)</remarks>
            public static Embedding Embedding(long num_embeddings, long embedding_dims, long? padding_idx = null, double? max_norm = null, double norm_type = 2.0, bool scale_grad_by_freq = false, bool sparse = false, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Embedding_ctor(num_embeddings, embedding_dims,
                    padding_idx.HasValue ? padding_idx.Value : -1, padding_idx.HasValue,
                    max_norm.HasValue ? max_norm.Value : 0.0, max_norm.HasValue,
                    norm_type, scale_grad_by_freq, sparse, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Embedding(res, boxedHandle).MoveModule<Embedding>(device,dtype);
            }

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
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            /// <remarks>Keep in mind that only a limited number of optimizers support sparse gradients: currently it’s optim.SGD (CUDA and CPU), optim.SparseAdam (CUDA and CPU) and optim.Adagrad (CPU)</remarks>
            public static Embedding Embedding_from_pretrained(Tensor embeddings, bool freeze = true, long? padding_idx = null, double? max_norm = null, double norm_type = 2.0, bool scale_grad_by_freq = false, bool sparse = false, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Embedding_from_pretrained(embeddings.Handle, freeze,
                    padding_idx.HasValue ? padding_idx.Value : -1, padding_idx.HasValue,
                    max_norm.HasValue ? max_norm.Value : 0.0, max_norm.HasValue,
                    norm_type, scale_grad_by_freq, sparse, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Embedding(res, boxedHandle).MoveModule<Embedding>(device, dtype);
            }
        }
    }
}
