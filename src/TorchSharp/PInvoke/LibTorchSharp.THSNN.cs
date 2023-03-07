// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {[DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_save(
            torch.nn.Module.HType handle,
            [MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string THSNN_Module_name(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_custom_module(
            [MarshalAs(UnmanagedType.LPStr)] string name,
            ForwardFunctionC forward,
            out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_RNN_forward_with_packed_input(torch.nn.Module.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, out IntPtr h_n);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_pack_padded_sequence(IntPtr input, IntPtr lengths, [MarshalAs(UnmanagedType.U1)] bool batch_first, [MarshalAs(UnmanagedType.U1)] bool enforce_sorted);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_pack_sequence(IntPtr[] sequences, int sequences_len, [MarshalAs(UnmanagedType.U1)] bool enforce_sorted);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_LSTM_forward_with_packed_input(torch.nn.Module.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, IntPtr c_0, out IntPtr h_n, out IntPtr c_n);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_GRU_forward_with_packed_input(torch.nn.Module.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, out IntPtr h_n);

        [DllImport("LibTorchSharp")]
        // align_corners -- 0=None, 1=true, 2=false
        internal static extern IntPtr THSNN_Upsample_ctor(IntPtr size, int size_length, IntPtr scale_factor, int scale_factor_length, byte mode, byte align_corners, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        // align_corners -- 0=None, 1=true, 2=false
        internal static extern IntPtr THSNN_interpolate(IntPtr input, IntPtr size, int size_len, IntPtr scale_factor, int scale_factor_len, byte mode, byte align_corners, [MarshalAs(UnmanagedType.U1)] bool recompute_scale_factor);

        [DllImport("LibTorchSharp")]
        // align_corners -- 0=None, 1=true, 2=false
        internal static extern IntPtr THSNN_grid_sample(IntPtr input, IntPtr grid, byte mode, byte padding_mode, byte align_corners);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AlphaDropout_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AlphaDropout_ctor(double p, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_alpha_dropout(IntPtr input, double p, [MarshalAs(UnmanagedType.U1)] bool training, [MarshalAs(UnmanagedType.U1)] bool inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Bilinear_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Bilinear_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Bilinear_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Bilinear_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Bilinear_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Bilinear_ctor(long in1_features, long in2_features, long output_size, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_functional_bilinear(IntPtr input1, IntPtr input2, IntPtr weights, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_CosineSimilarity_ctor(long dim, double eps, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_CosineSimilarity_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_dropout(IntPtr input, double p, [MarshalAs(UnmanagedType.U1)] bool training, [MarshalAs(UnmanagedType.U1)] bool inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_dropout2d(IntPtr input, double p, [MarshalAs(UnmanagedType.U1)] bool training, [MarshalAs(UnmanagedType.U1)] bool inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout_ctor(double p, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout1d_ctor(double p, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);


        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout2d_ctor(double p, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);


        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Embedding_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_ctor(long num_embeddings, long embedding_dims, long padding_idx, [MarshalAs(UnmanagedType.U1)] bool hasPI, double max_norm, [MarshalAs(UnmanagedType.U1)] bool hasMN, double norm_type, [MarshalAs(UnmanagedType.U1)] bool scale_grad_by_freq, [MarshalAs(UnmanagedType.U1)] bool sparse, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_from_pretrained(IntPtr embeddings, [MarshalAs(UnmanagedType.U1)] bool freeze, long padding_idx, [MarshalAs(UnmanagedType.U1)] bool hasPI, double max_norm, [MarshalAs(UnmanagedType.U1)] bool hasMN, double norm_type, [MarshalAs(UnmanagedType.U1)] bool scale_grad_by_freq, [MarshalAs(UnmanagedType.U1)] bool sparse, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Dropout3d_ctor(double p, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_dropout3d(IntPtr input, double p, [MarshalAs(UnmanagedType.U1)] bool training, [MarshalAs(UnmanagedType.U1)] bool inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_forward(torch.nn.Module.HType module, IntPtr tensor, IntPtr offsets, IntPtr per_sample_weights);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_EmbeddingBag_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_ctor(long num_embeddings, long embedding_dims, double max_norm, [MarshalAs(UnmanagedType.U1)] bool hasMN, double norm_type, [MarshalAs(UnmanagedType.U1)] bool scale_grad_by_freq, long mode, [MarshalAs(UnmanagedType.U1)] bool sparse, [MarshalAs(UnmanagedType.U1)] bool include_last_offset, long padding_idx, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_from_pretrained(IntPtr embeddings, [MarshalAs(UnmanagedType.U1)] bool freeze, double max_norm, [MarshalAs(UnmanagedType.U1)] bool hasMN, double norm_type, [MarshalAs(UnmanagedType.U1)] bool scale_grad_by_freq, long mode, [MarshalAs(UnmanagedType.U1)] bool sparse, [MarshalAs(UnmanagedType.U1)] bool include_last_offset, long padding_idx, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FeatureAlphaDropout_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FeatureAlphaDropout_ctor(double p, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_feature_alpha_dropout(IntPtr input, double p, [MarshalAs(UnmanagedType.U1)] bool training, [MarshalAs(UnmanagedType.U1)] bool inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long ignore_index, [MarshalAs(UnmanagedType.U1)] bool hasII, long reduction, double smooting);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_binary_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_binary_cross_entropy_with_logits(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction, IntPtr posWeights);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_cosine_embedding_loss(IntPtr input1, IntPtr input2, IntPtr trgt, double margin, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ctc_loss(IntPtr log_probs, IntPtr targets, IntPtr input_lengths, IntPtr target_lengths, long blank, [MarshalAs(UnmanagedType.U1)] bool zero_infinity, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_hinge_embedding_loss(IntPtr input, IntPtr trgt, double margin, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_huber_loss(IntPtr input, IntPtr trgt, double delta, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_margin_ranking_loss(IntPtr input1, IntPtr input2, IntPtr target, double margin, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_multilabel_margin_loss(IntPtr input, IntPtr target, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_multilabel_soft_margin_loss(IntPtr input, IntPtr target, IntPtr weight, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_multi_margin_loss(IntPtr input, IntPtr target, long p, double margin, IntPtr weight, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_mse_loss(IntPtr srct, IntPtr trgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_l1_loss(IntPtr srct, IntPtr trgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_nll_loss(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_poisson_loss(IntPtr srct, IntPtr trgt, [MarshalAs(UnmanagedType.U1)] bool logInput, [MarshalAs(UnmanagedType.U1)] bool full, float eps, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_kl_div_loss(IntPtr input, IntPtr target, long reduction, [MarshalAs(UnmanagedType.U1)] bool logTarget);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_smooth_l1_loss(IntPtr srct, IntPtr trgt, long reduction, double beta);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_soft_margin_loss(IntPtr srct, IntPtr trgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_triplet_margin_loss(IntPtr anchor, IntPtr positive, IntPtr negative, double margin, long p, double eps, [MarshalAs(UnmanagedType.U1)] bool swap, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_triplet_margin_with_distance_loss(IntPtr anchor, IntPtr positive, IntPtr negative, DistanceFunctionNative? distance_function, double margin, [MarshalAs(UnmanagedType.U1)] bool swap, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Optimizer_dispose(torch.optim.Optimizer.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Optimizer_zero_grad(torch.optim.Optimizer.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Optimizer_step(torch.optim.Optimizer.HType module, torch.optim.Optimizer.LossClosure closure);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Optimizer_getParameters(torch.optim.Optimizer.HType module, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_dispose(torch.nn.Module.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_to_device_dtype(torch.nn.Module.HType module, sbyte dtype, long deviceType, long deviceIndex);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_to_device(torch.nn.Module.HType module, long deviceType, long deviceIndex);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_to_dtype(torch.nn.Module.HType module, sbyte dtype);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Module_load([MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_train(torch.nn.Module.HType module, [MarshalAs(UnmanagedType.U1)] bool on);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_eval(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSNN_Module_is_training(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_zero_grad(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_get_named_parameters(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_get_named_buffers(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_get_parameters(torch.nn.Module.HType module, AllocatePinnedArray allocator, [MarshalAs(UnmanagedType.U1)] bool recurse);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_AnyModule_dispose(torch.nn.BoxedModule.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNN_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0, out IntPtr h_n);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNN_flatten_parameters(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNN_bias_ih(torch.nn.Module.HType module, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNN_bias_hh(torch.nn.Module.HType module, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNN_set_bias_ih(torch.nn.Module.HType module, IntPtr tensor, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNN_set_bias_hh(torch.nn.Module.HType module, IntPtr tensor, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNN_weight_ih(torch.nn.Module.HType module, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNN_weight_hh(torch.nn.Module.HType module, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNN_set_weight_ih(torch.nn.Module.HType module, IntPtr tensor, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNN_set_weight_hh(torch.nn.Module.HType module, IntPtr tensor, long idx);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNN_ctor(long input_size, long hidden_size, long num_layers, long nonlinearity, [MarshalAs(UnmanagedType.U1)] bool bias, [MarshalAs(UnmanagedType.U1)] bool batchFirst, double dropout, [MarshalAs(UnmanagedType.U1)] bool bidirectional, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNNCell_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNNCell_bias_ih(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNNCell_set_bias_ih(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNNCell_bias_hh(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNNCell_set_bias_hh(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNNCell_weight_ih(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNNCell_set_weight_ih(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNNCell_weight_hh(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_RNNCell_set_weight_hh(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RNNCell_ctor(long input_size, long hidden_size, long nonlinearity, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Linear_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Linear_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Linear_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Linear_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Linear_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Linear_ctor(long input_size, long output_size, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_functional_linear(IntPtr input, IntPtr weights, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Flatten_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Flatten_ctor(long startDim, long endDim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Identity_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Identity_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTMCell_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0, IntPtr c_0, out IntPtr c_n);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTMCell_bias_ih(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LSTMCell_set_bias_ih(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTMCell_bias_hh(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LSTMCell_set_bias_hh(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTMCell_weight_ih(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LSTMCell_set_weight_ih(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTMCell_weight_hh(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LSTMCell_set_weight_hh(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTMCell_ctor(long input_size, long hidden_size, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_PackedSequence_dispose(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PackedSequence_data(torch.nn.utils.rnn.PackedSequence.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PackedSequence_batch_sizes(torch.nn.utils.rnn.PackedSequence.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PackedSequence_sorted_indices(torch.nn.utils.rnn.PackedSequence.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PackedSequence_unsorted_indices(torch.nn.utils.rnn.PackedSequence.HType handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_pad_packed_sequence(torch.nn.utils.rnn.PackedSequence.HType sequence, [MarshalAs(UnmanagedType.U1)] bool batch_first, double padding_value, long total_length, out IntPtr res1, out IntPtr res2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pad_sequence(IntPtr[] sequences, int sequences_len, [MarshalAs(UnmanagedType.U1)] bool batch_first, double padding_value);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_MultiheadAttention_forward(torch.nn.Module.HType module, IntPtr query, IntPtr key, IntPtr value, IntPtr key_padding_mask, [MarshalAs(UnmanagedType.U1)] bool need_weights, IntPtr attn_mask, out IntPtr res1, out IntPtr res2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MultiheadAttention_ctor(long embeded_dim, long num_heads, double dropout, [MarshalAs(UnmanagedType.U1)] bool bias, [MarshalAs(UnmanagedType.U1)] bool add_bias_kv, [MarshalAs(UnmanagedType.U1)] bool add_zero_attn, long kdim, long vdim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_one_hot(IntPtr self, long num_classes);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PairwiseDistance_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PairwiseDistance_ctor(double p, double eps, [MarshalAs(UnmanagedType.U1)] bool keep_dim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PixelUnshuffle_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PixelUnshuffle_ctor(long downscaleFactor, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PixelShuffle_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_PixelShuffle_ctor(long upscaleFactor, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRUCell_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRUCell_bias_ih(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GRUCell_set_bias_ih(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRUCell_bias_hh(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GRUCell_set_bias_hh(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRUCell_weight_ih(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GRUCell_set_weight_ih(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRUCell_weight_hh(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GRUCell_set_weight_hh(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRUCell_ctor(long input_size, long hidden_size, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTM_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0, IntPtr c_0, out IntPtr h_n, out IntPtr c_n);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LSTM_flatten_parameters(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTM_ctor(long input_size, long hidden_size, long num_layers, [MarshalAs(UnmanagedType.U1)] bool bias, [MarshalAs(UnmanagedType.U1)] bool batchFirst, double dropout, [MarshalAs(UnmanagedType.U1)] bool bidirectional, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GRU_flatten_parameters(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRU_ctor(long input_size, long hidden_size, long num_layers, [MarshalAs(UnmanagedType.U1)] bool bias, [MarshalAs(UnmanagedType.U1)] bool batchFirst, double dropout, [MarshalAs(UnmanagedType.U1)] bool bidirectional, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRU_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0, out IntPtr h_n);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LBFGS_ctor(IntPtr parameters, int len, double learningRate, long max_iter, long max_eval, double tolerange_grad, double tolerance_change, long history_size);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LBFGS_set_lr(torch.optim.Optimizer.HType optimizer, double lr);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Sequential_ctor();

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Transformer_forward(torch.nn.Module.HType module, IntPtr src, IntPtr tgt, IntPtr src_mask, IntPtr tgt_mask, IntPtr memory_mask, IntPtr src_key_padding_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Transformer_ctor(long d_model, long nhead, long num_encoder_layers, long num_decoder_layers, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerDecoder_forward(torch.nn.Module.HType module, IntPtr tgt, IntPtr memory, IntPtr tgt_mask, IntPtr memory_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerDecoder_ctor(torch.nn.Module.HType decoder_layer, long num_layers, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerDecoderLayer_forward(torch.nn.Module.HType module, IntPtr tgt, IntPtr memory, IntPtr tgt_mask, IntPtr memory_mask, IntPtr tgt_key_padding_mask, IntPtr memory_key_padding_mask);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerDecoderLayer_ctor(long d_model, long nhead, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerEncoderLayer_forward(torch.nn.Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerEncoderLayer_ctor(long d_model, long nhead, long dim_feedforward, double dropout, long activation, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Upsample_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerEncoder_ctor(torch.nn.Module.HType encoder_layer, long num_layers, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerEncoder_forward(torch.nn.Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pad(IntPtr input, IntPtr pad, int pad_length, byte mode, double value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_affine_grid(IntPtr theta, IntPtr size, int size_len, [MarshalAs(UnmanagedType.U1)] bool align_corners);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_CELU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_CELU_ctor(double alpha, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LeakyReLU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LeakyReLU_ctor(double negative_slope, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LogSoftmax_forward(torch.nn.Module.HType handle, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LogSoftmax_ctor(long dim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm3d_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm3d_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm3d_reset_stats(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_get_mean(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_get_var(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_get_batches(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm3d_set_mean(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm3d_set_var(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm3d_ctor(long features, double eps, double momentum, [MarshalAs(UnmanagedType.U1)] bool affine, [MarshalAs(UnmanagedType.U1)] bool track_running_stats, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LayerNorm_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LayerNorm_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LayerNorm_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LayerNorm_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LayerNorm_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LayerNorm_ctor(IntPtr norm_shape, long norm_shape_len, double eps, [MarshalAs(UnmanagedType.U1)] bool elementwise_affine, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm2d_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm2d_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm2d_reset_stats(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_get_mean(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_get_var(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_get_batches(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm2d_set_mean(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm2d_set_var(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm2d_ctor(long features, double eps, double momentum, [MarshalAs(UnmanagedType.U1)] bool affine, [MarshalAs(UnmanagedType.U1)] bool track_running_stats, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm1d_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm1d_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm1d_reset_stats(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_get_mean(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_get_var(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_get_batches(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm1d_set_mean(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_InstanceNorm1d_set_var(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_InstanceNorm1d_ctor(long features, double eps, double momentum, [MarshalAs(UnmanagedType.U1)] bool affine, [MarshalAs(UnmanagedType.U1)] bool track_running_stats, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv1d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Conv1d_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv1d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Conv1d_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv1d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long dilation, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv2d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Conv2d_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv2d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Conv2d_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv2d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long dilation, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv2d_ctor_1(long inputChannel, long outputChannel, long kernelSizeX, long kernelSizeY, long strideX, long strideY, long paddingX, long paddingY, long dilationX, long dilationY, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv3d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Conv3d_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv3d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Conv3d_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv3d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long dilation, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Conv3d_ctor_1(long inputChannel, long outputChannel, long kernelSizeX, long kernelSizeY, long kernelSizeZ, long strideX, long strideY, long strideZ, long paddingX, long paddingY, long paddingZ, long dilationX, long dilationY, long dilationZ, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose1d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_ConvTranspose1d_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose1d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_ConvTranspose1d_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose1d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long outputPadding, long dilation, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose3d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_ConvTranspose3d_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose3d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_ConvTranspose3d_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose3d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long outputPadding, long dilation, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm1d_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm1d_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm1d_reset_stats(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_get_mean(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_get_var(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_get_batches(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm1d_set_mean(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm1d_set_var(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm1d_ctor(long features, double eps, double momentum, [MarshalAs(UnmanagedType.U1)] bool affine, [MarshalAs(UnmanagedType.U1)] bool track_running_stats, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GroupNorm_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GroupNorm_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GroupNorm_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GroupNorm_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GroupNorm_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GroupNorm_ctor(long num_groups, long num_channels, double eps, [MarshalAs(UnmanagedType.U1)] bool affine, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Unflatten_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Unflatten_ctor(long dim, IntPtr shape, long shape_len, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_batch_norm(IntPtr input, IntPtr running_mean, IntPtr running_var, IntPtr weight, IntPtr bias, [MarshalAs(UnmanagedType.U1)] bool training, double momentum, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_group_norm(IntPtr input, long num_groups, IntPtr weight, IntPtr bias, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_instance_norm(IntPtr input, IntPtr running_mean, IntPtr running_var, IntPtr weight, IntPtr bias, [MarshalAs(UnmanagedType.U1)] bool use_input_stats, double momentum, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern unsafe IntPtr THSNN_layer_norm(IntPtr input, long* normalized_shape, long normalized_shape_len, IntPtr weight, IntPtr bias, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_local_response_norm(IntPtr input, long size, double alpha, double beta, double k);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose2d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_ConvTranspose2d_set_bias(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose2d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_ConvTranspose2d_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConvTranspose2d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long outputPadding, long dilation, long paddingMode, long groups, [MarshalAs(UnmanagedType.U1)] bool bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm2d_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm2d_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm2d_reset_stats(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_get_mean(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_get_var(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_get_batches(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm2d_set_mean(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm2d_set_var(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm2d_ctor(long features, double eps, double momentum, [MarshalAs(UnmanagedType.U1)] bool affine, [MarshalAs(UnmanagedType.U1)] bool track_running_stats, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_bias(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm3d_set_bias(torch.nn.Module.HType module, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm3d_set_weight(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm3d_reset_stats(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_get_mean(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_get_var(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_get_batches(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm3d_set_mean(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_BatchNorm3d_set_var(torch.nn.Module.HType module, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_BatchNorm3d_ctor(long features, double eps, double momentum, [MarshalAs(UnmanagedType.U1)] bool affine, [MarshalAs(UnmanagedType.U1)] bool track_running_stats, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool1d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool1d_ctor(IntPtr pkernelSize, IntPtr pStrides, IntPtr pPadding, IntPtr pDilation, [MarshalAs(UnmanagedType.U1)] bool ceilMode, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxUnpool3d_forward(torch.nn.Module.HType module, IntPtr tensor, IntPtr indices, IntPtr outSize, int outputSizeLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxUnpool3d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr pPadding, int paddingLength, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ELU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ELU_ctor(double alpha, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GELU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GELU_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GLU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GLU_ctor(long dim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Hardshrink_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Hardshrink_ctor(double lambd, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Hardtanh_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Hardtanh_ctor(double min_val, double max_val, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Mish_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Mish_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReLU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReLU_ctor([MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReLU6_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReLU6_ctor([MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RReLU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_RReLU_ctor(double lower, double upper, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_SELU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_SELU_ctor([MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Sigmoid_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Sigmoid_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_SiLU_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_SiLU_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softmax_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softmax_ctor(long dim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softmax2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softmax2d_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softmin_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softmin_ctor(long dim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softplus_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softplus_ctor(double beta, double threshold, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softshrink_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softshrink_ctor(double lambd, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softsign_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Softsign_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Tanh_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Tanh_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Tanhshrink_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Tanhshrink_ctor(out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Threshold_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Threshold_ctor(double threshold, double value, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LocalResponseNorm_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LocalResponseNorm_ctor(long size, double alpha, double beta, double k, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad1d_ctor(double value, long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad1d_ctor_tuple(double value, long padding_left, long padding_right, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad2d_ctor(double value, long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad2d_ctor_tuple(double value, long padding_left, long padding_right, long padding_top, long padding_bottom, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad3d_ctor(double value, long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ConstantPad3d_ctor_tuple(double value, long padding_left, long padding_right, long padding_top, long padding_bottom, long padding_front, long padding_back, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad1d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad1d_ctor_tuple(long padding_left, long padding_right, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad2d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad2d_ctor_tuple(long padding_left, long padding_right, long padding_top, long padding_bottom, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad3d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReflectionPad3d_ctor_tuple(long padding_left, long padding_right, long padding_top, long padding_bottom, long padding_front, long padding_back, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad1d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad1d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad1d_ctor_tuple(long padding_left, long padding_right, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad2d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad2d_ctor_tuple(long padding_left, long padding_right, long padding_top, long padding_bottom, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad3d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ReplicationPad3d_ctor_tuple(long padding_left, long padding_right, long padding_top, long padding_bottom, long padding_front, long padding_back, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ZeroPad2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ZeroPad2d_ctor(long padding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ZeroPad2d_ctor_tuple(long padding_left, long padding_right, long padding_top, long padding_bottom, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveAvgPool1d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveAvgPool1d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveAvgPool2d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveAvgPool2d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveAvgPool3d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveAvgPool3d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveMaxPool1d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveMaxPool1d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveMaxPool2d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveMaxPool2d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveMaxPool3d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AdaptiveMaxPool3d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AvgPool1d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AvgPool1d_ctor(IntPtr pkernelSize, IntPtr pstrides, IntPtr ppadding, [MarshalAs(UnmanagedType.U1)] bool ceil_mode, [MarshalAs(UnmanagedType.U1)] bool count_include_pad, long divisor_override, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AvgPool2d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AvgPool2d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr ppadding, int paddingLength, [MarshalAs(UnmanagedType.U1)] bool ceil_mode, [MarshalAs(UnmanagedType.U1)] bool count_include_pad, long divisor_override, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AvgPool3d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_AvgPool3d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr ppadding, int paddingLength, [MarshalAs(UnmanagedType.U1)] bool ceil_mode, [MarshalAs(UnmanagedType.U1)] bool count_include_pad, long divisor_override, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FractionalMaxPool2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FractionalMaxPool2d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FractionalMaxPool2d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pOutputSize, int sizeLength, IntPtr pOutputRatio, int ratioLength, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FractionalMaxPool3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FractionalMaxPool3d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_FractionalMaxPool3d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pOutputSize, int sizeLength, IntPtr pOutputRatio, int ratioLength, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LPPool1d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LPPool1d_ctor(double norm_type, IntPtr pkernelSize, IntPtr pstrides, [MarshalAs(UnmanagedType.U1)] bool ceil_mode, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LPPool2d_forward(IntPtr module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LPPool2d_ctor(double norm_type, IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, [MarshalAs(UnmanagedType.U1)] bool ceil_mode, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool2d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool2d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool2d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr pPadding, int paddingLength, IntPtr pDilation, int dilationLength, [MarshalAs(UnmanagedType.U1)] bool ceilMode, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool3d_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool3d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxPool3d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr pPadding, int paddingLength, IntPtr pDilation, int dilationLength, [MarshalAs(UnmanagedType.U1)] bool ceilMode, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxUnpool1d_forward(torch.nn.Module.HType module, IntPtr tensor, IntPtr indices, IntPtr outSize);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxUnpool1d_ctor(IntPtr pkernelSize, IntPtr pStrides, IntPtr pPadding, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxUnpool2d_forward(torch.nn.Module.HType module, IntPtr tensor, IntPtr indices, IntPtr outSize, int outputSizeLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MaxUnpool2d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr pPadding, int paddingLength, out IntPtr pBoxedModule);
    }
}
