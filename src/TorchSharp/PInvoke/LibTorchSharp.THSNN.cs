// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
#pragma warning disable CA2101
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern void THSNN_Module_save(
            torch.nn.Module.HType handle,
            [MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string THSNN_Module_name(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSNN_custom_module(
            [MarshalAs(UnmanagedType.LPStr)] string name,
            ForwardFunctionC forward,
            out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_RNN_forward_with_packed_input(torch.nn.Module.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, out IntPtr h_n);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_pack_padded_sequence(IntPtr input, IntPtr lengths, byte batch_first, byte enforce_sorted);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_pack_sequence(IntPtr[] sequences, int sequences_len, byte enforce_sorted);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_LSTM_forward_with_packed_input(torch.nn.Module.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, IntPtr c_0, out IntPtr h_n, out IntPtr c_n);

        [DllImport("LibTorchSharp")]
        internal static extern torch.nn.utils.rnn.PackedSequence.HType THSNN_GRU_forward_with_packed_input(torch.nn.Module.HType module, torch.nn.utils.rnn.PackedSequence.HType input, IntPtr h_0, out IntPtr h_n);

        [DllImport("LibTorchSharp")]
        // align_corners -- 0=None, 1=true, 2=false
        internal static extern IntPtr THSNN_interpolate(IntPtr input, IntPtr size, int size_len, IntPtr scale_factor, int scale_factor_len, byte mode, byte align_corners, byte recompute_scale_factor, byte antialias);

        [DllImport("LibTorchSharp")]
        // align_corners -- 0=None, 1=true, 2=false
        internal static extern IntPtr THSNN_grid_sample(IntPtr input, IntPtr grid, byte mode, byte padding_mode, byte align_corners);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_alpha_dropout(IntPtr input, double p, byte training, byte inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_functional_bilinear(IntPtr input1, IntPtr input2, IntPtr weights, IntPtr bias);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_cosine_similarity(IntPtr input1, IntPtr input2, long dim, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_dropout(IntPtr input, double p, byte training, byte inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_dropout2d(IntPtr input, double p, byte training, byte inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_forward(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Embedding_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_ctor(long num_embeddings, long embedding_dims, long padding_idx, byte hasPI, double max_norm, byte hasMN, double norm_type, byte scale_grad_by_freq, byte sparse, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_Embedding_from_pretrained(IntPtr embeddings, byte freeze, long padding_idx, byte hasPI, double max_norm, byte hasMN, double norm_type, byte scale_grad_by_freq, byte sparse, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_dropout3d(IntPtr input, double p, byte training, byte inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_forward(torch.nn.Module.HType module, IntPtr tensor, IntPtr offsets, IntPtr per_sample_weights);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_weight(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_EmbeddingBag_set_weight(torch.nn.Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_ctor(long num_embeddings, long embedding_dims, double max_norm, byte hasMN, double norm_type, byte scale_grad_by_freq, long mode, byte sparse, byte include_last_offset, long padding_idx, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_EmbeddingBag_from_pretrained(IntPtr embeddings, byte freeze, double max_norm, byte hasMN, double norm_type, byte scale_grad_by_freq, long mode, byte sparse, byte include_last_offset, long padding_idx, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_feature_alpha_dropout(IntPtr input, double p, byte training, byte inplace);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_fold(IntPtr input, long out1, long out2, long kernel1, long kernel2, long stride1, long stride, long pad1, long pad2, long dil1, long dil2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_unfold(IntPtr input, long kernel1, long kernel2, long stride1, long stride, long pad1, long pad2, long dil1, long dil2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long ignore_index, byte hasII, long reduction, double smooting);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_binary_cross_entropy(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_binary_cross_entropy_with_logits(IntPtr srct, IntPtr trgt, IntPtr wgt, long reduction, IntPtr posWeights);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_cosine_embedding_loss(IntPtr input1, IntPtr input2, IntPtr trgt, double margin, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_ctc_loss(IntPtr log_probs, IntPtr targets, IntPtr input_lengths, IntPtr target_lengths, long blank, byte zero_infinity, long reduction);

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
        internal static extern IntPtr THSNN_poisson_loss(IntPtr srct, IntPtr trgt, byte logInput, byte full, float eps, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_kl_div_loss(IntPtr input, IntPtr target, long reduction, byte logTarget);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_smooth_l1_loss(IntPtr srct, IntPtr trgt, long reduction, double beta);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_soft_margin_loss(IntPtr srct, IntPtr trgt, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_triplet_margin_loss(IntPtr anchor, IntPtr positive, IntPtr negative, double margin, long p, double eps, byte swap, long reduction);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_triplet_margin_with_distance_loss(IntPtr anchor, IntPtr positive, IntPtr negative, DistanceFunctionNative? distance_function, double margin, byte swap, long reduction);

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
        internal static extern void THSNN_Module_to_device_dtype(torch.nn.Module.HType module, sbyte dtype, long deviceType, long deviceIndex, byte non_blocking);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_to_device(torch.nn.Module.HType module, long deviceType, long deviceIndex, byte non_blocking);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_to_dtype(torch.nn.Module.HType module, sbyte dtype, byte non_blocking);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern IntPtr THSNN_Module_load([MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_train(torch.nn.Module.HType module, byte on);

        [DllImport("LibTorchSharp")]
        internal static extern byte THSNN_Module_is_training(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_zero_grad(torch.nn.Module.HType module, byte set_to_none);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_get_named_parameters(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_get_named_buffers(torch.nn.Module.HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_Module_get_parameters(torch.nn.Module.HType module, AllocatePinnedArray allocator, byte recurse);

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
        internal static extern IntPtr THSNN_RNN_ctor(long input_size, long hidden_size, long num_layers, long nonlinearity, byte bias, byte batchFirst, double dropout, byte bidirectional, out IntPtr pBoxedModule);

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
        internal static extern IntPtr THSNN_RNNCell_ctor(long input_size, long hidden_size, long nonlinearity, byte bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_functional_linear(IntPtr input, IntPtr weights, IntPtr bias);

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
        internal static extern IntPtr THSNN_LSTMCell_ctor(long input_size, long hidden_size, byte bias, out IntPtr pBoxedModule);

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
        internal static extern void THSNN_pad_packed_sequence(torch.nn.utils.rnn.PackedSequence.HType sequence, byte batch_first, double padding_value, long total_length, out IntPtr res1, out IntPtr res2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pad_sequence(IntPtr[] sequences, int sequences_len, byte batch_first, double padding_value);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_MultiheadAttention_forward(torch.nn.Module.HType module, IntPtr query, IntPtr key, IntPtr value, IntPtr key_padding_mask, byte need_weights, IntPtr attn_mask, out IntPtr res1, out IntPtr res2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_MultiheadAttention_ctor(long embeded_dim, long num_heads, double dropout, byte bias, byte add_bias_kv, byte add_zero_attn, long kdim, long vdim, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_one_hot(IntPtr self, long num_classes);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pairwise_distance(IntPtr input1, IntPtr input2, double p, double eps, byte keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pixel_unshuffle(IntPtr tensor, long downscale_factor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pixel_shuffle(IntPtr tensor, long upscale_factor);

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
        internal static extern IntPtr THSNN_GRUCell_ctor(long input_size, long hidden_size, byte bias, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTM_forward(torch.nn.Module.HType module, IntPtr input, IntPtr h_0, IntPtr c_0, out IntPtr h_n, out IntPtr c_n);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_LSTM_flatten_parameters(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_LSTM_ctor(long input_size, long hidden_size, long num_layers, byte bias, byte batchFirst, double dropout, byte bidirectional, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern void THSNN_GRU_flatten_parameters(torch.nn.Module.HType module);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_GRU_ctor(long input_size, long hidden_size, long num_layers, byte bias, byte batchFirst, double dropout, byte bidirectional, out IntPtr pBoxedModule);

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
        internal static extern IntPtr THSNN_TransformerEncoder_ctor(torch.nn.Module.HType encoder_layer, long num_layers, out IntPtr pBoxedModule);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_TransformerEncoder_forward(torch.nn.Module.HType module, IntPtr src, IntPtr src_mask, IntPtr src_key_padding_mask);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_pad(IntPtr input, IntPtr pad, int pad_length, byte mode, double value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_affine_grid(IntPtr theta, IntPtr size, int size_len, byte align_corners);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_normalize(IntPtr input, double p, long dim, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_batch_norm(IntPtr input, IntPtr running_mean, IntPtr running_var, IntPtr weight, IntPtr bias, byte training, double momentum, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_group_norm(IntPtr input, long num_groups, IntPtr weight, IntPtr bias, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_instance_norm(IntPtr input, IntPtr running_mean, IntPtr running_var, IntPtr weight, IntPtr bias, byte use_input_stats, double momentum, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern unsafe IntPtr THSNN_layer_norm(IntPtr input, long* normalized_shape, long normalized_shape_len, IntPtr weight, IntPtr bias, double eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_local_response_norm(IntPtr input, long size, double alpha, double beta, double k);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSNN_scaled_dot_product_attention(IntPtr query, IntPtr key, IntPtr value, IntPtr attention_mask, double p, byte casual);
    }
#pragma warning restore CA2101
}
