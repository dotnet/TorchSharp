// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

// API.

EXPORT_API(int)         THSNN_Module_has_parameter(const NNModule module, const char* name);
EXPORT_API(Tensor)      THSNN_Module_get_parameter(const NNModule module, const char* name);
EXPORT_API(void)        THSNN_Module_get_named_parameters(const NNModule module, Tensor* (*allocator1)(size_t length), const char** (*allocator2)(size_t length));
EXPORT_API(void)        THSNN_Module_get_named_buffers(const NNModule module, Tensor* (*allocator1)(size_t length), const char** (*allocator2)(size_t length));
EXPORT_API(void)        THSNN_Module_get_named_children(const NNModule module, NNModule* (*allocator1)(size_t length), const char** (*allocator2)(size_t length));
EXPORT_API(void)        THSNN_Module_get_named_modules(const NNModule module, NNModule* (*allocator1)(size_t length), const char** (*allocator2)(size_t length));
EXPORT_API(void)        THSNN_Module_get_parameters(const NNModule module, Tensor* (*allocator1)(size_t length), bool recurse);
EXPORT_API(int)         THSNN_Module_is_training(NNModule module);
EXPORT_API(void)        THSNN_Module_train(NNModule module, bool on);
EXPORT_API(long)        THSNN_Module_children_size(const NNModule module);
EXPORT_API(NNModule)    THSNN_Module_child(const NNModule module, const int index);
EXPORT_API(const char*) THSNN_Module_name(const NNModule module);
EXPORT_API(void)        THSNN_Module_zero_grad(const NNModule module, bool set_to_none);
EXPORT_API(void)        THSNN_Module_save(const NNModule module, const char* location);
EXPORT_API(NNModule)    THSNN_Module_load(const char* location);
EXPORT_API(void)        THSNN_Module_register_buffer(const NNModule module, const char* name, const Tensor submodule);
EXPORT_API(void)        THSNN_Module_register_parameter(const NNModule module, const char* name, const Tensor tensor, bool requires_grad);
EXPORT_API(void)        THSNN_Module_register_module(const NNModule module, const char* name, const NNModule submodule);
EXPORT_API(void)        THSNN_Module_dispose(const NNModule module);
EXPORT_API(void)        THSNN_Module_to_device(NNModule module, int64_t device, int64_t index, const bool non_blocking);
EXPORT_API(void)        THSNN_Module_to_dtype(NNModule module, int8_t dtype, const bool non_blocking);
EXPORT_API(void)        THSNN_Module_to_device_dtype(NNModule module, int8_t dtype, int64_t device, int64_t index, const bool non_blocking);

EXPORT_API(void)        THSNN_AnyModule_dispose(const NNAnyModule module);
//EXPORT_API(NNModule)    THSNN_AnyModule_get(const NNAnyModule module);

EXPORT_API(NNModule) THSNN_custom_module(const char* name, Tensor(*forward)(Tensor), NNAnyModule* outAsAnyModule);

// Normalization

EXPORT_API(Tensor)   THSNN_normalize(const Tensor input, const double p, const int64_t dim, const double eps);
EXPORT_API(Tensor)   THSNN_batch_norm(const Tensor input, const Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool training, const double momentum, const double eps);
EXPORT_API(Tensor)   THSNN_group_norm(const Tensor input, int64_t num_groups, const Tensor weight, const Tensor bias, const double eps);
EXPORT_API(Tensor)   THSNN_instance_norm(const Tensor input, const Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool use_input_stats, const double momentum, const double eps);
EXPORT_API(Tensor)   THSNN_layer_norm(const Tensor input, const int64_t* normalized_shape, const int64_t normalized_shape_len, const Tensor weight, const Tensor bias, const double eps);
EXPORT_API(Tensor)   THSNN_local_response_norm(const Tensor input, const int64_t size, const double alpha, const double beta, const double k);

// Dropout
EXPORT_API(Tensor) THSNN_dropout(const Tensor input, const double p, bool training, bool inplace);
EXPORT_API(Tensor) THSNN_dropout2d(const Tensor input, const double p, bool training, bool inplace);
EXPORT_API(Tensor) THSNN_dropout3d(const Tensor input, const double p, bool training, bool inplace);

EXPORT_API(Tensor) THSNN_alpha_dropout(const Tensor input, const double p, bool training, bool inplace);

EXPORT_API(Tensor) THSNN_feature_alpha_dropout(const Tensor input, const double p, bool training, bool inplace);

EXPORT_API(Tensor) THSNN_fold(const Tensor input, const int64_t out1, const int64_t out2, const int64_t kernel1, const int64_t kernel2, const int64_t stride1, const int64_t stride2, const int64_t pad1, const int64_t pad2, const int64_t dil1, const int64_t dil2);
EXPORT_API(Tensor) THSNN_unfold(const Tensor input, const int64_t kernel1, const int64_t kernel2, const int64_t stride1, const int64_t stride2, const int64_t pad1, const int64_t pad2, const int64_t dil1, const int64_t dil2);

// Linear

EXPORT_API(Tensor) THSNN_functional_linear(const Tensor input, const Tensor weights, const Tensor bias);
EXPORT_API(Tensor) THSNN_functional_bilinear(const Tensor input1, const Tensor input2, const Tensor weights, const Tensor bias);

// Vision -- Modules

EXPORT_API(Tensor)   THSNN_pixel_shuffle(const Tensor tensor, const int64_t upscale_factor);
EXPORT_API(Tensor)   THSNN_pixel_unshuffle(const Tensor tensor, const int64_t downscale_fasctor);

// Vision -- Functions

EXPORT_API(Tensor) THSNN_pad(const Tensor input, const int64_t* pad, const int pad_length, const int8_t mode, const double value);
EXPORT_API(Tensor) THSNN_interpolate(const Tensor input, const int64_t* size, const int size_len, const double* scale_factor, const int scale_factor_len, const int8_t mode, const int8_t align_corners, const bool recompute_scale_factor, const bool antialias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor) THSNN_grid_sample(const Tensor input, const Tensor grid, const int8_t mode, const int8_t padding_mode, const int8_t align_corners);
EXPORT_API(Tensor) THSNN_affine_grid(const Tensor theta, const int64_t* size, const int size_len, const bool align_corners);

// Sparse

EXPORT_API(NNModule) THSNN_Embedding_ctor(const int64_t num_embeddings, const int64_t embedding_dims, const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq, const bool sparse, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_Embedding_from_pretrained(const Tensor embeddings, const bool freeze, const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq, const bool sparse, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Embedding_forward(const NNModule module, const Tensor weights);
EXPORT_API(Tensor)   THSNN_Embedding_weight(const NNModule module);
EXPORT_API(void)     THSNN_Embedding_set_weight(const NNModule module, const Tensor weights);

EXPORT_API(NNModule) THSNN_EmbeddingBag_ctor(const int64_t num_embeddings, const int64_t embedding_dims, const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq, const int64_t mode, const bool sparse, const bool include_last_offset, const int64_t padding_idx, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_EmbeddingBag_from_pretrained(const Tensor embeddings, const bool freeze, const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq, const int64_t mode, const bool sparse, const bool include_last_offset, const int64_t padding_idx, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_EmbeddingBag_forward(const NNModule module, const Tensor input, const Tensor offsets, const Tensor per_sample_weights);
EXPORT_API(Tensor)   THSNN_EmbeddingBag_weight(const NNModule module);
EXPORT_API(void)     THSNN_EmbeddingBag_set_weight(const NNModule module, const Tensor weights);

// Transformer

EXPORT_API(NNModule) THSNN_Transformer_ctor(const int64_t d_model, const int64_t nhead, const int64_t num_encoder_layers, const int64_t num_decoder_layers, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Transformer_forward(const NNModule module, const Tensor src, const Tensor tgt, const Tensor src_mask, const Tensor tgt_mask, const Tensor memory_mask, const Tensor src_key_padding_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask);
EXPORT_API(NNModule) THSNN_TransformerEncoderLayer_ctor(const int64_t d_model, const int64_t nhead, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_TransformerEncoderLayer_forward(const NNModule module, const Tensor src, const Tensor src_mask, const Tensor src_key_padding_mask);
EXPORT_API(NNModule) THSNN_TransformerDecoderLayer_ctor(const int64_t d_model, const int64_t nhead, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_TransformerDecoderLayer_forward(const NNModule module, const Tensor tgt, const Tensor memory, const Tensor tgt_mask, const Tensor memory_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask);
EXPORT_API(NNModule) THSNN_TransformerEncoder_ctor(const NNModule encoder_layer, const int64_t num_layers, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_TransformerEncoder_forward(const NNModule module, const Tensor src, const Tensor src_mask, const Tensor src_key_padding_mask);
EXPORT_API(NNModule) THSNN_TransformerDecoder_ctor(const NNModule decoder_layer, const int64_t num_layers, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_TransformerDecoder_forward(const NNModule module, const Tensor tgt, const Tensor memory, const Tensor tgt_mask, const Tensor memory_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask);
EXPORT_API(NNModule) THSNN_MultiheadAttention_ctor(const int64_t embeded_dim, const int64_t num_heads, const double dropout, const bool bias, const bool add_bias_kv, const bool add_zero_attn, const int64_t kdim, const int64_t vdim, NNAnyModule* outAsAnyModule);
EXPORT_API(void)     THSNN_MultiheadAttention_forward(const NNModule module, const Tensor query, const Tensor key, const Tensor value, const Tensor key_padding_mask, const bool need_weights, const Tensor attn_mask, Tensor& res1, Tensor& res2);


// Recurrent

EXPORT_API(NNModule) THSNN_RNN_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const int64_t nonlinearity, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_RNN_forward(const NNModule module, const Tensor input1, const Tensor input2, Tensor* h_n);
EXPORT_API(PackedSequence) THSNN_RNN_forward_with_packed_input(const NNModule module, const PackedSequence input1, const Tensor input2, Tensor* h_n);
EXPORT_API(NNModule) THSNN_GRU_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_GRU_forward(const NNModule module, const Tensor input1, const Tensor input2, Tensor* h_n);
EXPORT_API(PackedSequence) THSNN_GRU_forward_with_packed_input(const NNModule module, const PackedSequence input1, const Tensor input2, Tensor* h_n);
EXPORT_API(NNModule) THSNN_LSTM_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LSTM_forward(const NNModule module, const Tensor input1, const Tensor h0, const Tensor c0, Tensor* h_n, Tensor* c_n);
EXPORT_API(PackedSequence) THSNN_LSTM_forward_with_packed_input(const NNModule module, const PackedSequence input1, const Tensor h0, const Tensor c0, Tensor* h_n, Tensor* c_n);

EXPORT_API(NNModule) THSNN_RNNCell_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t nonlinearity, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_RNNCell_forward(const NNModule module, const Tensor input1, const Tensor h0);
EXPORT_API(NNModule) THSNN_GRUCell_ctor(const int64_t input_size, const int64_t hidden_size, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_GRUCell_forward(const NNModule module, const Tensor input1, const Tensor h0);
EXPORT_API(NNModule) THSNN_LSTMCell_ctor(const int64_t input_size, const int64_t hidden_size, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LSTMCell_forward(const NNModule module, const Tensor input1, const Tensor h0, const Tensor c0, Tensor* c_n);

EXPORT_API(void) THSNN_RNN_flatten_parameters(const NNModule module);
EXPORT_API(void) THSNN_GRU_flatten_parameters(const NNModule module);
EXPORT_API(void) THSNN_LSTM_flatten_parameters(const NNModule module);

EXPORT_API(Tensor) THSNN_RNN_bias_ih(const NNModule module, const int64_t idx);
EXPORT_API(void) THSNN_RNN_set_bias_ih(const NNModule module, const Tensor bias, const int64_t idx);
EXPORT_API(Tensor) THSNN_RNN_weight_ih(const NNModule module, const int64_t idx);
EXPORT_API(void) THSNN_RNN_set_weight_ih(const NNModule module, const Tensor weight, const int64_t idx);
EXPORT_API(Tensor) THSNN_RNN_bias_hh(const NNModule module, const int64_t idx);
EXPORT_API(void) THSNN_RNN_set_bias_hh(const NNModule module, const Tensor bias, const int64_t idx);
EXPORT_API(Tensor) THSNN_RNN_weight_hh(const NNModule module, const int64_t idx);
EXPORT_API(void) THSNN_RNN_set_weight_hh(const NNModule module, const Tensor weight, const int64_t idx);

EXPORT_API(Tensor) THSNN_RNNCell_bias_ih(const NNModule module);
EXPORT_API(void) THSNN_RNNCell_set_bias_ih(const NNModule module, const Tensor bias);
EXPORT_API(Tensor) THSNN_RNNCell_weight_ih(const NNModule module);
EXPORT_API(void) THSNN_RNNCell_set_weight_ih(const NNModule module, const Tensor weight);
EXPORT_API(Tensor) THSNN_RNNCell_bias_hh(const NNModule module);
EXPORT_API(void) THSNN_RNNCell_set_bias_hh(const NNModule module, const Tensor bias);
EXPORT_API(Tensor) THSNN_RNNCell_weight_hh(const NNModule module);
EXPORT_API(void) THSNN_RNNCell_set_weight_hh(const NNModule module, const Tensor weight);

EXPORT_API(Tensor) THSNN_LSTMCell_bias_ih(const NNModule module);
EXPORT_API(void) THSNN_LSTMCell_set_bias_ih(const NNModule module, const Tensor bias);
EXPORT_API(Tensor) THSNN_LSTMCell_weight_ih(const NNModule module);
EXPORT_API(void) THSNN_LSTMCell_set_weight_ih(const NNModule module, const Tensor weight);
EXPORT_API(Tensor) THSNN_LSTMCell_bias_hh(const NNModule module);
EXPORT_API(void) THSNN_LSTMCell_set_bias_hh(const NNModule module, const Tensor bias);
EXPORT_API(Tensor) THSNN_LSTMCell_weight_hh(const NNModule module);
EXPORT_API(void) THSNN_LSTMCell_set_weight_hh(const NNModule module, const Tensor weight);

EXPORT_API(Tensor) THSNN_GRUCell_bias_ih(const NNModule module);
EXPORT_API(void) THSNN_GRUCell_set_bias_ih(const NNModule module, const Tensor bias);
EXPORT_API(Tensor) THSNN_GRUCell_weight_ih(const NNModule module);
EXPORT_API(void) THSNN_GRUCell_set_weight_ih(const NNModule module, const Tensor weight);
EXPORT_API(Tensor) THSNN_GRUCell_bias_hh(const NNModule module);
EXPORT_API(void) THSNN_GRUCell_set_bias_hh(const NNModule module, const Tensor bias);
EXPORT_API(Tensor) THSNN_GRUCell_weight_hh(const NNModule module);
EXPORT_API(void) THSNN_GRUCell_set_weight_hh(const NNModule module, const Tensor weight);

// Containers

EXPORT_API(NNModule) THSNN_Sequential_ctor();
EXPORT_API(void)     THSNN_Sequential_push_back(const NNModule module, const char* name, const NNAnyModule submodule);
EXPORT_API(Tensor)   THSNN_Sequential_forward(const NNModule module, const Tensor tensor);

// Loss functions

EXPORT_API(Tensor) THSNN_binary_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction);
EXPORT_API(Tensor) THSNN_binary_cross_entropy_with_logits(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction, const Tensor pos_weights_);
EXPORT_API(Tensor) THSNN_cosine_embedding_loss(const Tensor input1, const Tensor input2, const Tensor target, const double margin, const int64_t reduction);
EXPORT_API(Tensor) THSNN_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t ignore_index, const bool has_ii, const int64_t reduction, const double smoothing);
EXPORT_API(Tensor) THSNN_ctc_loss(const Tensor log_probs, const Tensor targets, const Tensor input_lengths, const Tensor target_lengths, int64_t blank, bool zero_infinity, const int64_t reduction);
EXPORT_API(Tensor) THSNN_hinge_embedding_loss(const Tensor input, const Tensor target, const double margin, const int64_t reduction);
EXPORT_API(Tensor) THSNN_huber_loss(const Tensor input, const Tensor target, const double delta, const int64_t reduction);
EXPORT_API(Tensor) THSNN_l1_loss(const Tensor input, const Tensor target, const int64_t reduction);
EXPORT_API(Tensor) THSNN_margin_ranking_loss(const Tensor input1, const Tensor input2, const Tensor target, const double margin, const int64_t reduction);
EXPORT_API(Tensor) THSNN_mse_loss(const Tensor input, const Tensor target, const int64_t reduction);
EXPORT_API(Tensor) THSNN_multilabel_margin_loss(const Tensor input, const Tensor target, const int64_t reduction);
EXPORT_API(Tensor) THSNN_multilabel_soft_margin_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction);
EXPORT_API(Tensor) THSNN_multi_margin_loss(const Tensor input, const Tensor target, const int64_t p, const double margin, const Tensor weight, const int64_t reduction);
EXPORT_API(Tensor) THSNN_nll_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction);
EXPORT_API(Tensor) THSNN_poisson_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction);
EXPORT_API(Tensor) THSNN_kl_div_loss(const Tensor input, const Tensor target, const int64_t reduction, const bool log_target);
EXPORT_API(Tensor) THSNN_smooth_l1_loss(const Tensor input, const Tensor target, const int64_t reduction, const double beta);
EXPORT_API(Tensor) THSNN_soft_margin_loss(const Tensor input, const Tensor target, const int64_t reduction);
EXPORT_API(Tensor) THSNN_triplet_margin_loss(const Tensor anchor, const Tensor positive, const Tensor negative, double margin, int64_t p, double eps, bool swap, const int64_t reduction);
EXPORT_API(Tensor) THSNN_triplet_margin_with_distance_loss(const Tensor anchor, const Tensor positive, const Tensor negative, Tensor (*distance_function)(const Tensor x, const Tensor y), double margin, bool swap, const int64_t reduction);

// Optimizers

EXPORT_API(Optimizer) THSNN_Adagrad_ctor(const Tensor* parameters, const int len, const double learning_rate, const double lr_decay, const double weight_decay, const double initial_accumulator_value, const double eps);
EXPORT_API(Optimizer) THSNN_Adam_ctor(const Tensor* parameters, const int len, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad);
EXPORT_API(Optimizer) THSNN_AdamW_ctor(const Tensor* parameters, const int len, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad);
EXPORT_API(Optimizer) THSNN_LBFGS_ctor(const Tensor* parameters, const int len, const double lr, const int64_t max_iter, const int64_t max_eval, const double tolerange_grad, const double tolerance_change, const int64_t history_size);
EXPORT_API(Optimizer) THSNN_RMSprop_ctor(const Tensor* parameters, const int length, const double learning_rate, const double alpha, const double eps, const double weight_decay, const double momentum, const bool centered);
EXPORT_API(Optimizer) THSNN_SGD_ctor(const Tensor* parameters, const int length, const double learning_rate, const double momentum, const double dampening, const double weight_decay, const bool nesterov);

EXPORT_API(void) THSNN_Adam_set_betas(const Optimizer optimizer, double beta1, double beta2);
EXPORT_API(void) THSNN_AdamW_set_betas(const Optimizer optimizer, double beta1, double beta2);
EXPORT_API(void) THSNN_RMSprop_set_momentum(const Optimizer optimizer, double momentum);
EXPORT_API(void) THSNN_SGD_set_momentum(const Optimizer optimizer, double momentum);

EXPORT_API(void)   THSNN_Optimizer_zero_grad(const Optimizer optimizer);
EXPORT_API(void)   THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length));
EXPORT_API(Tensor) THSNN_Optimizer_step(const Optimizer optimizer, Tensor(*loss_closure)());

EXPORT_API(void) THSNN_Optimizer_dispose(const Optimizer optimizer);

EXPORT_API(void) THSNN_Adagrad_set_lr(const Optimizer optimizer, const double lr);
EXPORT_API(void) THSNN_Adam_set_lr(const Optimizer optimizer, const double lr);
EXPORT_API(void) THSNN_AdamW_set_lr(const Optimizer optimizer, const double lr);
EXPORT_API(void) THSNN_LBFGS_set_lr(const Optimizer optimizer, const double lr);
EXPORT_API(void) THSNN_RMSprop_set_lr(const Optimizer optimizer, const double lr);
EXPORT_API(void) THSNN_SGD_set_lr(const Optimizer optimizer, const double lr);

// Misc.

EXPORT_API(Tensor) THSNN_one_hot(const Tensor self, const int64_t num_classes);

EXPORT_API(Tensor)   THSNN_cosine_similarity(const Tensor input1, const Tensor input2, int64_t dim, double eps);

EXPORT_API(Tensor) THSNN_pairwise_distance(const Tensor input1, const Tensor input2, double p, double eps, bool keepdim);

EXPORT_API(Tensor) THSNN_scaled_dot_product_attention(const Tensor query, const Tensor key, const Tensor value, const Tensor attention_mask, double p, bool casual);

// Initializers

EXPORT_API(void) THSNN_initUniform(Tensor twrapper, double low, double high);
EXPORT_API(void) THSNN_initKaimingUniform(Tensor tensor, double a);

// Utils RNN

EXPORT_API(Tensor) THSNN_PackedSequence_data(PackedSequence sequence);
EXPORT_API(Tensor) THSNN_PackedSequence_batch_sizes(PackedSequence sequence);
EXPORT_API(Tensor) THSNN_PackedSequence_sorted_indices(PackedSequence sequence);
EXPORT_API(Tensor) THSNN_PackedSequence_unsorted_indices(PackedSequence sequence);
EXPORT_API(void) THSNN_PackedSequence_dispose(PackedSequence sequence);
EXPORT_API(PackedSequence) THSNN_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first, bool enforce_sorted);
EXPORT_API(void) THSNN_pad_packed_sequence(PackedSequence sequence, bool batch_first, double padding_value, int64_t total_length, Tensor* res1, Tensor* res2);
EXPORT_API(Tensor) THSNN_pad_sequence(const Tensor* sequences, const int sequences_len, bool batch_first, double padding_value);
EXPORT_API(PackedSequence) THSNN_pack_sequence(const Tensor* sequences, int sequences_len, bool enforce_sorted);
