// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

// API.

EXPORT_API(int)         THSNN_Module_has_parameter(const NNModule module, const char* name);
EXPORT_API(Tensor)      THSNN_Module_get_parameter(const NNModule module, const char* name);
EXPORT_API(void)        THSNN_Module_get_named_parameters(const NNModule module, Tensor* (*allocator1)(size_t length), const char** (*allocator2)(size_t length));
EXPORT_API(void)        THSNN_Module_get_parameters(const NNModule module, Tensor* (*allocator1)(size_t length));
EXPORT_API(int)         THSNN_Module_is_training(NNModule module);
EXPORT_API(void)        THSNN_Module_train(NNModule module);
EXPORT_API(void)        THSNN_Module_eval(NNModule module);
EXPORT_API(long)        THSNN_Module_children_size(const NNModule module);
EXPORT_API(NNModule)    THSNN_Module_child(const NNModule module, const int index);
EXPORT_API(const char*) THSNN_Module_name(const NNModule module);
EXPORT_API(void)        THSNN_Module_zero_grad(const NNModule module);
EXPORT_API(void)        THSNN_Module_save(const NNModule module, const char* location);
EXPORT_API(NNModule)    THSNN_Module_load(const char* location, const char* name);
EXPORT_API(void)        THSNN_Module_register_module(const NNModule module, const char* name, const NNModule submodule);
EXPORT_API(void)        THSNN_Module_dispose(const NNModule module);
EXPORT_API(void)        THSNN_Module_to_device(NNModule module, int64_t device, int64_t index);

EXPORT_API(void)        THSNN_AnyModule_dispose(const NNAnyModule module);
//EXPORT_API(NNModule)    THSNN_AnyModule_get(const NNAnyModule module);

EXPORT_API(NNModule) THSNN_custom_module(const char* name, const char** names, at::Tensor** parameters, const bool* require_grad, const int length, Tensor(*forward)(Tensor), NNAnyModule* outAsAnyModule);

// Pooling

EXPORT_API(NNModule) THSNN_MaxPool1d_ctor(const int64_t* kernelSize, const int64_t* stride, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_MaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AdaptiveAvgPool1d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveAvgPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AdaptiveAvgPool2d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveAvgPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AdaptiveAvgPool3d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveAvgPool3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AdaptiveMaxPool1d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveMaxPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AdaptiveMaxPool2d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveMaxPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AdaptiveMaxPool3d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveMaxPool3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AvgPool1d_ctor(const int64_t* kernelSize, const int64_t* stride, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AvgPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool3d_forward(const NNModule module, const Tensor tensor);

// Convolution

EXPORT_API(NNModule) THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Conv1d_bias(const NNModule module);
EXPORT_API(void)     THSNN_Conv1d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_Conv1d_weight(const NNModule module);
EXPORT_API(void)     THSNN_Conv1d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(NNModule) THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Conv2d_weight(const NNModule module);
EXPORT_API(void)     THSNN_Conv2d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(Tensor)   THSNN_Conv2d_bias(const NNModule module);
EXPORT_API(void)     THSNN_Conv2d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(NNModule) THSNN_Conv3d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv3d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Conv3d_weight(const NNModule module);
EXPORT_API(void)     THSNN_Conv3d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(Tensor)   THSNN_Conv3d_bias(const NNModule module);
EXPORT_API(void)     THSNN_Conv3d_set_bias(const NNModule module, const Tensor bias);

EXPORT_API(NNModule) THSNN_ConvTranspose1d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ConvTranspose1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_ConvTranspose1d_bias(const NNModule module);
EXPORT_API(void)     THSNN_ConvTranspose1d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_ConvTranspose1d_weight(const NNModule module);
EXPORT_API(void)     THSNN_ConvTranspose1d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(NNModule) THSNN_ConvTranspose2d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ConvTranspose2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_ConvTranspose2d_weight(const NNModule module);
EXPORT_API(void)     THSNN_ConvTranspose2d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(Tensor)   THSNN_ConvTranspose2d_bias(const NNModule module);
EXPORT_API(void)     THSNN_ConvTranspose2d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(NNModule) THSNN_ConvTranspose3d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ConvTranspose3d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_ConvTranspose3d_weight(const NNModule module);
EXPORT_API(void)     THSNN_ConvTranspose3d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(Tensor)   THSNN_ConvTranspose3d_bias(const NNModule module);
EXPORT_API(void)     THSNN_ConvTranspose3d_set_bias(const NNModule module, const Tensor bias);

// Batch Normalization

EXPORT_API(NNModule) THSNN_BatchNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BatchNorm1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_BatchNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BatchNorm2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_BatchNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BatchNorm3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Dropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Dropout2d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Dropout3d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AlphaDropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AlphaDropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_FeatureAlphaDropout_ctor(double probability, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor);

// Linear

EXPORT_API(NNModule) THSNN_Identity_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Identity_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool with_bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Linear_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Linear_bias(const NNModule module);
EXPORT_API(void)     THSNN_Linear_set_bias(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Linear_weight(const NNModule module);
EXPORT_API(void)     THSNN_Linear_set_weight(const NNModule module, const Tensor tensor);

// Activation functions

EXPORT_API(NNModule) THSNN_CELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_CELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_GELU_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_GELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_LeakyReLU_ctor(const double negative_sloope, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LeakyReLU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ReLU_ctor(bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReLU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ReLU6_ctor(bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReLU6_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_RReLU_ctor(const double lower, const double upper, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_RReLU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_LogSoftmax_ctor(int64_t dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LogSoftmax_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_SELU_ctor(bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_SELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Sigmoid_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Sigmoid_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_SiLU_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_SiLU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Softmax_ctor(const int64_t dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Softmax_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Softmax2d_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Softmax2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Softmin_ctor(const int64_t dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Softmin_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Tanh_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Tanh_forward(const NNModule module, const Tensor tensor);

// Sparse

EXPORT_API(NNModule) THSNN_Embedding_ctor(const int64_t num_embeddings, const int64_t embedding_dims, const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq, const bool sparse, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_Embedding_from_pretrained(const Tensor embeddings, const bool freeze, const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type, const bool scale_grad_by_freq, const bool sparse, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Embedding_forward(const NNModule module, const Tensor weights);
EXPORT_API(Tensor)   THSNN_Embedding_weight(const NNModule module);
EXPORT_API(void)     THSNN_Embedding_set_weight(const NNModule module, const Tensor weights);

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


// Containers

EXPORT_API(NNModule) THSNN_Sequential_ctor();
EXPORT_API(void)     THSNN_Sequential_push_back(const NNModule module, const char* name, const NNAnyModule submodule);
EXPORT_API(Tensor)   THSNN_Sequential_forward(const NNModule module, const Tensor tensor);

// Loss functions

EXPORT_API(Tensor) THSNN_cross_entropy(const Tensor inputwrapper, const Tensor targetwrapper, const Tensor weightwrapper, const int64_t ignore_index, const bool has_ii, const int64_t reduction);
EXPORT_API(Tensor) THSNN_binary_cross_entropy(const Tensor inputwrapper, const Tensor targetwrapper, const Tensor weightwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_binary_cross_entropy_with_logits(const Tensor inputwrapper, const Tensor targetwrapper, const Tensor weightwrapper, const int64_t reduction, const Tensor pos_weights_wrapper);
EXPORT_API(Tensor) THSNN_l1_loss(const Tensor inputwrapper, const Tensor targetwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_mse_loss(const Tensor inputwrapper, const Tensor targetwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_nll_loss(const Tensor inputwrapper, const Tensor targetwrapper, const Tensor weightwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_poisson_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction);

// Optimizers

EXPORT_API(Optimizer) THSNN_Adagrad_ctor(const Tensor* parameters, const int len, const double learning_rate, const double lr_decay, const double weight_decay, const double initial_accumulator_value, const double eps);
EXPORT_API(Optimizer) THSNN_Adam_ctor(const Tensor* parameters, const int len, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad);
EXPORT_API(Optimizer) THSNN_RMSprop_ctor(const Tensor* parameters, const int length, const double learning_rate, const double alpha, const double eps, const double weight_decay, const double momentum, const bool centered);
EXPORT_API(Optimizer) THSNN_SGD_ctor(const Tensor* parameters, const int length, const double learning_rate, const double momentum, const double dampening, const double weight_decay, const bool nesterov);

EXPORT_API(void) THSNN_Optimizer_zero_grad(const Optimizer optimizer);
EXPORT_API(void) THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length));
EXPORT_API(void) THSNN_Optimizer_step(const Optimizer optimizer);
EXPORT_API(void) THSNN_Optimizer_dispose(const Optimizer optimizer);

// Misc.

EXPORT_API(Tensor) THSNN_one_hot(const Tensor self, const int64_t num_classes);

EXPORT_API(NNModule) THSNN_Flatten_ctor(const int64_t start_dim, const int64_t end_dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Flatten_forward(const NNModule module, const Tensor tensor);

// Initializers

EXPORT_API(void) THSNN_initUniform(Tensor twrapper, double low, double high);
EXPORT_API(void) THSNN_initKaimingUniform(Tensor tensor, double a);

