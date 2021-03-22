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
EXPORT_API(void)        THSNN_Module_save(const NNModule module, const char * location);
EXPORT_API(NNModule)    THSNN_Module_load(const char * location, const char * name);
EXPORT_API(void)        THSNN_Module_register_module(const NNModule module, const char* name, const NNModule submodule);
EXPORT_API(void)        THSNN_Module_dispose(const NNModule module);

EXPORT_API(void)        THSNN_AnyModule_dispose(const NNAnyModule module);
//EXPORT_API(NNModule)    THSNN_AnyModule_get(const NNAnyModule module);

EXPORT_API(NNModule) THSNN_custom_module(const char* name, const char** names, at::Tensor** parameters, const bool* require_grad, const int length, Tensor(*forward)(Tensor), NNAnyModule* outAsAnyModule);

EXPORT_API(NNModule) THSNN_AvgPool1d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool1d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv1d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_MaxPool1d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool1d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AdaptiveAvgPool2d_ctor(const int64_t* sizes, const int length, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AdaptiveAvgPool2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Dropout_ctor(double probability, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_FeatureAlphaDropout_ctor(double probability, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool with_bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Linear_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Linear_bias(const NNModule module);
EXPORT_API(void)     THSNN_Linear_set_bias(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Linear_weight(const NNModule module);
EXPORT_API(void)     THSNN_Linear_set_weight(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_ReLU_ctor(bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReLU_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_LogSoftMax_ctor(int64_t dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LogSoftMax_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Sequential_ctor();
EXPORT_API(void)     THSNN_Sequential_push_back(const NNModule module, const char* name, const NNAnyModule submodule);
EXPORT_API(Tensor)   THSNN_Sequential_forward(const NNModule module, const Tensor tensor);

EXPORT_API(void) THSNN_Optimizer_zero_grad(const Optimizer optimizer);
EXPORT_API(void) THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length));
EXPORT_API(void) THSNN_Optimizer_step(const Optimizer optimizer);
EXPORT_API(void) THSNN_Optimizer_dispose(const Optimizer optimizer);

EXPORT_API(Tensor) THSNN_binary_cross_entropy(const Tensor inputwrapper, const Tensor targetwrapper, const Tensor weightwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_mse_loss(const Tensor inputwrapper, const Tensor targetwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_nll_loss(const Tensor inputwrapper, const Tensor targetwrapper, const Tensor weightwrapper, const int64_t reduction);
EXPORT_API(Tensor) THSNN_poisson_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction);

EXPORT_API(Optimizer) THSNN_Adagrad_ctor(const Tensor* parameters, const int len, const double learning_rate, const double lr_decay, const double weight_decay);
EXPORT_API(Optimizer) THSNN_Adam_ctor(const Tensor* parameters, const int len, const double learning_rate);
EXPORT_API(Optimizer) THSNN_RMSprop_ctor(const Tensor* parameters, const int len, const double learning_rate, const double alpha);

EXPORT_API(Optimizer) THSNN_SGD_ctor(const Tensor* parameters, const int len, const double learnig_rate, const double momentum);

EXPORT_API(void) THSNN_initUniform(Tensor twrapper, double low, double high);

EXPORT_API(void) THSNN_initKaimingUniform(Tensor tensor, double a);

