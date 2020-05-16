// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

// API.

// Save a module
EXPORT_API(int) THSNN_Module_save(const char * location, const NNModule module);

// Load a module
EXPORT_API(NNModule) THSNN_Module_load(const char * location, const char * name);

// Returns a layer.
EXPORT_API(NNModule) THSNN_ReLU_ctor(bool inplace);

// Returns a layer.
EXPORT_API(NNModule) THSNN_AdaptiveAvgPool2d_ctor(const int64_t* sizes, const int length);
EXPORT_API(Tensor) THSNN_AdaptiveAvgPool2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength);
EXPORT_API(Tensor) THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength);
EXPORT_API(Tensor) THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Dropout_ctor(double probability);
EXPORT_API(Tensor) THSNN_Dropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_FeatureAlphaDropout_ctor(double probability);
EXPORT_API(Tensor) THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool with_bias);
EXPORT_API(Tensor) THSNN_Linear_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor) THSNN_Linear_bias(const NNModule module);
EXPORT_API(void) THSNN_Linear_set_bias(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor) THSNN_Linear_weight(const NNModule module);
EXPORT_API(void) THSNN_Linear_set_weight(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding);
EXPORT_API(Tensor) THSNN_Conv2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Sequential_ctor();
EXPORT_API(Tensor) THSNN_Sequential_forward(const NNModule module, const Tensor tensor);


// Returns a Conv2d layer.

EXPORT_API(NNModule) THSNN_new_module(const char ** names, at::Tensor ** parameters, const bool * require_grad, const int length);

EXPORT_API(int) THSNN_Module_has_parameter(const NNModule module, const char * name);

EXPORT_API(Tensor) THSNN_Module_get_parameter(const NNModule module, const char * name);

// Gets the named parameters of the module.
EXPORT_API(void) THSNN_Module_get_named_parameters(
    const NNModule module,
    Tensor* (*allocator1)(size_t length),
    const char** (*allocator2)(size_t length));

// Gets the parameters of the module.
EXPORT_API(void) THSNN_Module_get_parameters(
    const NNModule module,
    Tensor* (*allocator1)(size_t length));

// Whether the module is in train mode.
EXPORT_API(int) THSNN_Module_is_training(NNModule module);

// Notify the module to run in train mode.
EXPORT_API(void) THSNN_Module_train(NNModule module);

// Notify the module to run in eval mode.
EXPORT_API(void) THSNN_Module_eval(NNModule module);

// Gets the number of children modules.
EXPORT_API(long) THSNN_Module_children_size(const NNModule module);

// Returns the module name of the child submodule.
EXPORT_API(const char *) THSNN_getChildModuleName(const NNModule module, const int index);

// Returns the module name.
EXPORT_API(const char *) THSNN_Module_name(const NNModule module);

// Zero-ing the grad parameters for the input functional module.
EXPORT_API(void) THSNN_Module_zeroGrad(const NNModule module);

// Zero-ing the grad parameters for the input optimizer.
EXPORT_API(void) THSNN_Optimizer_zeroGrad(const Optimizer optimizer);

// Fetches the parameters for the optimizer.
EXPORT_API(void) THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length));

// Computes the Binary Cross Entropy (BCE) loss between input and target tensors, using a specified reduction type
// and weights if classes are unbalanced.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss for further details.
EXPORT_API(Tensor) THSTorch_binary_cross_entropy(
    const Tensor inputwrapper,
    const Tensor targetwrapper,
    const Tensor weightwrapper,
    const int64_t reduction);

// Computes the Mean squared Error (MSE, squared L2 norm) loss between the input and target tensors, using a specified reduction type.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss for further details.
EXPORT_API(Tensor) THSTorch_mse_loss(const Tensor inputwrapper, const Tensor targetwrapper, const int64_t reduction);

// Computes the Negative Log Likelihood (NLL) loss between the input and target tensors, using a specified reduction type
// and weights if classes are unbalanced. It is useful to train a classification problem with C classes.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss for further details.
EXPORT_API(Tensor) THSTorch_nll_loss(
    const Tensor inputwrapper, 
    const Tensor targetwrapper, 
    const Tensor weightwrapper, 
    const int64_t reduction);

// Negative log likelihood loss with Poisson distribution of target.
// See https://pytorch.org/docs/stable/nn.html#poisson-nll-loss for further details.
EXPORT_API(Tensor) THSTorch_poisson_nll_loss(
    const Tensor input,
    const Tensor target,
    const bool logInput,
    const bool full,
    const double eps,
    const int64_t reduction);

// Sets up the Adam optimizer
EXPORT_API(Optimizer) THSNN_Adam_ctor(const Tensor* parameters, const int len, const double learnig_rate);

// Sets up the SGD optimizer
EXPORT_API(Optimizer) THSNN_SGD_ctor(const Tensor* parameters, const int len, const double learnig_rate, const double momentum);

// Zero-ing the grad parameters for the input optimizer.
EXPORT_API(void) THSNN_Optimizer_step(const Optimizer optimizer);

/// Fills the given 2-dimensional input tensor with values drawn from a uniform
/// distribution parameterized by `low` and `high`.
/// No gradient will be recorded for this operation. This opeartion is in place.
EXPORT_API(void) THSNN_initUniform(Tensor twrapper, double low, double high);

// Fills the input `Tensor` with values according to the method
/// described in "Delving deep into rectifiers: Surpassing human-level
/// performance on ImageNet classification" - He, K. et al. (2015), using a
/// uniform distribution. Also known as He initialization.
/// No gradient will be recorded for this operation.
EXPORT_API(void) THSNN_initKaimingUniform(Tensor tensor, double a);

// Disposes the optimizer.
EXPORT_API(void) THSNN_Optimizer_dispose(const Optimizer optimizer);

// Disposes the module.
EXPORT_API(void) THSNN_Module_dispose(const NNModule module);
