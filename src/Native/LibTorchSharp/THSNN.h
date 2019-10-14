#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

// API.

// Returns a ReLu layer.
EXPORT_API(NNModule) THSNN_reluModule();

// Returns a linear layer.
EXPORT_API(NNModule) THSNN_linearModule(const int64_t input_size, const int64_t output_size, const bool with_bias);

// Returns a Conv2d layer.
EXPORT_API(NNModule) THSNN_conv2dModule(
    const int64_t inputChannel,
    const int64_t outputChannel,
    const int64_t kernelSize,
    const int64_t stride,
    const int64_t padding);

EXPORT_API(NNModule) THSNN_new_module(const char ** names, at::Tensor ** parameters, const bool * require_grad, const int length);

EXPORT_API(int) THSNN_has_parameter(const NNModule module, const char * name);

EXPORT_API(Tensor) THSNN_get_parameter(const NNModule module, const char * name);

// Gets the named parameters of the module.
EXPORT_API(void) THSNN_get_named_parameters(
    const NNModule module,
    Tensor* (*allocator1)(size_t length),
    const char** (*allocator2)(size_t length));

// Gets the parameters of the module.
EXPORT_API(void) THSNN_get_parameters(
    const NNModule module,
    Tensor* (*allocator1)(size_t length));

// Whether the module is in train mode.
EXPORT_API(int) THSNN_is_training(NNModule module);

// Notify the module to run in train mode.
EXPORT_API(void) THSNN_train(NNModule module);

// Notify the module to run in eval mode.
EXPORT_API(void) THSNN_eval(NNModule module);

// Gets the number of children modules.
EXPORT_API(long) THSNN_getNumberOfChildren(const NNModule module);

// Returns the module name of the child submodule.
EXPORT_API(const char *) THSNN_getChildModuleName(const NNModule module, const int index);

// Returns the module name.
EXPORT_API(const char *) THSNN_getModuleName(const NNModule module);

// Applies a ReLu activation function on the input tensor. 
EXPORT_API(Tensor) THSNN_reluApply(const Tensor tensor);

// Applies a maxpool 2d on the input tensor. 
EXPORT_API(Tensor) THSNN_maxPool2DApply(
    const Tensor tensor,
    const int kernelSizeLength,
    const int64_t* kernelSize,
    const int strideLength,
    const int64_t* stride);

// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
EXPORT_API(Tensor) THSNN_adaptiveAvgPool2DApply(const Tensor tensor, const int length, const int64_t* outputSize);

// Applies a avgpool 2d on the input tensor. 
EXPORT_API(Tensor) THSNN_avgPool2DApply(
	const Tensor tensor,
	const int kernelSizeLength,
	const int64_t* kernelSize,
	const int strideLength,
	const int64_t* stride);

// Applies a log soft max on the input tensor. 
EXPORT_API(Tensor) THSNN_logSoftMaxApply(const Tensor tensor, const int64_t dimension);

// Applies a log soft max on the input tensor. 
EXPORT_API(Tensor) THSNN_featureDropoutApply(const Tensor tensor);

// Applies drop out on the input tensor. 
EXPORT_API(Tensor) THSNN_dropoutModuleApply(
    const Tensor tensor, 
    const double probability, 
    const bool isTraining);

// Triggers a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
EXPORT_API(Tensor) THSNN_linearModuleApply(const NNModule module, const Tensor tensor);

// Triggers a forward pass over an input linear module (e.g., activation functions) using the input tensor. 
EXPORT_API(Tensor) THSNN_conv2DModuleApply(
    const NNModule module,
    const Tensor tensor);

// Whether the linear module was setup with bias or not.
EXPORT_API(int) THSNN_linear_with_bias(const NNModule module);

// Returns the bias term of the linear module.
EXPORT_API(Tensor) THSNN_linear_get_bias(const NNModule module);

// Sets the bias term for the linear module.
EXPORT_API(void) THSNN_linear_set_bias(const NNModule module, const Tensor tensor);

// Returns the weights of the linear module.
EXPORT_API(Tensor) THSNN_linear_get_weight(const NNModule module);

// Sets the weights of the linear module.
EXPORT_API(void) THSNN_linear_set_weight(const NNModule module, Tensor tensor);

// Zero-ing the grad parameters for the input functional module.
EXPORT_API(void) THSNN_moduleZeroGrad(const NNModule module);

// Zero-ing the grad parameters for the input optimizer.
EXPORT_API(void) THSNN_optimizerZeroGrad(const Optimizer optimizer);

// Fetches the parameters for the optimizer.
EXPORT_API(void) THSNN_optimizer_get_parameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length));

// Computes the Binary Cross Entropy (BCE) loss between input and target tensors, using a specified reduction type
// and weights if classes are unbalanced.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss for further details.
EXPORT_API(Tensor) THSNN_lossBCE(
    const Tensor inputwrapper,
    const Tensor targetwrapper,
    const Tensor weightwrapper,
    const int64_t reduction);

// Computes the Mean squared Error (MSE, squared L2 norm) loss between the input and target tensors, using a specified reduction type.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss for further details.
EXPORT_API(Tensor) THSNN_lossMSE(const Tensor inputwrapper, const Tensor targetwrapper, const int64_t reduction);

// Computes the Negative Log Likelihood (NLL) loss between the input and target tensors, using a specified reduction type
// and weights if classes are unbalanced. It is useful to train a classification problem with C classes.
// See https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss for further details.
EXPORT_API(Tensor) THSNN_lossNLL(
    const Tensor inputwrapper, 
    const Tensor targetwrapper, 
    const Tensor weightwrapper, 
    const int64_t reduction);

// Negative log likelihood loss with Poisson distribution of target.
// See https://pytorch.org/docs/stable/nn.html#poisson-nll-loss for further details.
EXPORT_API(Tensor) THSNN_loss_poisson_nll(
    const Tensor input,
    const Tensor target,
    const bool logInput,
    const bool full,
    const double eps,
    const int64_t reduction);

// Sets up the Adam optimizer
EXPORT_API(Optimizer) THSNN_optimizerAdam(const Tensor* parameters, const int len, const double learnig_rate);

// Sets up the SGD optimizer
EXPORT_API(Optimizer) THSNN_optimizerSGD(const Tensor* parameters, const int len, const double learnig_rate, const double momentum);

// Zero-ing the grad parameters for the input optimizer.
EXPORT_API(void) THSNN_optimizerStep(const Optimizer optimizer);

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
EXPORT_API(void) THSNN_optimizerDispose(const Optimizer optimizer);

// Disposes the module.
EXPORT_API(void) THSNN_moduleDispose(const NNModule module);
