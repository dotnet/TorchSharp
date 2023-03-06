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
EXPORT_API(void)        THSNN_Module_zero_grad(const NNModule module);
EXPORT_API(void)        THSNN_Module_save(const NNModule module, const char* location);
EXPORT_API(NNModule)    THSNN_Module_load(const char* location);
EXPORT_API(void)        THSNN_Module_register_buffer(const NNModule module, const char* name, const Tensor submodule);
EXPORT_API(void)        THSNN_Module_register_parameter(const NNModule module, const char* name, const Tensor tensor, bool requires_grad);
EXPORT_API(void)        THSNN_Module_register_module(const NNModule module, const char* name, const NNModule submodule);
EXPORT_API(void)        THSNN_Module_dispose(const NNModule module);
EXPORT_API(void)        THSNN_Module_to_device(NNModule module, int64_t device, int64_t index);
EXPORT_API(void)        THSNN_Module_to_dtype(NNModule module, int8_t dtype);
EXPORT_API(void)        THSNN_Module_to_device_dtype(NNModule module, int8_t dtype, int64_t device, int64_t index);

EXPORT_API(void)        THSNN_AnyModule_dispose(const NNAnyModule module);
//EXPORT_API(NNModule)    THSNN_AnyModule_get(const NNAnyModule module);

EXPORT_API(NNModule) THSNN_custom_module(const char* name, Tensor(*forward)(Tensor), NNAnyModule* outAsAnyModule);

// Pooling

EXPORT_API(NNModule) THSNN_MaxPool1d_ctor(const int64_t* kernelSize, const int64_t* stride, const int64_t* padding, const int64_t* dilation, bool ceil_mode, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_MaxPool1d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor *indices);

EXPORT_API(NNModule) THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, const int64_t* padding, const int paddingLength, const int64_t* dilation, const int dilationLength, bool ceil_mode, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_MaxPool2d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices);

EXPORT_API(NNModule) THSNN_MaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, const int64_t* padding, const int paddingLength, const int64_t* dilation, const int dilationLength, bool ceil_mode, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxPool3d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_MaxPool3d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices);

EXPORT_API(NNModule) THSNN_FractionalMaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* outputSize, const int outputSizeLength, const double* outputRatio, const int outputRatioLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_FractionalMaxPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_FractionalMaxPool2d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices);

EXPORT_API(NNModule) THSNN_FractionalMaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* outputSize, const int outputSizeLength, const double* outputRatio, const int outputRatioLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_FractionalMaxPool3d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_FractionalMaxPool3d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices);

EXPORT_API(NNModule) THSNN_MaxUnpool1d_ctor(const int64_t* kernelSize, const int64_t* stride, const int64_t* padding, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxUnpool1d_forward(const NNModule module, const Tensor tensor, const Tensor indices, const int64_t* outputSize);

EXPORT_API(NNModule) THSNN_MaxUnpool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, const int64_t* padding, const int paddingLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxUnpool2d_forward(const NNModule module, const Tensor tensor, const Tensor indices, const int64_t* outputSize, const int outputSizeLength);

EXPORT_API(NNModule) THSNN_MaxUnpool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, const int64_t* padding, const int paddingLength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MaxUnpool3d_forward(const NNModule module, const Tensor tensor, const Tensor indices, const int64_t* outputSize, const int outputSizeLength);

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

EXPORT_API(NNModule) THSNN_AvgPool1d_ctor(const int64_t* kernelSize, const int64_t* stride, const int64_t* padding, bool ceil_mode, bool count_include_pad, int64_t divisor_override, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, const int64_t* padding, const int paddingLength, bool ceil_mode, bool count_include_pad, int64_t divisor_override, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_AvgPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, const int64_t* padding, const int paddingLength, bool ceil_mode, bool count_include_pad, int64_t divisor_override, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AvgPool3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_LPPool1d_ctor(double norm_type, const int64_t* kernelSize, const int64_t* stride, bool ceil_mode, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LPPool1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_LPPool2d_ctor(double norm_type, const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength, bool ceil_mode, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LPPool2d_forward(const NNModule module, const Tensor tensor);

// Padding

EXPORT_API(NNModule) THSNN_ZeroPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ZeroPad2d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ZeroPad2d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_ConstantPad1d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ConstantPad1d_ctor_tuple(const double value, const int64_t padding_left, const int64_t padding_right, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ConstantPad1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ConstantPad2d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ConstantPad2d_ctor_tuple(const double value, const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ConstantPad2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ConstantPad3d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ConstantPad3d_ctor_tuple(const double value, const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, const int64_t padding_front, const int64_t padding_back, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ConstantPad3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_ReplicationPad1d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ReplicationPad1d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReplicationPad1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ReplicationPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ReplicationPad2d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReplicationPad2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ReplicationPad3d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ReplicationPad3d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, const int64_t padding_front, const int64_t padding_back, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReplicationPad3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_ReflectionPad1d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ReflectionPad1d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReflectionPad1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ReflectionPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ReflectionPad2d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReflectionPad2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ReflectionPad3d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_ReflectionPad3d_ctor_tuple(const int64_t padding_left, const int64_t padding_right, const int64_t padding_top, const int64_t padding_bottom, const int64_t padding_front, const int64_t padding_back, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ReflectionPad3d_forward(const NNModule module, const Tensor tensor);

// Convolution

EXPORT_API(NNModule) THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Conv1d_bias(const NNModule module);
EXPORT_API(void)     THSNN_Conv1d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_Conv1d_weight(const NNModule module);
EXPORT_API(void)     THSNN_Conv1d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(NNModule) THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_Conv2d_ctor_1(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelX, const int64_t kernelY, const int64_t strideX, const int64_t strideY, const int64_t paddingX, const int64_t paddingY, const int64_t dilationX, const int64_t dilationY, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Conv2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Conv2d_weight(const NNModule module);
EXPORT_API(void)     THSNN_Conv2d_set_weight(const NNModule module, const Tensor weight);
EXPORT_API(Tensor)   THSNN_Conv2d_bias(const NNModule module);
EXPORT_API(void)     THSNN_Conv2d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(NNModule) THSNN_Conv3d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
EXPORT_API(NNModule) THSNN_Conv3d_ctor_1(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelX, const int64_t kernelY, const int64_t kernelZ, const int64_t strideX, const int64_t strideY, const int64_t strideZ, const int64_t paddingX, const int64_t paddingY, const int64_t paddingZ, const int64_t dilationX, const int64_t dilationY, const int64_t dilationZ, const int64_t paddingMode, const int64_t groups, const bool bias, NNAnyModule* outAsAnyModule);
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

// Normalization

EXPORT_API(NNModule) THSNN_BatchNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BatchNorm1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_BatchNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BatchNorm2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_BatchNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BatchNorm3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor)   THSNN_BatchNorm1d_bias(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm1d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_BatchNorm1d_weight(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm1d_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_BatchNorm2d_bias(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm2d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_BatchNorm2d_weight(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm2d_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_BatchNorm3d_bias(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm3d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_BatchNorm3d_weight(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm3d_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(void)     THSNN_BatchNorm1d_reset_stats(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm2d_reset_stats(const NNModule module);
EXPORT_API(void)     THSNN_BatchNorm3d_reset_stats(const NNModule module);

EXPORT_API(Tensor)   THSNN_BatchNorm1d_get_mean(const NNModule module);
EXPORT_API(Tensor)   THSNN_BatchNorm2d_get_mean(const NNModule module);
EXPORT_API(Tensor)   THSNN_BatchNorm3d_get_mean(const NNModule module);

EXPORT_API(void)     THSNN_BatchNorm1d_set_mean(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_BatchNorm2d_set_mean(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_BatchNorm3d_set_mean(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_BatchNorm1d_get_var(const NNModule module);
EXPORT_API(Tensor)   THSNN_BatchNorm2d_get_var(const NNModule module);
EXPORT_API(Tensor)   THSNN_BatchNorm3d_get_var(const NNModule module);

EXPORT_API(void)     THSNN_BatchNorm1d_set_var(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_BatchNorm2d_set_var(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_BatchNorm3d_set_var(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_BatchNorm1d_get_batches(const NNModule module);
EXPORT_API(Tensor)   THSNN_BatchNorm2d_get_batches(const NNModule module);
EXPORT_API(Tensor)   THSNN_BatchNorm3d_get_batches(const NNModule module);

EXPORT_API(NNModule) THSNN_InstanceNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_InstanceNorm1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_InstanceNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_InstanceNorm2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_InstanceNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_InstanceNorm3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor)   THSNN_InstanceNorm1d_bias(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm1d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_InstanceNorm1d_weight(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm1d_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_InstanceNorm2d_bias(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm2d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_InstanceNorm2d_weight(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm2d_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_InstanceNorm3d_bias(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm3d_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_InstanceNorm3d_weight(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm3d_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(void)     THSNN_InstanceNorm1d_reset_stats(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm2d_reset_stats(const NNModule module);
EXPORT_API(void)     THSNN_InstanceNorm3d_reset_stats(const NNModule module);

EXPORT_API(Tensor)   THSNN_InstanceNorm1d_get_mean(const NNModule module);
EXPORT_API(Tensor)   THSNN_InstanceNorm2d_get_mean(const NNModule module);
EXPORT_API(Tensor)   THSNN_InstanceNorm3d_get_mean(const NNModule module);

EXPORT_API(void)     THSNN_InstanceNorm1d_set_mean(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_InstanceNorm2d_set_mean(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_InstanceNorm3d_set_mean(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_InstanceNorm1d_get_var(const NNModule module);
EXPORT_API(Tensor)   THSNN_InstanceNorm2d_get_var(const NNModule module);
EXPORT_API(Tensor)   THSNN_InstanceNorm3d_get_var(const NNModule module);

EXPORT_API(void)     THSNN_InstanceNorm1d_set_var(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_InstanceNorm2d_set_var(const NNModule module, const Tensor weight);
EXPORT_API(void)     THSNN_InstanceNorm3d_set_var(const NNModule module, const Tensor weight);

EXPORT_API(Tensor)   THSNN_InstanceNorm1d_get_batches(const NNModule module);
EXPORT_API(Tensor)   THSNN_InstanceNorm2d_get_batches(const NNModule module);
EXPORT_API(Tensor)   THSNN_InstanceNorm3d_get_batches(const NNModule module);



EXPORT_API(NNModule) THSNN_LayerNorm_ctor(const int64_t* norm_shape, const int64_t norm_shape_len, const double eps, const bool elementwise_affine, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LayerNorm_forward(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor)   THSNN_LayerNorm_bias(const NNModule module);
EXPORT_API(void)     THSNN_LayerNorm_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_LayerNorm_weight(const NNModule module);
EXPORT_API(void)     THSNN_LayerNorm_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(NNModule) THSNN_GroupNorm_ctor(const int64_t num_groups, const int64_t num_channels, const double eps, const bool affine, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_GroupNorm_forward(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor)   THSNN_GroupNorm_bias(const NNModule module);
EXPORT_API(void)     THSNN_GroupNorm_set_bias(const NNModule module, const Tensor bias);
EXPORT_API(Tensor)   THSNN_GroupNorm_weight(const NNModule module);
EXPORT_API(void)     THSNN_GroupNorm_set_weight(const NNModule module, const Tensor weight);

EXPORT_API(NNModule) THSNN_LocalResponseNorm_ctor(const int64_t size, const double alpha, const double beta, const double k, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LocalResponseNorm_forward(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor)   THSNN_batch_norm(const Tensor input, const Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool training, const double momentum, const double eps);
EXPORT_API(Tensor)   THSNN_group_norm(const Tensor input, int64_t num_groups, const Tensor weight, const Tensor bias, const double eps);
EXPORT_API(Tensor)   THSNN_instance_norm(const Tensor input, const Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool use_input_stats, const double momentum, const double eps);
EXPORT_API(Tensor)   THSNN_layer_norm(const Tensor input, const int64_t* normalized_shape, const int64_t normalized_shape_len, const Tensor weight, const Tensor bias, const double eps);
EXPORT_API(Tensor)   THSNN_local_response_norm(const Tensor input, const int64_t size, const double alpha, const double beta, const double k);

// Dropout

EXPORT_API(NNModule) THSNN_Dropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Dropout1d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout1d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Dropout2d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout2d_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Dropout3d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Dropout3d_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_AlphaDropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_AlphaDropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_FeatureAlphaDropout_ctor(double probability, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor) THSNN_dropout(const Tensor input, const double p, bool training, bool inplace);
EXPORT_API(Tensor) THSNN_dropout2d(const Tensor input, const double p, bool training, bool inplace);
EXPORT_API(Tensor) THSNN_dropout3d(const Tensor input, const double p, bool training, bool inplace);

EXPORT_API(Tensor) THSNN_alpha_dropout(const Tensor input, const double p, bool training, bool inplace);

EXPORT_API(Tensor) THSNN_feature_alpha_dropout(const Tensor input, const double p, bool training, bool inplace);

// Linear

EXPORT_API(NNModule) THSNN_Identity_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Identity_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool with_bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Linear_forward(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Linear_bias(const NNModule module);
EXPORT_API(void)     THSNN_Linear_set_bias(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Linear_weight(const NNModule module);
EXPORT_API(void)     THSNN_Linear_set_weight(const NNModule module, const Tensor tensor);

EXPORT_API(Tensor) THSNN_functional_linear(const Tensor input, const Tensor weights, const Tensor bias);
EXPORT_API(Tensor) THSNN_functional_bilinear(const Tensor input1, const Tensor input2, const Tensor weights, const Tensor bias);

EXPORT_API(NNModule) THSNN_Bilinear_ctor(const int64_t input_size_1, const int64_t input_size_2, const int64_t output_size, const bool with_bias, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Bilinear_forward(const NNModule module, const Tensor x1, const Tensor x2);
EXPORT_API(Tensor)   THSNN_Bilinear_bias(const NNModule module);
EXPORT_API(void)     THSNN_Bilinear_set_bias(const NNModule module, const Tensor tensor);
EXPORT_API(Tensor)   THSNN_Bilinear_weight(const NNModule module);
EXPORT_API(void)     THSNN_Bilinear_set_weight(const NNModule module, const Tensor tensor);

// Vision -- Modules

EXPORT_API(NNModule) THSNN_PixelShuffle_ctor(const int64_t upscale_factor, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_PixelShuffle_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_PixelUnshuffle_ctor(const int64_t downscale_factor, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_PixelUnshuffle_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Upsample_ctor(const int64_t* size, const int size_len, const double* scale_factor, const int scale_factor_len, const int8_t mode, const int8_t align_corners, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Upsample_forward(const NNModule module, const Tensor tensor);

// Vision -- Functions

EXPORT_API(Tensor) THSNN_pad(const Tensor input, const int64_t* pad, const int pad_length, const int8_t mode, const double value);
EXPORT_API(Tensor) THSNN_interpolate(const Tensor input, const int64_t* size, const int size_len, const double* scale_factor, const int scale_factor_len, const int8_t mode, const int8_t align_corners, const bool recompute_scale_factor, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor) THSNN_grid_sample(const Tensor input, const Tensor grid, const int8_t mode, const int8_t padding_mode, const int8_t align_corners);
EXPORT_API(Tensor) THSNN_affine_grid(const Tensor theta, const int64_t* size, const int size_len, const bool align_corners);

// Activation functions

EXPORT_API(NNModule) THSNN_CELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_CELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_ELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_GELU_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_GELU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_GLU_ctor(const int64_t dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_GLU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Hardshrink_ctor(const double lambda, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Hardshrink_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Hardtanh_ctor(const double min_val, const double max_val, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Hardtanh_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_LeakyReLU_ctor(const double negative_sloope, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_LeakyReLU_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Mish_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Mish_forward(const NNModule module, const Tensor tensor);
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
EXPORT_API(NNModule) THSNN_Softplus_ctor(const double beta, const double threshold, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Softplus_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Softshrink_ctor(const double lambda, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Softshrink_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Softsign_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Softsign_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Tanh_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Tanh_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Tanhshrink_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Tanhshrink_forward(const NNModule module, const Tensor tensor);
EXPORT_API(NNModule) THSNN_Threshold_ctor(const double threshold, const double value, const bool inplace, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Threshold_forward(const NNModule module, const Tensor tensor);

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

EXPORT_API(NNModule) THSNN_Flatten_ctor(const int64_t start_dim, const int64_t end_dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Flatten_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_Unflatten_ctor(const int64_t dim, const int64_t* shape, const int64_t shape_len, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Unflatten_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_CosineSimilarity_ctor(const int64_t dim, double eps, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_CosineSimilarity_forward(const NNModule module, const Tensor input1, const Tensor input2);

EXPORT_API(NNModule) THSNN_PairwiseDistance_ctor(double p, double eps, bool keep_dim, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_PairwiseDistance_forward(const NNModule module, const Tensor input1, const Tensor input2);

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
