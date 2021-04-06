// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>


void THSNN_Module_save(const NNModule module, const char* location)
{
    CATCH(
        auto output = torch::serialize::OutputArchive();

        output.save_to(location);
        (*module)->save(output);
    );
}

//NNModule THSNN_AnyModule_get(const NNAnyModule module)
//{
//	return new std::shared_ptr< torch::nn::Module>(&( (*module)->get<torch::nn::Module>()));
//}

void THSNN_Module_register_module(const NNModule module, const char* name, const NNModule submodule)
{
    CATCH(
        (*module)->register_module(name, *submodule);
    );
}

NNModule THSNN_Module_load(const char* location, const char* name)
{
    CATCH_RETURN_NNModule(
        auto module = new torch::nn::Module();
        auto input = torch::serialize::InputArchive();

        input.load_from(location);
        module->load(input);
        res = new std::shared_ptr<torch::nn::Module>(module);
    );
}

int THSNN_Module_has_parameter(const NNModule module, const char* name)
{
    CATCH_RETURN(int, 0, (*module)->named_parameters().contains(name));
}

Tensor THSNN_Module_get_parameter(const NNModule module, const char* name)
{
    CATCH_TENSOR(*(*module)->named_parameters().find(name));
}

void THSNN_Module_get_parameters(const NNModule module, Tensor* (*allocator1)(size_t length))
{
    auto parameters = (*module)->parameters();
    Tensor* result1 = allocator1(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result1[i] = ResultTensor(parameters[i]);
    }
}

void THSNN_Module_get_named_parameters(const NNModule module, Tensor* (*allocator1)(size_t length), const char** (*allocator2)(size_t length))
{
    auto parameters = (*module)->named_parameters();
    Tensor* result1 = allocator1(parameters.size());
    const char** result2 = allocator2(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result1[i] = ResultTensor(parameters[i].value());
        result2[i] = make_sharable_string(parameters[i].key());
    }
}

int THSNN_Module_is_training(NNModule module)
{
    return (*module)->is_training();
}

void THSNN_Module_train(NNModule module)
{
    (*module)->train();
}

void THSNN_Module_eval(NNModule module)
{
    (*module)->eval();
}

long THSNN_Module_children_size(const NNModule module)
{
    return (*module)->children().size();
}

NNModule THSNN_Module_child(const NNModule module, const int index)
{
    return new std::shared_ptr<torch::nn::Module>((*module)->children()[index]);
}

const char* THSNN_Module_name(const NNModule module)
{
    return make_sharable_string((*module)->name());
}

void THSNN_Module_zero_grad(const NNModule module)
{
    (*module)->zero_grad();
}

void THSNN_Module_to_device(NNModule module, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    (*module)->to(torch::Device(dev, index));
}


// Utilities

template <typename T>
Tensor get_weight(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight);
}

template <typename T>
void set_weight(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight = *weights;
    );
}

template <typename T>
Tensor get_bias(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias);
}

template <typename T>
void set_bias(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias = *bias;
    );
}

template<typename TImpl>
NNModule create_module(NNAnyModule* outAsAnyModule)
{
    auto mod = std::make_shared<TImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<TImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    return new std::shared_ptr<torch::nn::Module>(mod);
}

template<typename TImpl, typename TOptions>
NNModule create_module(const TOptions& opts, NNAnyModule* outAsAnyModule)
{
    auto mod = std::make_shared<TImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<TImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    return new std::shared_ptr<torch::nn::Module>(mod);
}

// Wrapper class used to enable .NET definitions ot new modules describing parameters and with delegates to implement forward function
class CustomModule : public torch::nn::Module
{
public:
    CustomModule(
        const char* name,
        const char** names,
        at::Tensor** parameters,
        const bool* require_grad,
        const int length,
        Tensor(*forward)(Tensor))
        : torch::nn::Module(name), _forward(forward)
    {
        for (int i = 0; i < length; i++)
        {
            register_parameter(names[i], *parameters[i], require_grad[i]);
        }

    }

    Tensor(*_forward)(Tensor);

    at::Tensor forward(at::Tensor input) {
        return *(*_forward)(&input);
    }

};

NNModule THSNN_custom_module(const char* name,
    const char** names,
    at::Tensor** parameters,
    const bool* require_grad,
    const int length,
    Tensor(*forward)(Tensor),
    NNAnyModule *outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = new CustomModule(name, names, parameters, require_grad, length, forward);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto modShared = new std::shared_ptr<CustomModule>(mod);
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<CustomModule>(*modShared));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>((torch::nn::Module*)mod);
    );
}

NNModule THSNN_ELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ELUOptions().alpha(alpha).inplace(inplace);
        res = create_module<torch::nn::ELUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ELU>()->forward(*tensor));
}

NNModule THSNN_CELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::CELUOptions().alpha(alpha).inplace(inplace);
        res = create_module<torch::nn::CELUImpl>(opts, outAsAnyModule); 
    );
}

Tensor THSNN_CELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::CELU>()->forward(*tensor));
}

NNModule THSNN_ReLU_ctor(bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReLUOptions(inplace);
        res = create_module<torch::nn::ReLUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ReLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReLU>()->forward(*tensor));
}

NNModule THSNN_RReLU_ctor(const double lower, const double upper, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(inplace);
        res = create_module<torch::nn::RReLUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_RReLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::RReLU>()->forward(*tensor));
}

NNModule THSNN_ReLU6_ctor(bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReLU6Options(inplace);
        res = create_module<torch::nn::ReLU6Impl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ReLU6_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReLU6>()->forward(*tensor));
}

NNModule THSNN_LeakyReLU_ctor(const double negative_sloope, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LeakyReLUOptions().negative_slope(negative_sloope).inplace(inplace);
        res = create_module<torch::nn::LeakyReLUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LeakyReLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LeakyReLU>()->forward(*tensor));
}

NNModule THSNN_SELU_ctor(bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SELUOptions(inplace);
        res = create_module<torch::nn::SELUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_SELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::SELU>()->forward(*tensor));
}

NNModule THSNN_Tanh_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::TanhImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Tanh_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Tanh>()->forward(*tensor));
}

NNModule THSNN_Sigmoid_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::SigmoidImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Sigmoid_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Sigmoid>()->forward(*tensor));
}

NNModule THSNN_Softmax2d_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::Softmax2dImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Softmax2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmax2d>()->forward(*tensor));
}

NNModule THSNN_GELU_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::GELUImpl>(outAsAnyModule);
    );
}

Tensor THSNN_GELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::GELU>()->forward(*tensor));
}

NNModule THSNN_SiLU_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::SiLUImpl>(outAsAnyModule);
    );
}

Tensor THSNN_SiLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::SiLU>()->forward(*tensor));
}

NNModule THSNN_Softmax_ctor(const int64_t dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SoftmaxOptions(dim);
        res = create_module<torch::nn::SoftmaxImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Softmax_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmax>()->forward(*tensor));
}

NNModule THSNN_Softmin_ctor(const int64_t dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SoftminOptions(dim);
        res = create_module<torch::nn::SoftminImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Softmin_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmin>()->forward(*tensor));
}

NNModule THSNN_LogSoftmax_ctor(int64_t dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LogSoftmaxOptions(dim);
        res = create_module<torch::nn::LogSoftmaxImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LogSoftmax_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LogSoftmax>()->forward(*tensor));
}

NNModule THSNN_AvgPool1d_ctor(const int64_t* kernelSize, const int64_t* stride,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AvgPool1dOptions(at::ArrayRef<int64_t>(kernelSize, 1));
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, 1));
        res = create_module<torch::nn::AvgPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AvgPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AvgPool1d>()->forward(*tensor));
}

NNModule THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AvgPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));
        res = create_module<torch::nn::AvgPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AvgPool2d>()->forward(*tensor));
}

NNModule THSNN_AvgPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AvgPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    if (stride)
        opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));
        res = create_module<torch::nn::AvgPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AvgPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AvgPool3d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool1d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveAvgPool1dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        res = create_module<torch::nn::AdaptiveAvgPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveAvgPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool1d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveAvgPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        res = create_module<torch::nn::AdaptiveAvgPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveAvgPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool2d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveAvgPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        res = create_module<torch::nn::AdaptiveAvgPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveAvgPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool3d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveMaxPool1d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveMaxPool1dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        res = create_module<torch::nn::AdaptiveMaxPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveMaxPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveMaxPool1d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveMaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveMaxPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        res = create_module<torch::nn::AdaptiveMaxPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveMaxPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveMaxPool2d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveMaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveMaxPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        res = create_module<torch::nn::AdaptiveMaxPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveMaxPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveMaxPool3d>()->forward(*tensor));
}

NNModule THSNN_MaxPool1d_ctor(const int64_t* kernelSize, const int64_t* stride,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MaxPool1dOptions(at::ArrayRef<int64_t>(kernelSize, 1));
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, 1));

        res = create_module<torch::nn::MaxPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MaxPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool1d>()->forward(*tensor));
}

NNModule THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MaxPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));

        res = create_module<torch::nn::MaxPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool2d>()->forward(*tensor));
}

NNModule THSNN_MaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MaxPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));

        res = create_module<torch::nn::MaxPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MaxPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool3d>()->forward(*tensor));
}


NNModule THSNN_ZeroPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ZeroPad2dOptions(padding);
        res = create_module<torch::nn::ZeroPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ZeroPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ZeroPad2d>()->forward(*tensor));
}

NNModule THSNN_ConstantPad1d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConstantPad1dOptions(padding, value);
        res = create_module<torch::nn::ConstantPad1dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ConstantPad1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConstantPad1d>()->forward(*tensor));
}

NNModule THSNN_ConstantPad2d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConstantPad2dOptions(padding, value);
        res = create_module<torch::nn::ConstantPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ConstantPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConstantPad2d>()->forward(*tensor));
}

NNModule THSNN_ConstantPad3d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConstantPad3dOptions(padding, value);
        res = create_module<torch::nn::ConstantPad3dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ConstantPad3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConstantPad3d>()->forward(*tensor));
}

NNModule THSNN_ReplicationPad1d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReplicationPad1dOptions(padding);
        res = create_module<torch::nn::ReplicationPad1dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReplicationPad1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReplicationPad1d>()->forward(*tensor));
}

NNModule THSNN_ReplicationPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReplicationPad2dOptions(padding);
        res = create_module<torch::nn::ReplicationPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReplicationPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReplicationPad2d>()->forward(*tensor));
}

NNModule THSNN_ReplicationPad3d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReplicationPad3dOptions(padding);
        res = create_module<torch::nn::ReplicationPad3dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReplicationPad3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReplicationPad3d>()->forward(*tensor));
}

NNModule THSNN_ReflectionPad1d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReflectionPad1dOptions(padding);
        res = create_module<torch::nn::ReflectionPad1dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReflectionPad1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReflectionPad1d>()->forward(*tensor));
}

NNModule THSNN_ReflectionPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReflectionPad2dOptions(padding);
        res = create_module<torch::nn::ReflectionPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReflectionPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReflectionPad2d>()->forward(*tensor));
}

NNModule THSNN_BatchNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm1dOptions(features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);

        res = create_module<torch::nn::BatchNorm1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BatchNorm1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm1d>()->forward(*tensor));
}

NNModule THSNN_BatchNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm2dOptions(features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);

        res = create_module<torch::nn::BatchNorm2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BatchNorm2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm2d>()->forward(*tensor));
}

NNModule THSNN_BatchNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm3dOptions(features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);

        res = create_module<torch::nn::BatchNorm3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BatchNorm3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm3d>()->forward(*tensor));
}

NNModule THSNN_InstanceNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::InstanceNorm1dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

    res = create_module<torch::nn::InstanceNorm1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_InstanceNorm1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::InstanceNorm1d>()->forward(*tensor));
}

NNModule THSNN_InstanceNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::InstanceNorm2dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

    res = create_module<torch::nn::InstanceNorm2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_InstanceNorm2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::InstanceNorm2d>()->forward(*tensor));
}

NNModule THSNN_InstanceNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::InstanceNorm3dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

    res = create_module<torch::nn::InstanceNorm3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_InstanceNorm3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::InstanceNorm3d>()->forward(*tensor));
}

NNModule THSNN_LayerNorm_ctor(const int64_t* norm_shape, const int64_t norm_shape_len, const double eps, const bool elementwise_affine, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        std::vector<int64_t> normalized_shape;
        for (int64_t i = 0; i < norm_shape_len; ++i)
        {
            normalized_shape.push_back(norm_shape[i]);
        }
        auto opts = torch::nn::LayerNormOptions(normalized_shape).eps(eps).elementwise_affine(elementwise_affine);
        res = create_module<torch::nn::LayerNormImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LayerNorm_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LayerNorm>()->forward(*tensor));
}

NNModule THSNN_GroupNorm_ctor(const int64_t num_groups, const int64_t num_channels, const double eps, const bool affine, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::GroupNormOptions(num_groups, num_channels).eps(eps).affine(affine);
        res = create_module<torch::nn::GroupNormImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_GroupNorm_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::GroupNorm>()->forward(*tensor));
}

NNModule THSNN_LocalResponseNorm_ctor(const int64_t size, const double alpha, const double beta, const double k, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LocalResponseNormOptions(size)
            .alpha(alpha)
            .beta(beta)
            .k(k);
        res = create_module<torch::nn::LocalResponseNormImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LocalResponseNorm_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LocalResponseNorm>()->forward(*tensor));
}

NNModule THSNN_Identity_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::IdentityImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Identity_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Identity>()->forward(*tensor));
}

NNModule THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LinearOptions(input_size, output_size).bias(bias);
        res = create_module<torch::nn::LinearImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Linear_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Linear>()->forward(*tensor));
}

Tensor THSNN_Linear_bias(const NNModule module)
{
    return get_bias<torch::nn::Linear>(module);
}

void THSNN_Linear_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Linear>(module, bias);
}

Tensor THSNN_Linear_weight(const NNModule module)
{
    return get_weight<torch::nn::Linear>(module);
}

void THSNN_Linear_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Linear>(module, weight);
}

NNModule THSNN_Dropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::DropoutOptions(probability).inplace(inplace);
        res = create_module<torch::nn::DropoutImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout>()->forward(*tensor));
}

NNModule THSNN_AlphaDropout_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AlphaDropoutOptions(probability).inplace(inplace);
    res = create_module<torch::nn::AlphaDropoutImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AlphaDropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AlphaDropout>()->forward(*tensor));
}

NNModule THSNN_Dropout2d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Dropout2dOptions(probability).inplace(inplace);
        res = create_module<torch::nn::Dropout2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout2d>()->forward(*tensor));
}

NNModule THSNN_Dropout3d_ctor(double probability, bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Dropout3dOptions(probability).inplace(inplace);
        res = create_module<torch::nn::Dropout3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Dropout3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout3d>()->forward(*tensor));
}

NNModule THSNN_FeatureAlphaDropout_ctor(double probability, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::FeatureAlphaDropoutOptions(probability);
        res = create_module<torch::nn::FeatureAlphaDropoutImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::FeatureAlphaDropout>()->forward(*tensor));
}

NNModule THSNN_Embedding_ctor(const int64_t num_embeddings, const int64_t embedding_dims,
    const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type,
    const bool scale_grad_by_freq, const bool sparse,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::EmbeddingOptions(num_embeddings, embedding_dims)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .sparse(sparse);

        if (has_pi)
            opts.padding_idx(padding_idx);
        if (has_mn)
            opts.max_norm(max_norm);

        res = create_module<torch::nn::EmbeddingImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_Embedding_from_pretrained(const Tensor embeddings, const bool freeze,
    const int64_t padding_idx, bool has_pi, const double max_norm, const bool has_mn, const double norm_type,
    const bool scale_grad_by_freq, const bool sparse,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto rows = embeddings->size(0);
        auto cols = embeddings->size(1);

        auto opts = torch::nn::EmbeddingOptions(rows, cols)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .sparse(sparse);

        if (has_pi)
            opts.padding_idx(padding_idx);
        if (has_mn)
            opts.max_norm(max_norm);

        // Can't use the template function here -- custom logic.
        auto mod = std::make_shared<torch::nn::EmbeddingImpl>(opts);
        mod->weight = *embeddings;
        mod->weight.set_requires_grad(!freeze);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::EmbeddingImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Embedding_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Embedding>()->forward(*tensor));
}

Tensor THSNN_Embedding_weight(const NNModule module)
{
    return get_weight<torch::nn::Embedding>(module);
}

void THSNN_Embedding_set_weight(const NNModule module, const Tensor weights)
{
    set_weight<torch::nn::Embedding>(module, weights);
}

template<typename T>
void ApplyPaddingMode(T& opts, const int64_t padding)
{
    if (padding == 0)
        opts = opts.padding_mode(torch::kZeros);
    if (padding == 1)
        opts = opts.padding_mode(torch::kReflect);
    if (padding == 2)
        opts = opts.padding_mode(torch::kReplicate);
    if (padding == 3)
        opts = opts.padding_mode(torch::kCircular);
}


NNModule THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv1dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv1d>()->forward(*tensor));
}

Tensor THSNN_Conv1d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv1d>(module);
}

void THSNN_Conv1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv1d>(module, bias);
}

Tensor THSNN_Conv1d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv1d>(module);
}

void THSNN_Conv1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv1d>(module, weight);
}

NNModule THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv2d>()->forward(*tensor));
}

Tensor THSNN_Conv2d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv2d>(module);
}

void THSNN_Conv2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv2d>(module, bias);
}

Tensor THSNN_Conv2d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv2d>(module);
}

void THSNN_Conv2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv2d>(module, weight);
}

NNModule THSNN_Conv3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv3dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv3d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv3d>(module);
}

void THSNN_Conv3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv3d>(module, bias);
}

Tensor THSNN_Conv3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv3d>()->forward(*tensor));
}

Tensor THSNN_Conv3d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv3d>(module);
}

void THSNN_Conv3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv3d>(module, weight);
}


NNModule THSNN_ConvTranspose1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose1dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::ConvTranspose1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose1d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose1d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose1d>(module);
}

void THSNN_ConvTranspose1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose1d>(module, bias);
}

Tensor THSNN_ConvTranspose1d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose1d>(module);
}

void THSNN_ConvTranspose1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose1d>(module, weight);
}

NNModule THSNN_ConvTranspose2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose2dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::ConvTranspose2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose2d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose2d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose2d>(module);
}

void THSNN_ConvTranspose2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose2d>(module, bias);
}

Tensor THSNN_ConvTranspose2d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose2d>(module);
}

void THSNN_ConvTranspose2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose2d>(module, weight);
}

NNModule THSNN_ConvTranspose3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose3dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias)
            .output_padding(output_padding);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::ConvTranspose3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose3d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose3d>(module);
}

void THSNN_ConvTranspose3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose3d>(module, bias);
}

Tensor THSNN_ConvTranspose3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose3d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose3d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose3d>(module);
}

void THSNN_ConvTranspose3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose3d>(module, weight);
}

template<typename T>
void ApplyTransformerActivation(T& opts, const int64_t activation)
{
    if (activation == 0)
        opts = opts.activation(torch::kReLU);
    if (activation == 1)
        opts = opts.activation(torch::kGELU);
}

template<typename T>
void ApplyRnnActivation(T& opts, const int64_t activation)
{
    if (activation == 0)
        opts = opts.nonlinearity(torch::kReLU);
    if (activation == 1)
        opts = opts.nonlinearity(torch::kTanh);
}

NNModule THSNN_Transformer_ctor(const int64_t d_model, const int64_t nhead, const int64_t num_encoder_layers, const int64_t num_decoder_layers, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TransformerOptions(d_model, nhead)
            .num_encoder_layers(num_encoder_layers)
            .num_decoder_layers(num_decoder_layers)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        ApplyTransformerActivation(opts, activation);

        res = create_module<torch::nn::TransformerImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_Transformer_forward(const NNModule module, const Tensor src, const Tensor tgt, const Tensor src_mask, const Tensor tgt_mask, const Tensor memory_mask, const Tensor src_key_padding_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::Transformer>()->forward(
        *src,
        *tgt,
        (src_mask ? *src_mask : at::Tensor()),
        (tgt_mask ? *tgt_mask : at::Tensor()),
        (memory_mask ? *memory_mask : at::Tensor()),
        (src_key_padding_mask ? *src_key_padding_mask : at::Tensor()),
        (tgt_key_padding_mask ? *tgt_key_padding_mask : at::Tensor()),
        (memory_key_padding_mask ? *memory_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerEncoderLayer_ctor(const int64_t d_model, const int64_t nhead, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        ApplyTransformerActivation(opts, activation);

        res = create_module<torch::nn::TransformerEncoderLayerImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerEncoderLayer_forward(const NNModule module, const Tensor src, const Tensor src_mask, const Tensor src_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerEncoderLayer>()->forward(
        *src,
        (src_mask ? *src_mask : at::Tensor()),
        (src_key_padding_mask ? *src_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerDecoderLayer_ctor(const int64_t d_model, const int64_t nhead, const int64_t dim_feedforward, const double dropout, const int64_t activation, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TransformerDecoderLayerOptions(d_model, nhead)
            .dim_feedforward(dim_feedforward)
            .dropout(dropout);
        ApplyTransformerActivation(opts, activation);

        res = create_module<torch::nn::TransformerDecoderLayerImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerDecoderLayer_forward(const NNModule module, const Tensor tgt, const Tensor memory, const Tensor tgt_mask, const Tensor memory_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerDecoderLayer>()->forward(
        *tgt,
        *memory,
        (tgt_mask ? *tgt_mask : at::Tensor()),
        (memory_mask ? *memory_mask : at::Tensor()),
        (tgt_key_padding_mask ? *tgt_key_padding_mask : at::Tensor()),
        (memory_key_padding_mask ? *memory_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerEncoder_ctor(const NNModule encoder_layer, const int64_t num_layers, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto enc = (*encoder_layer)->as<torch::nn::TransformerEncoderLayer>();
        auto opts = torch::nn::TransformerEncoderOptions(torch::nn::TransformerEncoderLayer(*enc), num_layers);

        res = create_module<torch::nn::TransformerEncoderImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerEncoder_forward(const NNModule module, const Tensor src, const Tensor src_mask, const Tensor src_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerEncoder>()->forward(
        *src,
        (src_mask ? *src_mask : at::Tensor()),
        (src_key_padding_mask ? *src_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_TransformerDecoder_ctor(const NNModule decoder_layer, const int64_t num_layers, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto dec = (*decoder_layer)->as<torch::nn::TransformerDecoderLayer>();
        auto opts = torch::nn::TransformerDecoderOptions(torch::nn::TransformerDecoderLayer(*dec), num_layers);

        res = create_module<torch::nn::TransformerDecoderImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_TransformerDecoder_forward(const NNModule module, const Tensor tgt, const Tensor memory, const Tensor tgt_mask, const Tensor memory_mask, const Tensor tgt_key_padding_mask, const Tensor memory_key_padding_mask)
{
    CATCH_TENSOR((*module)->as<torch::nn::TransformerDecoder>()->forward(
        *tgt,
        *memory,
        (tgt_mask ? *tgt_mask : at::Tensor()),
        (memory_mask ? *memory_mask : at::Tensor()),
        (tgt_key_padding_mask ? *tgt_key_padding_mask : at::Tensor()),
        (memory_key_padding_mask ? *memory_key_padding_mask : at::Tensor()))
    );
}

NNModule THSNN_Flatten_ctor(const int64_t start_dim, const int64_t end_dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::FlattenOptions()
            .start_dim(start_dim)
            .end_dim(end_dim);

        res = create_module<torch::nn::FlattenImpl>(opts, outAsAnyModule);
    );
}
Tensor   THSNN_Flatten_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Flatten>()->forward(*tensor));
}

NNModule THSNN_Unflatten_ctor(const int64_t dim, const int64_t* shape, const int64_t shape_len, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        std::vector<int64_t> sizes;
    for (int64_t i = 0; i < shape_len; ++i)
    {
        sizes.push_back(shape[i]);
    }
    auto opts = torch::nn::UnflattenOptions(dim, sizes);
    res = create_module<torch::nn::UnflattenImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Unflatten_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Unflatten>()->forward(*tensor));
}



NNModule THSNN_RNN_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const int64_t nonlinearity, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::RNNOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batchFirst)
            .dropout(dropout)
            .bidirectional(bidirectional);

        ApplyRnnActivation(opts, nonlinearity);

        res = create_module<torch::nn::RNNImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_RNN_forward(const NNModule module, const Tensor input1, const Tensor input2, Tensor* h_n)
{
    Tensor output;
    CATCH(
        auto result = (*module)->as<torch::nn::RNN>()->forward(*input1, (input2 ? *input2 : at::Tensor()));
        output = new torch::Tensor(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}

NNModule THSNN_GRU_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::GRUOptions(input_size, hidden_size)
            .num_layers(num_layers)
            .bias(bias)
            .batch_first(batchFirst)
            .dropout(dropout)
            .bidirectional(bidirectional);

        res = create_module<torch::nn::GRUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_GRU_forward(const NNModule module, const Tensor input1, const Tensor input2, Tensor* h_n)
{
    Tensor output;
    CATCH(
        auto result = (*module)->as<torch::nn::GRU>()->forward(*input1, (input2 ? *input2 : at::Tensor()));
        output = new torch::Tensor(std::get<0>(result));
        *h_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}

NNModule THSNN_LSTM_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool bias, const bool batchFirst, const double dropout, const bool bidirectional, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LSTMOptions(input_size, hidden_size)
        .num_layers(num_layers)
        .bias(bias)
        .batch_first(batchFirst)
        .dropout(dropout)
        .bidirectional(bidirectional);

        res = create_module<torch::nn::LSTMImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LSTM_forward(const NNModule module, const Tensor input1, const Tensor h0, const Tensor c0, Tensor* h_n, Tensor* c_n)
{
    const std::tuple<at::Tensor, at::Tensor>& second_arg = (h0 == nullptr || c0 == nullptr) ? std::make_tuple(at::Tensor(), at::Tensor()) : std::make_tuple(*h0, *c0);

    Tensor output;
    CATCH(
        auto result = (*module)->as<torch::nn::LSTM>()->forward(*input1, second_arg);
    output = new torch::Tensor(std::get<0>(result));
    *h_n = new torch::Tensor(std::get<0>(std::get<1>(result)));
    *c_n = new torch::Tensor(std::get<1>(std::get<1>(result)));
    );
    return output;
}

NNModule THSNN_RNNCell_ctor(const int64_t input_size, const int64_t hidden_size, const int64_t nonlinearity, const bool bias, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::RNNCellOptions(input_size, hidden_size)
            .bias(bias);

        ApplyRnnActivation(opts, nonlinearity);

        res = create_module<torch::nn::RNNCellImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_RNNCell_forward(const NNModule module, const Tensor input1, const Tensor h0)
{
    CATCH_TENSOR((*module)->as<torch::nn::RNNCell>()->forward(*input1, (h0 ? *h0 : at::Tensor())));
}

NNModule THSNN_GRUCell_ctor(const int64_t input_size, const int64_t hidden_size, const bool bias, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::GRUCellOptions(input_size, hidden_size)
        .bias(bias);

        res = create_module<torch::nn::GRUCellImpl>(opts, outAsAnyModule);
    );
}

Tensor  THSNN_GRUCell_forward(const NNModule module, const Tensor input1, const Tensor h0)
{
    CATCH_TENSOR((*module)->as<torch::nn::GRUCell>()->forward(*input1, (h0 ? *h0 : at::Tensor())));
}

NNModule THSNN_LSTMCell_ctor(const int64_t input_size, const int64_t hidden_size, const bool bias, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LSTMCellOptions(input_size, hidden_size)
        .bias(bias);

        res = create_module<torch::nn::LSTMCellImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LSTMCell_forward(const NNModule module, const Tensor input1, const Tensor h0, const Tensor c0, Tensor* c_n)
{
    const std::tuple<at::Tensor, at::Tensor>& second_arg = (h0 == nullptr || c0 == nullptr) ? std::make_tuple(at::Tensor(), at::Tensor()) : std::make_tuple(*h0, *c0);

    Tensor output;
    CATCH(
        auto result = (*module)->as<torch::nn::LSTMCell>()->forward(*input1, second_arg);
        output = new torch::Tensor(std::get<0>(result));
        *c_n = new torch::Tensor(std::get<1>(result));
    );
    return output;
}


NNModule THSNN_Sequential_ctor( /* NNAnyModule *submodules, const int length */ )
{
    //std::vector<torch::nn::NamedAnyModule> modules;
    //for (int i = 0; i < length; i++)
    //{
    //	modules.push_back(*(*submodules[i])->as<torch::nn::NamedAnyModule>());
    //}

    auto mod = std::make_shared<torch::nn::SequentialImpl>( /* std::begin(modules), std::end(modules) */ );
    return new std::shared_ptr<torch::nn::Module>(mod);
}

void THSNN_Sequential_push_back(const NNModule module, const char *name, const NNAnyModule submodule)
{
    CATCH (
        (*module)->as<torch::nn::Sequential>()->push_back(name, *(*submodule));
    )
}

Tensor THSNN_Sequential_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Sequential>()->forward(*tensor));
}

void THSNN_Optimizer_zero_grad(const Optimizer optimizer)
{
    (*optimizer)->zero_grad();
}

void THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length))
{
    auto parameters = (*optimizer)->parameters();
    Tensor * result = allocator(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result[i] = ResultTensor(parameters[i]);
    }
}

template<typename T>
void ApplyReduction(T& opts, const int64_t reduction)
{
    if (reduction == 0)
        opts = opts.reduction(torch::kNone);
    if (reduction == 1)
        opts = opts.reduction(torch::kMean);
    if (reduction == 2)
        opts = opts.reduction(torch::kSum);
}

Tensor THSNN_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t ignore_index, const bool has_ii, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::CrossEntropyFuncOptions();
        ApplyReduction(opts, reduction);
        if (has_ii)
            opts = opts.ignore_index(ignore_index);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = ResultTensor(torch::nn::functional::cross_entropy(*input, *target, opts));
    )
}

Tensor THSNN_binary_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::BinaryCrossEntropyFuncOptions();
        ApplyReduction(opts, reduction);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = ResultTensor(torch::nn::functional::binary_cross_entropy(*input, *target, opts));
    )
}

Tensor THSNN_binary_cross_entropy_with_logits(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction, const Tensor pos_weights_wrapper)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::BCEWithLogitsLossOptions();
        ApplyReduction(opts, reduction);
        if (pos_weights_wrapper != nullptr)
            opts = opts.pos_weight(*pos_weights_wrapper);
        if (weight != nullptr)
            opts = opts.weight(*weight);
        res = ResultTensor(torch::nn::functional::binary_cross_entropy_with_logits(*input, *target, opts));
    )
}

Tensor THSNN_l1_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MSELossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::mse_loss(*input, *target, opts));
     )
}

Tensor THSNN_mse_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MSELossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::mse_loss(*input, *target, opts));
    )
}

Tensor THSNN_nll_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::NLLLossFuncOptions();
        ApplyReduction(opts, reduction);
        if (weight != NULL)
            opts = opts.weight(*weight);

        res = ResultTensor(torch::nn::functional::nll_loss(*input, *target, opts));
    )
}

Tensor THSNN_poisson_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction)
{
   CATCH_RETURN_Tensor(
       auto opts = torch::nn::functional::PoissonNLLLossFuncOptions().log_input(logInput).full(full).eps(eps);
       ApplyReduction(opts, reduction);

       res = ResultTensor(torch::nn::functional::poisson_nll_loss(*input, *target, opts));
    )
}

Tensor THSNN_kl_div_loss(const Tensor input, const Tensor target, const int64_t reduction, const bool log_target)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::KLDivFuncOptions().log_target(log_target);
        ApplyReduction(opts, reduction);
    
        res = ResultTensor(torch::nn::functional::kl_div(*input, *target, opts));
    )
}

Tensor THSNN_smooth_l1_loss(const Tensor input, const Tensor target, const int64_t reduction, const double beta)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::SmoothL1LossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::smooth_l1_loss(*input, *target, opts));
    )
}

Tensor THSNN_soft_margin_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::SoftMarginLossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::soft_margin_loss(*input, *target, opts));
    )
}


Optimizer THSNN_Adagrad_ctor(const Tensor* parameters, const int length, const double learning_rate, const double lr_decay, const double weight_decay, const double initial_accumulator_value, const double eps)
{
    auto params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::AdagradOptions(learning_rate)
        .lr_decay(lr_decay)
        .weight_decay(weight_decay)
        .initial_accumulator_value(initial_accumulator_value)
        .eps(eps);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adagrad>(torch::optim::Adagrad(params, options)));
}

Optimizer THSNN_Adam_ctor(const Tensor* parameters, const int length, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::AdamOptions(learning_rate)
        .betas(std::make_tuple(beta1, beta2))
        .eps(eps)
        .weight_decay(weight_decay)
        .amsgrad(amsgrad);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, options)));
}

Optimizer THSNN_AdamW_ctor(const Tensor* parameters, const int length, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::AdamWOptions(learning_rate)
        .betas(std::make_tuple(beta1, beta2))
        .eps(eps)
        .weight_decay(weight_decay)
        .amsgrad(amsgrad);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::AdamW>(torch::optim::AdamW(params, options)));
}

Optimizer THSNN_RMSprop_ctor(const Tensor* parameters, const int length, const double learning_rate, const double alpha, const double eps, const double weight_decay, const double momentum, const bool centered)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);

    auto options = torch::optim::RMSpropOptions(learning_rate)
        .alpha(alpha)
        .eps(eps)
        .weight_decay(weight_decay)
        .momentum(momentum)
        .centered(centered);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::RMSprop>(torch::optim::RMSprop(params, options)));
}

Optimizer THSNN_SGD_ctor(const Tensor* parameters, const int length, const double learning_rate, const double momentum, const double dampening, const double weight_decay, const bool nesterov)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto opts = torch::optim::SGDOptions(learning_rate)
        .momentum(momentum)
        .dampening(dampening)
        .weight_decay(weight_decay)
        .nesterov(nesterov);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::SGD>(torch::optim::SGD(params, opts)));
}

void THSNN_Optimizer_step(const Optimizer optimizer)
{
    (*optimizer)->step();
}

void THSNN_initUniform(Tensor tensor, double low, double high)
{
    torch::nn::init::uniform_(*tensor, low, high);
}

// ########## To remove when updating to libtorch > 1.0.1 ############
enum class Nonlinearity {
    Linear,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose1D,
    ConvTranspose2D,
    ConvTranspose3D,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU
};

enum class FanMode { FanIn, FanOut };

struct Fan {
    explicit Fan(torch::Tensor& tensor) {
        const auto dimensions = tensor.ndimension();
        TORCH_CHECK(
            dimensions >= 2,
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
        if (dimensions == 2) {
            in = tensor.size(1);
            out = tensor.size(0);
        }
        else {
            in = tensor.size(1) * tensor[0][0].numel();
            out = tensor.size(0) * tensor[0][0].numel();
        }
    }
    int64_t in;
    int64_t out;
};

double calculate_gain(Nonlinearity nonlinearity, double param) {
    if (nonlinearity == Nonlinearity::Tanh) {
        return 5.0 / 3.0;
    }
    else if (nonlinearity == Nonlinearity::ReLU) {
        return std::sqrt(2.0);
    }
    else if (nonlinearity == Nonlinearity::LeakyReLU) {
        return std::sqrt(2.0 / (1 + pow(param, 2)));
    }

    return 1.0;
}

double calculate_kaiming_std(
    Tensor tensor,
    double a,
    FanMode mode,
    Nonlinearity nonlinearity) {
    torch::NoGradGuard guard;
    Fan fan((*tensor));
    const auto gain = calculate_gain(nonlinearity, a);
    double std = 0.0;
    if (mode == FanMode::FanIn) {
        std = gain / std::sqrt(fan.in);
    }
    else {
        std = gain / std::sqrt(fan.out);
    }
    return std;
}

// ######################################################

void THSNN_initKaimingUniform(Tensor tensor, double a)
{
    //torch::nn::init::kaiming_uniform_(*tensor, a);
    // Since this is not available in PyTorch 1.0.1 will just used the original code for the moment
    auto std = calculate_kaiming_std(tensor, a, FanMode::FanIn, Nonlinearity::LeakyReLU);
    // Calculate uniform bounds from standard deviation
    const auto bound = std::sqrt(3.0) * std;
    tensor->uniform_(-bound, bound);
}

void THSNN_Optimizer_dispose(const Optimizer optimizer)
{
    delete optimizer; // NOTE: this reduces the ref count on the shared_ptr
}

void THSNN_Module_dispose(const NNModule module)
{
    delete module; // NOTE: this only deletes the shared_ptr
}

void THSNN_AnyModule_dispose(const NNAnyModule module)
{
    delete module; // NOTE: this only deletes the shared_ptr
}

Tensor THSNN_one_hot(const Tensor self, const int64_t num_classes)
{
    CATCH_RETURN_Tensor(
        res = ResultTensor(torch::nn::functional::one_hot(*self, num_classes));
    )
}
