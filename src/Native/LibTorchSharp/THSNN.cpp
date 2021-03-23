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
        auto mod = std::make_shared<torch::nn::ELUImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::ELUImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::CELUImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::CELUImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::ReLUImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::ReLUImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::RReLUImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::RReLUImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::ReLU6Impl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::ReLU6Impl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::LeakyReLUImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::LeakyReLUImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::SELUImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::SELUImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_SELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::SELU>()->forward(*tensor));
}

NNModule THSNN_Tanh_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = std::make_shared<torch::nn::TanhImpl>();

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::TanhImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Tanh_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Tanh>()->forward(*tensor));
}

NNModule THSNN_Sigmoid_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = std::make_shared<torch::nn::SigmoidImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::SigmoidImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Sigmoid_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Sigmoid>()->forward(*tensor));
}

NNModule THSNN_Softmax2d_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = std::make_shared<torch::nn::Softmax2dImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::Softmax2dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Softmax2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmax2d>()->forward(*tensor));
}

NNModule THSNN_GELU_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = std::make_shared<torch::nn::GELUImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::GELUImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_GELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::GELU>()->forward(*tensor));
}

NNModule THSNN_SiLU_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = std::make_shared<torch::nn::SiLUImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::SiLUImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::SoftmaxImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::SoftmaxImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::SoftminImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::SoftminImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }

        res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::LogSoftmaxImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::LogSoftmaxImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::AvgPool1dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AvgPool1dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AvgPool2dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AvgPool2dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AvgPool3dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AvgPool3dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
        auto mod = std::make_shared<torch::nn::AdaptiveAvgPool1dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AdaptiveAvgPool1dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AdaptiveAvgPool2dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AdaptiveAvgPool2dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AdaptiveAvgPool3dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AdaptiveAvgPool3dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AdaptiveMaxPool1dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AdaptiveMaxPool1dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AdaptiveMaxPool2dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AdaptiveMaxPool2dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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
    auto mod = std::make_shared<torch::nn::AdaptiveMaxPool3dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::AdaptiveMaxPool3dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
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

        auto mod = std::make_shared<torch::nn::MaxPool1dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::MaxPool1dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    )
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

        auto mod = std::make_shared<torch::nn::MaxPool2dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::MaxPool2dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    )
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

    auto mod = std::make_shared<torch::nn::MaxPool3dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::MaxPool3dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
    )
}

Tensor THSNN_MaxPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool3d>()->forward(*tensor));
}

NNModule THSNN_BatchNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm1dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

    auto mod = std::make_shared<torch::nn::BatchNorm1dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::BatchNorm1dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
    )
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

        auto mod = std::make_shared<torch::nn::BatchNorm2dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::BatchNorm2dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    )
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

    auto mod = std::make_shared<torch::nn::BatchNorm3dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::BatchNorm3dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
    )
}

Tensor THSNN_BatchNorm3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm3d>()->forward(*tensor));
}

NNModule THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LinearOptions(input_size, output_size);
        opts = opts.bias(bias);

        auto mod = std::make_shared<torch::nn::LinearImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::LinearImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Linear_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Linear>()->forward(*tensor));
}

Tensor THSNN_Linear_bias(const NNModule module)
{
    CATCH_TENSOR((*module)->as<torch::nn::Linear>()->bias);
}

void THSNN_Linear_set_bias(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::Linear>()->bias = *bias;
    )
}

Tensor THSNN_Linear_weight(const NNModule module)
{
    CATCH_TENSOR((*module)->as<torch::nn::Linear>()->weight);
}

void THSNN_Linear_set_weight(const NNModule module, const Tensor weight)
{
    CATCH(
        (*module)->as<torch::nn::Linear>()->weight = *weight;
    )
}

NNModule THSNN_Dropout_ctor(double probability, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::DropoutOptions(probability);
    auto mod = std::make_shared<torch::nn::DropoutImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::DropoutImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Dropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Dropout>()->forward(*tensor));
}

NNModule THSNN_FeatureAlphaDropout_ctor(double probability, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::FeatureAlphaDropoutOptions(probability);
    auto mod = std::make_shared<torch::nn::FeatureAlphaDropoutImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::FeatureAlphaDropoutImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::FeatureAlphaDropout>()->forward(*tensor));
}

NNModule THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t groups, const int64_t bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv1dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias);

        auto mod = std::make_shared<torch::nn::Conv1dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::Conv1dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Conv1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv1d>()->forward(*tensor));
}

NNModule THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t groups, const int64_t bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);

        auto mod = std::make_shared<torch::nn::Conv2dImpl>(opts);

        // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
        // a Module can only be boxed to AnyModule at the point its static type is known).
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::Conv2dImpl>(*mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Conv2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv2d>()->forward(*tensor));
}

NNModule THSNN_Conv3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t groups, const int64_t bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv3dOptions(inputChannel, outputChannel, kernelSize).stride(stride).padding(padding);

    auto mod = std::make_shared<torch::nn::Conv3dImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::Conv3dImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>(mod);
    );
}

Tensor THSNN_Conv3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv3d>()->forward(*tensor));
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

Tensor THSNN_binary_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::BinaryCrossEntropyFuncOptions();
        if (reduction == 0)
            opts = opts.reduction(torch::kNone);
        if (reduction == 1)
            opts = opts.reduction(torch::kMean);
        if (reduction == 2)
            opts = opts.reduction(torch::kSum);
        if (weight != NULL)
            opts = opts.weight(*weight);

        res = ResultTensor(torch::nn::functional::binary_cross_entropy(*input, *target, opts));
    )
}

Tensor THSNN_mse_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MSELossFuncOptions();
        if (reduction == 0)
            opts = opts.reduction(torch::kNone);
        if (reduction == 1)
            opts = opts.reduction(torch::kMean);
        if (reduction == 2)
            opts = opts.reduction(torch::kSum);

        res = ResultTensor(torch::nn::functional::mse_loss(*input, *target, opts));
     )
}

Tensor THSNN_nll_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::NLLLossFuncOptions();
        if (reduction == 0)
            opts = opts.reduction(torch::kNone);
        if (reduction == 1)
            opts = opts.reduction(torch::kMean);
        if (reduction == 2)
            opts = opts.reduction(torch::kSum);
        if (weight != NULL)
            opts = opts.weight(*weight);

        res = ResultTensor(torch::nn::functional::nll_loss(*input, *target, opts));
    )
}

Tensor THSNN_poisson_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction)
{
   CATCH_RETURN_Tensor(
       auto opts = torch::nn::functional::PoissonNLLLossFuncOptions().log_input(logInput).full(full).eps(eps);
       if (reduction == 0)
            opts = opts.reduction(torch::kNone);
       if (reduction == 1)
           opts = opts.reduction(torch::kMean);
       if (reduction == 2)
           opts = opts.reduction(torch::kSum);

       res = ResultTensor(torch::nn::functional::poisson_nll_loss(*input, *target, opts));
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

