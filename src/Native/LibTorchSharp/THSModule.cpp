// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

// General Module functions

int THSNN_Module_is_training(NNModule module)
{
    return (*module)->is_training();
}

void THSNN_Module_train(NNModule module, bool on)
{
    (*module)->train(on);
}

const char* THSNN_Module_name(const NNModule module)
{
    return make_sharable_string((*module)->name());
}

void THSNN_Module_zero_grad(const NNModule module, bool set_to_none)
{
    (*module)->zero_grad(set_to_none);
}

void THSNN_Module_to_device(NNModule module, int64_t device, int64_t index, const bool non_blocking)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;
    (*module)->to(torch::Device(dev, index), non_blocking);
}

void THSNN_Module_to_dtype(NNModule module, int8_t dtype, const bool non_blocking)
{
    (*module)->to((at::ScalarType)dtype, non_blocking);
}

void THSNN_Module_to_device_dtype(NNModule module, int8_t dtype, int64_t device, int64_t index, const bool non_blocking)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;
    (*module)->to(torch::Device(dev, index), (at::ScalarType)dtype, non_blocking);
}

void THSNN_Module_dispose(const NNModule module)
{
    delete module; // NOTE: this only deletes the shared_ptr
}

void THSNN_AnyModule_dispose(const NNAnyModule module)
{
    delete module; // NOTE: this only deletes the shared_ptr
}

//NNModule THSNN_AnyModule_get(const NNAnyModule module)
//{
//	return new std::shared_ptr< torch::nn::Module>(&( (*module)->get<torch::nn::Module>()));
//}

// Sub-module handling, parameters, etc.

void THSNN_Module_register_module(const NNModule module, const char* name, const NNModule submodule)
{
    CATCH(
        (*module)->register_module(name, *submodule);
    );
}

void THSNN_Module_register_parameter(const NNModule module, const char* name, const Tensor tensor, bool requires_grad)
{
    CATCH(
        (*module)->register_parameter(name, (tensor == nullptr) ? at::Tensor() : *tensor, requires_grad);
    );
}

void THSNN_Module_register_buffer(const NNModule module, const char* name, const Tensor tensor)
{
    CATCH(
        (*module)->register_buffer(name, *tensor);
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

void THSNN_Module_get_parameters(const NNModule module, Tensor* (*allocator1)(size_t length), bool recurse)
{
    auto parameters = (*module)->parameters(recurse);
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

void THSNN_Module_get_named_buffers(const NNModule module, Tensor* (*allocator1)(size_t length), const char** (*allocator2)(size_t length))
{
    auto buffers = (*module)->named_buffers();
    Tensor* result1 = allocator1(buffers.size());
    const char** result2 = allocator2(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++)
    {
        result1[i] = ResultTensor(buffers[i].value());
        result2[i] = make_sharable_string(buffers[i].key());
    }
}

void THSNN_Module_get_named_children(const NNModule module, NNModule* (*allocator1)(size_t length), const char** (*allocator2)(size_t length))
{
    auto buffers = (*module)->named_children();
    NNModule* result1 = allocator1(buffers.size());
    const char** result2 = allocator2(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++)
    {
        result1[i] = new std::shared_ptr<torch::nn::Module>(buffers[i].value());
        result2[i] = make_sharable_string(buffers[i].key());
    }
}

void THSNN_Module_get_named_modules(const NNModule module, NNModule* (*allocator1)(size_t length), const char** (*allocator2)(size_t length))
{
    auto buffers = (*module)->named_modules();
    NNModule* result1 = allocator1(buffers.size());
    const char** result2 = allocator2(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++)
    {
        result1[i] = new std::shared_ptr<torch::nn::Module>(buffers[i].value());
        result2[i] = make_sharable_string(buffers[i].key());
    }
}

long THSNN_Module_children_size(const NNModule module)
{
    return (*module)->children().size();
}

NNModule THSNN_Module_child(const NNModule module, const int index)
{
    return new std::shared_ptr<torch::nn::Module>((*module)->children()[index]);
}


// Save and restore

NNModule THSNN_Module_load(const char* location)
{
    CATCH_RETURN_NNModule(
        auto module = new torch::nn::Module();
    auto input = torch::serialize::InputArchive();

    input.load_from(location);
    module->load(input);
    return new std::shared_ptr<torch::nn::Module>(module);
    );
}

void THSNN_Module_save(const NNModule module, const char* location)
{
    CATCH(
        auto output = torch::serialize::OutputArchive();

    (*module)->save(output);
    output.save_to(location);
    );
}


// Wrapper class used to enable .NET definitions ot new modules describing parameters and with delegates to implement forward function
class CustomModule : public torch::nn::Module
{
public:
    CustomModule(
        const char* name,
        Tensor(*forward)(Tensor))
        : torch::nn::Module(name), _forward(forward)
    {
    }

    Tensor(*_forward)(Tensor);

    at::Tensor forward(at::Tensor input) {
        return *(*_forward)(&input);
    }

};

NNModule THSNN_custom_module(const char* name,
    Tensor(*forward)(Tensor),
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto mod = new CustomModule(name, forward);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != nullptr)
    {
        auto modShared = new std::shared_ptr<CustomModule>(mod);
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<CustomModule>(*modShared));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }
    res = new std::shared_ptr<torch::nn::Module>((torch::nn::Module*)mod);
    );
}

#if 0
struct TORCH_API ModuleBackWardHook1 : public torch::autograd::FunctionPostHook {

    virtual ~ModuleBackWardHook1() { }
    virtual torch::autograd::variable_list operator()(
        const torch::autograd::variable_list& outputs /* grad_inputs */,
        const torch::autograd::variable_list& inputs /* grad_outputs */)
    {
    } 
};
#endif
