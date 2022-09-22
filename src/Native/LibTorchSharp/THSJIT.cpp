//// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSJIT.h"

JITModule THSJIT_load(const char* filename)
{
    CATCH(
        auto res = torch::jit::load(filename);
        auto copy = new torch::jit::Module(res);        
        return new std::shared_ptr<torch::jit::Module>(copy);
    );

    return nullptr;
}

void THSJIT_save(JITModule module, const char* filename)
{
    CATCH(
        (*module)->save(filename);
    );
}

int THSJIT_Module_is_training(JITModule module)
{
    return (*module)->is_training();
}

void THSJIT_Module_train(JITModule module, bool on)
{
    (*module)->train(on);
}

void THSJIT_Module_eval(JITModule module)
{
    (*module)->eval();
}

void THSJIT_Module_to_device_dtype(JITModule module, int8_t dtype, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    (*module)->to(torch::Device(dev, index), (at::ScalarType)dtype);
}

void THSJIT_Module_to_device(JITModule module, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    (*module)->to(torch::Device(dev, index));
}

void THSJIT_Module_to_dtype(JITModule module, int8_t dtype)
{
    (*module)->to((at::ScalarType)dtype);
}

void THSJIT_Module_modules(const JITModule module, JITModule* (*allocator)(size_t length))
{
    auto modules = (*module)->modules();
    JITModule* result = allocator(modules.size());
    int i = 0;
    for (const auto& child : modules) {
        auto copy = new torch::jit::Module(child);
        result[i++] = new std::shared_ptr<torch::jit::Module>(copy);
    }
}

void THSJIT_Module_named_modules(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto modules = (*module)->named_modules();
    JITModule* result = allocator(modules.size());
    const char** names = allocator2(modules.size());
    int i = 0;
    for (const auto& child : modules) {
        auto copy = new torch::jit::Module(child.value);
        result[i] = new std::shared_ptr<torch::jit::Module>(copy);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_named_children(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto modules = (*module)->named_children();
    JITModule* result = allocator(modules.size());
    const char** names = allocator2(modules.size());
    int i = 0;
    for (const auto& child : modules) {
        auto copy = new torch::jit::Module(child.value);
        result[i] = new std::shared_ptr<torch::jit::Module>(copy);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_parameters(const JITModule module, Tensor* (*allocator)(size_t length))
{
    auto parameters = (*module)->parameters();
    Tensor* result = allocator(parameters.size());
    int i = 0;
    for (const auto& child : parameters) {
        result[i++] = new torch::Tensor(child);
    }
}

void THSJIT_Module_named_parameters(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto parameters = (*module)->named_parameters();
    Tensor* result = allocator(parameters.size());
    const char** names = allocator2(parameters.size());
    int i = 0;
    for (const auto& child : parameters) {
        result[i] = new torch::Tensor(child.value);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_named_buffers(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto parameters = (*module)->named_buffers();
    Tensor* result = allocator(parameters.size());
    const char** names = allocator2(parameters.size());
    int i = 0;
    for (const auto& child : parameters) {
        result[i] = new torch::Tensor(child.value);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

JITMethod THSJIT_Module_get_method(const JITModule module, const char* name)
{
    auto method = (*module)->get_method(name);
    auto copy = new torch::jit::Method(method);
    return new std::shared_ptr<torch::jit::Method>(copy);
}

Tensor THSJIT_Module_forward(const JITModule module, const Tensor* tensorPtrs, const int length)
{
    CATCH_TENSOR((*module)->forward(toTensors<c10::IValue>((torch::Tensor**)tensorPtrs, length)).toTensor());
}

void THSJIT_Module_dispose(const JITModule module)
{
    delete module;
}

const char* THSJIT_Method_name(const JITMethod method)
{
    return make_sharable_string((*method)->name());
}

int THSJIT_Method_num_inputs(const JITMethod method)
{
    return (int)(*method)->num_inputs();
}

int THSJIT_Module_num_inputs(const JITModule module)
{
    return (int)(*module)->get_method("forward").num_inputs() - 1; // Don't count the 'self' argument.
}

int THSJIT_Module_num_outputs(const JITModule module)
{
    return (int)(*module)->get_method("forward").function().getSchema().returns().size();
}

JITFunction THSJIT_Method_function(const JITMethod method)
{
    return new std::shared_ptr<torch::jit::Function>(&(*method)->function());
}

void THSJIT_Method_dispose(const JITMethod method)
{
    delete method;
}


//-------------------------------------------------------------------------------------
// JITFunction

int THSJIT_Function_num_inputs(const JITFunction function)
{
    return (int)(*function)->num_inputs();
}

// TODO other function operations

void THSJIT_Function_dispose(const JITFunction function)
{
    delete function;
}

void THSJIT_Type_dispose(const JITType type)
{
    delete type;
}

void THSJIT_TensorType_dispose(const JITTensorType type)
{
    delete type;
}

void* THSJIT_Type_cast(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::TensorType:
        return new std::shared_ptr<torch::jit::TensorType>((*type)->cast<c10::TensorType>());
        //case c10::TypeKind::DimensionedTensorType:
        //	return new std::shared_ptr<torch::jit::DimensionedTensorType>((*type)->cast<c10::DimensionedTensorType>());
    default:
        return NULL;
    }
}

int8_t THSJIT_TensorType_dtype(const JITTensorType type)
{
    auto scT = (*type)->scalarType();
    if (scT.has_value()) {
        return (int8_t)scT.value();
    }
    else {
        return -1;
    }
}

void THSJIT_TensorType_sizes(const JITTensorType type, int64_t* (*allocator)(int64_t length))
{
    //CATCH(
    auto& t = *type;
    auto dim = t->dim();
    auto res = (*type)->sizes().concrete_sizes();
    if (res.has_value()) {
        const size_t sz = res.value().size();
        auto& vec = res.value();
        int64_t* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = vec[i];
    }
    //);
}

int8_t THSJIT_Type_kind(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::TensorType:
        return (int8_t)TypeKind::TensorType;
        //case c10::TypeKind::DimensionedTensorType:
        //	return (int8_t)TypeKind::DimensionedTensorType;
    default:
        return -1;
    }
}

JITType THSJIT_Module_getInputType(JITModule module, int8_t index)
{
    auto typ = (*module)->type();
    c10::TypeKind kind = typ->kind();
    auto& schema = typ->getMethod("forward").getSchema();
    return new std::shared_ptr<c10::Type>(schema.arguments()[1 + index].type()->cast<c10::TensorType>());
}

//int8_t THSJIT_getScalarFromDimensionedTensorType(const JITDimensionedTensorType type)
//{
//	return (int8_t)(*type)->scalarType();
//}
//
//int THSJIT_getDimensionedTensorTypeDimensions(const JITDimensionedTensorType type)
//{
//	return (*type)->dim();
//}
//
//const char* THSJIT_getDimensionedTensorDevice(const JITDimensionedTensorType type)
//{
//	auto device = (*type)->device();
//
//	auto device_type = DeviceTypeName(device.type());
//
//	std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);
//
//	return make_sharable_string(device_type);
//}



void THSJIT_typeDispose(const JITType type)
{
    delete type;
}