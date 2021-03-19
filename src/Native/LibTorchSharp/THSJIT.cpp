//// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSJIT.h"

JITModule THSJIT_load(const char* filename)
{
	auto res = torch::jit::load(filename);
	auto copy = new torch::jit::Module(res);
	return new std::shared_ptr<torch::jit::Module>(copy);
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

JITMethod THSJIT_Module_get_method(const JITModule module, const char* name)
{
    auto method = (*module)->get_method(name);
    auto copy = new torch::jit::Method(method);
    return new std::shared_ptr<torch::jit::Method>(copy);
}

Tensor THSJIT_Module_forward(const JITModule module, const Tensor* tensorPtrs, const int length)
{
    return new torch::Tensor((*module)->forward(toTensors<c10::IValue>((torch::Tensor**)tensorPtrs, length)).toTensor());
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

//void* THSJIT_typeCast(const JITType type)
//{
//	switch ((*type)->kind())
//	{
//	case c10::TypeKind::TensorType:
//		return new std::shared_ptr<torch::jit::TensorType>((*type)->cast<c10::TensorType>());
//	case c10::TypeKind::DimensionedTensorType:
//		return new std::shared_ptr<torch::jit::DimensionedTensorType>((*type)->cast<c10::DimensionedTensorType>());
//	default:
//		return NULL;
//	}
//}
//
//int8_t THSJIT_typeKind(const JITType type)
//{
//	switch ((*type)->kind())
//	{
//	case c10::TypeKind::TensorType:
//		return (int8_t)TypeKind::TensorType;
//	case c10::TypeKind::DimensionedTensorType:
//		return (int8_t)TypeKind::DimensionedTensorType;
//	default:
//		return -1;
//	}
//}
//
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
//

//
//void THSJIT_typeDispose(const JITType type)
//{
//    delete type;
//}