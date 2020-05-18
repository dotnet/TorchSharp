//// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSJIT.h"
//
//JITModule THSJIT_loadModule(const char* filename)
//{
//	auto res = torch::jit::load(filename);
//	auto copy = new torch::jit::script::Module(res);
//	return new std::shared_ptr<torch::jit::script::Module>(copy);
//}
//
//long THSJIT_getNumModules(const JITModule module)
//{
//    return (*module)->get_modules().size();
//}
//
//JITModule THSJIT_getSubModule(const JITModule module, const int index)
//{
//	int count;
//	for (const auto& child : (*module)->get_modules()) {
//		if (count == index)
//		{
//			auto copy = new torch::jit::script::Module(child);
//			return new std::shared_ptr<torch::jit::script::Module>(copy);
//		}
//		count++;
//	}
//}
//
//JITModule THSJIT_getSubModuleByName(const JITModule module, const char* name)
//{
//	auto res = (*module)->get_module(name);
//	auto copy = new torch::jit::script::Module(res);
//	return new std::shared_ptr<torch::jit::script::Module>(copy);
//}
//
//int THSJIT_getNumberOfInputs(const JITModule module)
//{
//	c10::optional<torch::jit::script::Method method = (*module)->find_method("forward");
//	auto args = method-> // getSchema().arguments();
//    return args.size();
//}
//
//int THSJIT_getNumberOfOutputs(const JITModule module)
//{
//    auto method = (*module)->find_method("forward");
//    auto outputs = method->getSchema().returns();
//    return outputs.size();
//}
//
//JITType THSJIT_getInputType(const JITModule module, const int n)
//{
//    auto method = (*module)->find_method("forward");
//    auto args = method->getSchema().arguments();
//    auto type = args[n].type();
//
//    return new std::shared_ptr<c10::Type>(type);
//}
//
//JITType THSJIT_getOutputType(const JITModule module, const int n)
//{
//    auto method = (*module)->find_method("forward");
//    auto outputs = method->getSchema().returns();
//    auto type = outputs[n].type();
//
//    return new std::shared_ptr<c10::Type>(type);
//}
//
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
//Tensor THSJIT_forward(const JITModule module, const Tensor* tensorPtrs, const int length)
//{
//    return new torch::Tensor((*module)->forward(toTensors<c10::IValue>((torch::Tensor**)tensorPtrs, length)).toTensor());
//}
//
//void THSJIT_moduleDispose(const JITModule module)
//{
//    delete module;
//}
//
//void THSJIT_typeDispose(const JITType type)
//{
//    delete type;
//}