#include "THSJIT.h"

JITModule THSJIT_loadModule(const char* filename)
{
    return new std::shared_ptr<torch::jit::script::Module>(torch::jit::load(filename));
}

long THSJIT_getNumModules(const JITModule module)
{
    return (*module)->get_modules().size();
}

const char* THSJIT_getModuleName(const JITModule module, const int index)
{
    auto keys = (*module)->get_modules().keys();

    return make_sharable_string(keys[index]);
}

JITModule THSJIT_getModuleFromIndex(const JITModule module, const int index)
{
    auto values = (*module)->get_modules().values();

    return new std::shared_ptr<torch::jit::script::Module>(values[index].module);
}

JITModule THSJIT_getModuleFromName(const JITModule module, const char* name)
{
    return new std::shared_ptr<torch::jit::script::Module>((*module)->get_module(name));
}

int THSJIT_getNumberOfInputs(const JITModule module)
{
    auto method = (*module)->find_method("forward");
    auto args = method->getSchema().arguments();
    return args.size();
}

int THSJIT_getNumberOfOutputs(const JITModule module)
{
    auto method = (*module)->find_method("forward");
    auto outputs = method->getSchema().returns();
    return outputs.size();
}

JITType THSJIT_getInputType(const JITModule module, const int n)
{
    auto method = (*module)->find_method("forward");
    auto args = method->getSchema().arguments();
    auto type = args[n].type();

    return new std::shared_ptr<c10::Type>(type);
}

JITType THSJIT_getOutputType(const JITModule module, const int n)
{
    auto method = (*module)->find_method("forward");
    auto outputs = method->getSchema().returns();
    auto type = outputs[n].type();

    return new std::shared_ptr<c10::Type>(type);
}

void * THSJIT_typeCast(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::DynamicType:
        return new std::shared_ptr<torch::jit::DynamicType>((*type)->cast<c10::DynamicType>());
    case c10::TypeKind::TensorType:
        return new std::shared_ptr<torch::jit::TensorType>((*type)->cast<c10::TensorType>());
    default:
        return NULL;
    }
}

int8_t THSJIT_typeKind(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::DynamicType:
        return (int8_t)TypeKind::DynamicType;
    case c10::TypeKind::TensorType:
        return (int8_t)TypeKind::TensorType;
    default:
        return -1;
    }
}

int8_t THSJIT_getScalarFromTensorType(const JITTensorType type)
{
    return (int8_t)(*type)->scalarType();
}

int THSJIT_getTensorTypeDimensions(const JITTensorType type)
{
    return (*type)->dim();
}

const char * THSJIT_getTensorDevice(const JITTensorType type)
{
    auto device = (*type)->device();

    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return make_sharable_string(device_type);
}

Tensor THSJIT_forward(const JITModule module, const Tensor* tensorPtrs, const int length)
{
    return new torch::Tensor((*module)->forward(toTensors<c10::IValue>((torch::Tensor**)tensorPtrs, length)).toTensor());
}

void THSJIT_moduleDispose(const JITModule module)
{
    delete module;
}

void THSJIT_typeDispose(const JITType type)
{
    delete type;
}