// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSJIT.h"

JITModule THSJIT_loadModule(const char* filename)
{
    return new std::shared_ptr<torch::jit::script::Module>(torch::jit::load(filename));
}

long THSJIT_getNumModules(const JITModule module)
{
    return (*module)->get_modules().size();
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