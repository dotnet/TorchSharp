//// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSJIT.h"

JITModule THSJIT_load(const char* filename, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;

    CATCH(
        auto res = torch::jit::load(filename, torch::Device(dev, index));
        auto copy = new torch::jit::Module(res);
        return new std::shared_ptr<torch::jit::Module>(copy);
    );

    return nullptr;
}

JITModule THSJIT_load_byte_array(char* bytes, int64_t size, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;

    CATCH(
        std::istringstream stream(std::string(bytes, size));

        auto res = torch::jit::load(stream, torch::Device(dev, index));
        auto copy = new torch::jit::Module(res);
        return new std::shared_ptr<torch::jit::Module>(copy);
    );

    return nullptr;
}

JITCompilationUnit THSJIT_compile(const char* script)
{
    CATCH(
        auto res = torch::jit::compile(script);
        return new std::shared_ptr<torch::jit::CompilationUnit>(res);
    );

    return nullptr;
}

void THSJIT_save(JITModule module, const char* filename)
{
    CATCH(
        (*module)->save(filename);
    );
}

void THSJIT_save_byte_array(JITModule module, char* bytes, int64_t size)
{
    CATCH(
        std::ostringstream stream(std::string(bytes, size));

        (*module)->save(stream);
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
    if (device == 13)
        dev = c10::kMPS;

    (*module)->to(torch::Device(dev, index));
}

void THSJIT_Module_to_device(JITModule module, int64_t device, int64_t index)
{
    c10::Device dev = (device == 1) ? torch::Device(c10::kCUDA, index) : torch::Device(c10::kCPU);

    (*module)->to(dev);
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

void THSJIT_Module_named_attributes(const JITModule module, bool recurse,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto attributes = (*module)->named_attributes(recurse);
    Tensor* result = allocator(attributes.size());
    const char** names = allocator2(attributes.size());
    int i = 0;
    for (const auto& child : attributes) {
        if (!child.name.empty() && child.value.isTensor())
        {
            auto& t = child.value.toTensor();
            result[i] = new torch::Tensor(child.value.toTensor());
            names[i] = make_sharable_string(child.name);
            i++;
        }
    }
}

void THSJIT_Module_set_attribute(const JITModule module, const char *name, Tensor tensor)
{
    CATCH((*module)->setattr(name, *tensor););
}

JITMethod THSJIT_Module_get_method(const JITModule module, const char* name)
{
    auto method = (*module)->get_method(name);
    auto copy = new torch::jit::Method(method);
    return new std::shared_ptr<torch::jit::Method>(copy);
}

TensorOrScalar* validate(TensorOrScalar* ptr)
{
    if (ptr == nullptr)
        torch_last_err = strdup("null returned from TorchSharp return value allocator.");
    return ptr;
}
TensorOrScalar* ReturnHelper(c10::IValue result, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t* idx)
{
    // TypeCode:
    //
    // 0 -- Not supported
    // 1 -- Single tensor
    // 2 -- Tuple of tensors
    // 3 -- List of tensors
    // 4 -- Single scalar
    // 5 -- Scalar tuple
    // 6 -- List of scalars
    // 7 -- List of scalars and tensors
    // 8 -- None / null
    // 9 -- List of anything
    // 10 -- Tuple of anything

    if (result.isNone())
    {
        TensorOrScalar* output = validate(allocator(idx[0]++, 1));
        if (output == nullptr) return output;
        output[0] = { 8, -1, (ptrdiff_t)0 };
        *typeCode = 8;
        return output;
    }

    if (result.isScalar())
    {
        TensorOrScalar* output = validate(allocator(idx[0]++, 1));
        if (output == nullptr) return output;
        output[0] = { 0, -1, (ptrdiff_t)new torch::Scalar(result.toScalar()) };
        *typeCode = 4;
        return output;
    }

    if (result.isTensor()) {
        TensorOrScalar* output = validate(allocator(idx[0]++, 1));
        if (output == nullptr) return output;
        output[0] = { 0, -1, (ptrdiff_t)ResultTensor(result.toTensor()) };
        *typeCode = 1;
        return output;
    }

    if (result.isTensorList()) {
        auto list = result.toTensorList();
        *typeCode = 3;
        TensorOrScalar* output = validate(allocator(idx[0]++, list.size()));
        if (output == nullptr) return output;
        for (size_t i = 0; i < list.size(); i++)
            output[i] = { 0, -1, (ptrdiff_t)ResultTensor(list[i]) };
        return output;
    }

    if (result.isList())
    {
        int foundTensor = 0;
        int foundScalar = 0;
        int foundNull = 0;
        int foundListOrTuple = 0;

        auto list = result.toList();
        TensorOrScalar* output = validate(allocator(idx[0]++, list.size()));
        if (output == nullptr) return output;

        for (int i = 0; i < list.size(); ++i)
        {
            output[i].Handle = -1;
            c10::IValue value = list[i];

            if (value.isTensor())
            {
                output[i] = { 0, -1, (ptrdiff_t)ResultTensor(value.toTensor()) };
                foundTensor += 1;
                continue;
            }
            if (value.isScalar())
            {
                output[i] = { 4, -1, (ptrdiff_t)new torch::Scalar(value.toScalar()) };
                foundScalar += 1;
                continue;
            }
            if (value.isNone())
            {
                output[i] = { 8, -1, (ptrdiff_t)0 };
                foundNull += 1;
                continue;
            }
            else {
                int8_t nestedTC = 0;
                int64_t arrIdx = idx[0];
                auto nested = ReturnHelper(value, allocator, &nestedTC, idx);
                if (nested == nullptr) return nested;
                foundListOrTuple += 1;
                output[i] = { nestedTC, arrIdx, (ptrdiff_t)nested };
            }
        }

        if (foundListOrTuple > 0) {
            *typeCode = 9;
        }
        else {
            *typeCode = 7;
            if (foundScalar == 0 && foundNull == 0)
                *typeCode = 3;
            if (foundTensor == 0 && foundNull == 0)
                *typeCode = 6;
        }
        return output;
    }

    if (result.isTuple()) {
        int foundTensor = 0;
        int foundScalar = 0;
        int foundNull = 0;
        int foundListOrTuple = 0;

        auto& list = result.toTuple()->elements();
        TensorOrScalar* output = validate(allocator(idx[0]++, list.size()));
        if (output == nullptr) return output;

        for (int i = 0; i < list.size(); ++i)
        {
            output[i].Handle = -1;
            c10::IValue value = list[i];

            if (value.isTensor())
            {
                output[i] = { 0, -1, (ptrdiff_t)ResultTensor(value.toTensor()) };
                foundTensor += 1;
                continue;
            }
            if (value.isScalar())
            {
                output[i] = { 4, -1, (ptrdiff_t)new torch::Scalar(value.toScalar()) };
                foundScalar += 1;
                continue;
            }
            if (value.isNone())
            {
                output[i] = { 8, -1, (ptrdiff_t)0 };
                foundNull += 1;
                continue;
            }
            else {
                int8_t nestedTC = 0;
                int64_t arrIdx = idx[0];
                auto nested = ReturnHelper(value, allocator, &nestedTC, idx);
                if (nested == nullptr) return nested;
                foundListOrTuple += 1;
                output[i] = { nestedTC, arrIdx, (ptrdiff_t)nested };
            }
        }

        *typeCode = 10;
        if (foundListOrTuple == 0) {
            if (foundScalar == 0 && foundNull == 0)
                *typeCode = 2;
            if (foundTensor == 0 && foundNull == 0)
                *typeCode = 5;
        }

        return output;
    }

    *typeCode = 0;
    return nullptr;
}

c10::impl::GenericList toScalarValueList(const TensorOrScalar* tensorPtrs, const int length)
{
    auto list = c10::impl::GenericList(c10::ScalarTypeType::get());

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            switch (tensorPtrs[i].TypeCode) {
            case 1:
                list.push_back(*(torch::Scalar*)(tensorPtrs[i].Handle));
                break;
            }
        }
    }

    return list;
}

c10::impl::GenericList toTensorValueList(const TensorOrScalar* tensorPtrs, const int length)
{
    auto list = c10::impl::GenericList(c10::TensorType::get());

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            switch (tensorPtrs[i].TypeCode) {
            case 0:
                list.push_back(*(torch::Tensor*)(tensorPtrs[i].Handle));
                break;
            }
        }
    }

    return list;
}

std::vector<c10::IValue> toIValue(const TensorOrScalar* tensorPtrs, const int length)
{
    // TypeCode:
    //
    // 0 -- Single tensor
    // 1 -- Single scalar
    // 2 -- Boolean
    // 3 -- Int32
    // 5 -- List of tensors
    // 6 -- List of scalars
    // 8 -- None / null

    std::vector<c10::IValue> tensors;

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            switch (tensorPtrs[i].TypeCode) {
            case 0:
                tensors.push_back(*(torch::Tensor*)(tensorPtrs[i].Handle));
                break;
            case 1:
                tensors.push_back(*(torch::Scalar*)(tensorPtrs[i].Handle));
                break;
            case 2:
                tensors.push_back(tensorPtrs[i].Handle != 0);
                break;
            case 3:
                tensors.push_back((int)tensorPtrs[i].Handle);
                break;
            case 5:
            {
                auto ts = toTensorValueList(reinterpret_cast<const TensorOrScalar*>(tensorPtrs[i].Handle), (int)tensorPtrs[i].ArrayIndex);
                tensors.push_back(ts);
                break;
            }
            case 6:
            {
                auto ts = toScalarValueList(reinterpret_cast<const TensorOrScalar*>(tensorPtrs[i].Handle), (int)tensorPtrs[i].ArrayIndex);
                tensors.push_back(ts);
                break;
            }
            //case 4:
            //    tensors.push_back(c10::IValue(tensorPtrs[i].Handle)); // Clang on MacOS doesn't like. Pass as Scalar from .NET.
            //    break;
            case 8:
                tensors.push_back(c10::nullopt);
                break;
            }
        }
    }
    return tensors;
}

void THSJIT_Module_forward(const JITModule module, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx)
{
    *typeCode = 0;

    CATCH(
        auto result = (*module)->forward(toIValue(tensorPtrs, length));
        ReturnHelper(result, allocator, typeCode, &idx);
    )
}

void THSJIT_Module_invoke(const JITModule module, const char* name, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx)
{
    *typeCode = 0;

    CATCH(
        auto method = (*module)->get_method(name);
        auto result = method(toIValue(tensorPtrs, length));
        ReturnHelper(result, allocator, typeCode, &idx);
    )
}

void THSJIT_CompilationUnit_Invoke(const JITCompilationUnit module, const char* method, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx)
{
    *typeCode = 0;

    CATCH(
        auto args = toIValue(tensorPtrs, length);
        auto func = (*module)->find_function(method);
        auto result = (*func)(args);
        ReturnHelper(result, allocator, typeCode, &idx);
    )
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

void THSJIT_CompilationUnit_dispose(const JITCompilationUnit module)
{
    delete module;
}

void* THSJIT_Type_cast(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::TensorType:
        return new std::shared_ptr<torch::jit::TensorType>((*type)->cast<c10::TensorType>());
    default:
        return nullptr;
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

void THSJIT_typeDispose(const JITType type)
{
    delete type;
}


TensorOrScalar* THSJIT_AllocateTensorOrScalarArray(int32_t size)
{
    auto result = new TensorOrScalar[size];
    memset(result, 0, size * sizeof(TensorOrScalar));
    return result;
}

void THSJIT_FreeTensorOrScalarArray(TensorOrScalar* ptr)
{
    delete ptr;
}

void THSJIT_SetTensorOrScalar(TensorOrScalar* array, int32_t index, int64_t type_code, int64_t array_index, ptrdiff_t handle)
{
    array[index].TypeCode = type_code;
    array[index].ArrayIndex = array_index;
    array[index].Handle = handle;
}

TensorOrScalar* THSJIT_GetTensorOrScalar(TensorOrScalar* array, int32_t index)
{
    return array + index;
}