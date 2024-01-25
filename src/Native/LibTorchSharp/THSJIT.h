// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/script.h"

#include "Utils.h"

// Copied from libtorch to share the type as an int8_t.
enum TypeKind : int8_t {
#define DEFINE_TYPE(T) T,
    C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

// API.

struct TensorOrScalar
{
    int64_t TypeCode;
    int64_t ArrayIndex;
    ptrdiff_t Handle;
};


EXPORT_API(JITModule) THSJIT_load(const char* filename, int64_t device, int64_t index);
EXPORT_API(JITModule) THSJIT_load_byte_array(char* bytes, int64_t size, int64_t device, int64_t index);

EXPORT_API(void) THSJIT_save(JITModule module, const char* filename);
EXPORT_API(void) THSJIT_save_byte_array(JITModule module, char* bytes, int64_t size);

EXPORT_API(JITCompilationUnit) THSJIT_compile(const char* script);

EXPORT_API(void) THSJIT_Module_dispose(const JITModule module);
EXPORT_API(void) THSJIT_CompilationUnit_dispose(const JITCompilationUnit module);

EXPORT_API(int) THSJIT_Module_num_inputs(const JITModule method);
EXPORT_API(int) THSJIT_Module_num_outputs(const JITModule method);

EXPORT_API(void) THSJIT_Module_forward(const JITModule module, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx);
EXPORT_API(void) THSJIT_Module_invoke(const JITModule module, const char* name, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx);

EXPORT_API(void) THSJIT_CompilationUnit_Invoke(const JITCompilationUnit module, const char* method, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx);

EXPORT_API(int) THSJIT_Module_is_training(JITModule module);
EXPORT_API(void) THSJIT_Module_train(JITModule module, bool on);
EXPORT_API(void) THSJIT_Module_eval(JITModule module);

EXPORT_API(void) THSJIT_Module_to_device_dtype(JITModule module, int8_t dtype, int64_t device, int64_t index);
EXPORT_API(void) THSJIT_Module_to_device(JITModule module, int64_t device, int64_t index);
EXPORT_API(void) THSJIT_Module_to_dtype(JITModule module, int8_t dtype);

EXPORT_API(JITType) THSJIT_Module_getInputType(JITModule module, int8_t dtype);

EXPORT_API(int8_t) THSJIT_Type_kind(JITType handle);
EXPORT_API(void*) THSJIT_Type_cast(const JITType type);

EXPORT_API(int8_t) THSJIT_TensorType_dtype(const JITTensorType type);
EXPORT_API(void) THSJIT_TensorType_sizes(const JITTensorType type, int64_t* (*allocator)(int64_t length));

EXPORT_API(void) THSJIT_Type_dispose(const JITType type);
EXPORT_API(void) THSJIT_TensorType_dispose(const JITTensorType type);

EXPORT_API(void) THSJIT_Module_modules(const JITModule module, JITModule* (*allocator)(size_t length));
EXPORT_API(void) THSJIT_Module_named_modules(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_named_children(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(JITMethod) THSJIT_Module_get_method(const JITModule module, const char* name);

EXPORT_API(void) THSJIT_Module_parameters(const JITModule module, Tensor* (*allocator)(size_t length));
EXPORT_API(void) THSJIT_Module_named_parameters(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_named_buffers(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_named_attributes(const JITModule module, bool recurse,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_set_attribute(const JITModule module, const char* name, Tensor tensor);

EXPORT_API(int) THSJIT_Method_num_inputs(const JITMethod method);

EXPORT_API(void) THSJIT_Method_dispose(const JITMethod method);

EXPORT_API(const char*) THSJIT_Method_name(const JITMethod method);

EXPORT_API(TensorOrScalar*) THSJIT_AllocateTensorOrScalarArray(int32_t size);
EXPORT_API(void) THSJIT_FreeTensorOrScalarArray(TensorOrScalar* ptr);
EXPORT_API(void) THSJIT_SetTensorOrScalar(TensorOrScalar* array, int32_t index, int64_t type_code, int64_t array_index, ptrdiff_t handle);
EXPORT_API(TensorOrScalar*) THSJIT_GetTensorOrScalar(TensorOrScalar* array, int32_t index);
