// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/script.h"

#include "Utils.h"

//// Copied from libtorch to share the type as an int8_t.
//enum TypeKind : int8_t {
//#define DEFINE_TYPE(T) T,
//    C10_FORALL_TYPES(DEFINE_TYPE)
//#undef DEFINE_TYPE
//};
//
//// API.


EXPORT_API(JITModule) THSJIT_load(const char* filename);

EXPORT_API(void) THSJIT_Module_modules(const JITModule module, JITModule* (*allocator)(size_t length));

EXPORT_API(void) THSJIT_Module_named_modules(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(JITMethod) THSJIT_Module_get_method(const JITModule module, const char* name);

EXPORT_API(void) THSJIT_Module_parameters(const JITModule module, Tensor* (*allocator)(size_t length));

EXPORT_API(void) THSJIT_Module_named_parameters(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(Tensor) THSJIT_Module_forward(const JITModule module, const Tensor* tensorPtrs, const int length);

EXPORT_API(void) THSJIT_Module_dispose(const JITModule module);

EXPORT_API(const char*) THSJIT_Method_name(const JITMethod method);

EXPORT_API(int) THSJIT_Method_num_inputs(const JITMethod method);

EXPORT_API(void) THSJIT_Method_dispose(const JITMethod method);
