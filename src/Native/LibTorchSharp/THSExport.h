// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"
#include "torch/csrc/inductor/aoti_package/model_package_loader.h"

#include "Utils.h"

// torch.export support via AOTInductor - Load and execute PyTorch ExportedProgram models (.pt2 files)
// ExportedProgram is PyTorch 2.x's recommended way to export models for production deployment
//
// IMPORTANT: This implementation uses torch::inductor::AOTIModelPackageLoader which is
// INFERENCE-ONLY. Training, parameter updates, and device movement are not supported.
// Models must be compiled with torch._inductor.aoti_compile_and_package() in Python.

// Load an AOTInductor-compiled model package from a .pt2 file
EXPORT_API(ExportedProgramModule) THSExport_load(const char* filename);

// Dispose of an ExportedProgram module
EXPORT_API(void) THSExport_Module_dispose(const ExportedProgramModule module);

// Execute the ExportedProgram's forward method (inference only)
// Input: Array of tensors
// Output: Array of result tensors (caller must free)
EXPORT_API(void) THSExport_Module_run(
    const ExportedProgramModule module,
    const Tensor* input_tensors,
    const int input_length,
    Tensor** result_tensors,
    int* result_length);
