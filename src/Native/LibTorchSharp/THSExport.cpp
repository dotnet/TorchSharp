// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSExport.h"

// torch.export support via AOTInductor
// This uses torch::inductor::AOTIModelPackageLoader which is INFERENCE-ONLY
// Models must be compiled with torch._inductor.aoti_compile_and_package() in Python

ExportedProgramModule THSExport_load(const char* filename)
{
    CATCH(
        // Load .pt2 file using AOTIModelPackageLoader
        // This requires models to be compiled with aoti_compile_and_package()
        auto* loader = new torch::inductor::AOTIModelPackageLoader(filename);
        return loader;
    );

    return nullptr;
}

void THSExport_Module_dispose(const ExportedProgramModule module)
{
    delete module;
}

void THSExport_Module_run(
    const ExportedProgramModule module,
    const Tensor* input_tensors,
    const int input_length,
    Tensor** result_tensors,
    int* result_length)
{
    CATCH(
        // Convert input tensor pointers to std::vector<torch::Tensor>
        std::vector<torch::Tensor> inputs;
        inputs.reserve(input_length);
        for (int i = 0; i < input_length; i++) {
            inputs.push_back(*input_tensors[i]);
        }

        // Run inference
        std::vector<torch::Tensor> outputs = module->run(inputs);

        // Allocate output array and copy results
        *result_length = outputs.size();
        *result_tensors = new Tensor[outputs.size()];

        for (size_t i = 0; i < outputs.size(); i++) {
            (*result_tensors)[i] = new torch::Tensor(outputs[i]);
        }
    );
}
