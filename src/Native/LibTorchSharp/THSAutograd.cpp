// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSAutograd.h"

#include "torch/torch.h"

bool THSAutograd_isGradEnabled()
{
    bool result = torch::autograd::GradMode::is_enabled();
    return result;
}

void THSAutograd_setGrad(bool enabled)
{
    torch::autograd::GradMode::set_enabled(enabled);
}

bool THSAutograd_isInferenceModeEnabled()
{
    bool result = torch::InferenceMode::is_enabled();
    return result;
}

torch::InferenceMode* THSAutograd_getInferenceModeGuard(bool mode)
{
    auto ptr = new torch::InferenceMode(mode);
    return ptr;
}

void THSAutograd_deleteInferenceModeGuard(torch::InferenceMode* ptr)
{
    delete ptr;
}

bool THSAutograd_isAnomalyEnabled()
{
    bool result = torch::autograd::AnomalyMode::is_enabled();
    return result;
}

bool THSAutograd_shouldCheckNaN()
{
    return torch::autograd::AnomalyMode::should_check_nan();
}

void THSAutograd_setAnomaly(bool enabled, bool check_nan)
{
    torch::autograd::AnomalyMode::set_enabled(enabled, check_nan);
}

void THSAutograd_grad(
    Tensor* outputs, const int64_t oLength,
    Tensor* inputs, const int64_t iLength,
    Tensor* grad_outs, const int64_t gLength,
    bool retain_graph, bool create_graph, bool allow_unused,
    Tensor* (*allocator)(size_t length))
{
    auto res = torch::autograd::grad(
        toTensors<at::Tensor>((torch::Tensor**)outputs, oLength),
        toTensors<at::Tensor>((torch::Tensor**)inputs, iLength),
        toTensors<at::Tensor>((torch::Tensor**)grad_outs, gLength),
        retain_graph, create_graph, allow_unused);

    const size_t sz = res.size();

    Tensor* result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = ResultTensor(res[i]);
}

void THSAutograd_backward(
    Tensor* tensors, const int64_t tLength,
    Tensor* grad_tensors, const int64_t gtLength,
    bool retain_graph, bool create_graph,
    Tensor* inputs, const int64_t iLength)
{
    CATCH(
        torch::autograd::backward(
            toTensors<at::Tensor>((torch::Tensor**)tensors, tLength),
            toTensors<at::Tensor>((torch::Tensor**)grad_tensors, gtLength),
            retain_graph, create_graph,
            toTensors<at::Tensor>((torch::Tensor**)inputs, iLength));
    );
}
