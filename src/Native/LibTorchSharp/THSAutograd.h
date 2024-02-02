// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// Returns whether the grad is enabled or not.
EXPORT_API(bool) THSAutograd_isGradEnabled();

// Enables / disables grad.
EXPORT_API(void) THSAutograd_setGrad(bool enabled);

// Returns whether inference mode is enabled or not
EXPORT_API(bool) THSAutograd_isInferenceModeEnabled();

// Returns a pointer to an RAII guard for inference mode
EXPORT_API(torch::InferenceMode*) THSAutograd_getInferenceModeGuard(bool mode);

// Destroys the RAII guard and restores the original state
EXPORT_API(void) THSAutograd_deleteInferenceModeGuard(torch::InferenceMode* ptr);

// Returns whether the grad is enabled or not.
EXPORT_API(bool) THSAutograd_isAnomalyEnabled();

EXPORT_API(bool) THSAutograd_shouldCheckNaN();

// Enables / disables grad.
EXPORT_API(void) THSAutograd_setAnomaly(bool enabled, bool check_nan);

EXPORT_API(void) THSAutograd_grad(
    Tensor* outputs, const int64_t oLength,
    Tensor* inputs, const int64_t iLength,
    Tensor* grad_outs, const int64_t gLenght,
    bool retain_graph, bool create_graph, bool allow_unused,
    Tensor* (*allocator)(size_t length));

EXPORT_API(void) THSAutograd_backward(
    Tensor* tensors, const int64_t tLength,
    Tensor* grad_tensors, const int64_t gtLength,
    bool retain_graph, bool create_graph,
    Tensor* inputs, const int64_t iLength);


using variable_list = torch::autograd::variable_list;
struct CSharpNode : public torch::autograd::Node {
    CSharpNode(std::function<TensorArray (Tensor*, int)> applyFunc, std::function<void()> managedDeleteNode)
        : applyFunc(applyFunc), managedDeleteNode(managedDeleteNode) {}

    std::function<TensorArray (Tensor*, int)> applyFunc;
    std::function<void()> managedDeleteNode;

    variable_list apply(variable_list&& inputs) override;
};

struct CSharpNodePtr {
    std::shared_ptr<CSharpNode> *shared_ptr;
    std::weak_ptr<CSharpNode> *weak_ptr;
};

EXPORT_API(CSharpNodePtr) THSAutograd_CSharpNode_ctor(TensorArray(*applyFunc)(Tensor*, int), void (*managedDeleteNode)());

EXPORT_API(void) THSAutograd_CSharpNode_disposeSharedPtr(CSharpNodePtr node);

EXPORT_API(void) THSAutograd_CSharpNode_disposeWeakPtr(CSharpNodePtr node);

EXPORT_API(void) THSAutograd_CSharpNode_setNextEdges(CSharpNodePtr node, TensorArray vars_, bool is_executable);

EXPORT_API(void) THSAutograd_CSharpNode_clearInputMetadata(CSharpNodePtr node);

EXPORT_API(void) THSAutograd_Function_wrapOutputs(TensorArray vars_, TensorArray nonDiff_, TensorArray dirty_, TensorArray outputs_, CSharpNodePtr node, Tensor* (*allocator)(size_t length));

EXPORT_API(SavedVariable) THSAutograd_SavedVariable_ctor(Tensor variable, CSharpNodePtr node, bool is_inplace_on_view);

EXPORT_API(void) THSAutograd_SavedVariable_dispose(SavedVariable var);

EXPORT_API(Tensor) THSAutograd_SavedVariable_unpack(SavedVariable saved_variable, CSharpNodePtr saved_for);

EXPORT_API(void) THSAutograd_SavedVariable_reset_data(SavedVariable saved_variable);