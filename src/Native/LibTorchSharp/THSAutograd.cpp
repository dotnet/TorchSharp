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

variable_list CSharpNode::apply(variable_list&& inputs) {
    at::OptionalDeviceGuard _device_guard;

    std::vector<Tensor> converted_inputs;
    converted_inputs.reserve(inputs.size());
    for (const auto t : inputs)
        converted_inputs.push_back(ResultTensor(t));

    auto res = applyFunc(converted_inputs.data(), converted_inputs.size());

    variable_list output;
    output.reserve(res.size);
    for (int i = 0; i < res.size; i++) {
        if (res.array[i] == nullptr)
            output.emplace_back();
        else output.emplace_back(*res.array[i]);
    }

    return output;
}

void deleteNode(CSharpNode* node) {
    node->managedDeleteNode();
    torch::autograd::deleteNode(node);
}


CSharpNodePtr THSAutograd_CSharpNode_ctor(TensorArray(*applyFunc)(Tensor*, int size), void (*managedDeleteNode)())
{
    CATCH_RETURN_RES(CSharpNodePtr, CSharpNodePtr(),
        res.shared_ptr = new std::shared_ptr<CSharpNode>(new CSharpNode(applyFunc, managedDeleteNode), deleteNode);
        res.weak_ptr = new std::weak_ptr<CSharpNode>(*res.shared_ptr);
    )
}

void THSAutograd_CSharpNode_disposeSharedPtr(CSharpNodePtr node) {
    CATCH(
        delete node.shared_ptr;
    )
}

void THSAutograd_CSharpNode_disposeWeakPtr(CSharpNodePtr node) {
    CATCH(
        delete node.weak_ptr;
    )
}

void THSAutograd_CSharpNode_setNextEdges(CSharpNodePtr node, TensorArray vars_, bool is_executable) {
    CATCH(
        auto next_edges = is_executable ? torch::autograd::collect_next_edges(toTensors<at::Tensor>(vars_.array, vars_.size)) : torch::autograd::edge_list();
        node.weak_ptr->lock()->set_next_edges(std::move(next_edges));
    )
}

void THSAutograd_CSharpNode_clearInputMetadata(CSharpNodePtr node) {
    CATCH(
        node.weak_ptr->lock()->clear_input_metadata();
    )
}

void THSAutograd_Function_wrapOutputs(TensorArray vars_, TensorArray nonDiff_, TensorArray dirty_, TensorArray outputs_, CSharpNodePtr node, Tensor* (*allocator)(size_t length)) {
    CATCH(
    auto vars = toTensors<at::Tensor>(vars_.array, vars_.size);
    auto output_tensors = toTensors<at::Tensor>(outputs_.array, outputs_.size);
    auto outputs = torch::autograd::to_optional(output_tensors);

    // Convert the list of Tensor to a set of unsafe impl
    std::unordered_set<at::TensorImpl*> nonDiff;
    nonDiff.reserve(nonDiff_.size);
    for (int i = 0; i < nonDiff_.size; i++)
        nonDiff.insert(nonDiff_.array[i]->unsafeGetTensorImpl());

    // Convert the list of Tensors to a set of unsafe impl, and then apply the behavior of AutogradContext::get_and_bump_dirty()
    std::unordered_set<at::TensorImpl*> dirty;
    dirty.reserve(dirty_.size);
    for (int i = 0; i < dirty_.size; i++) {
        auto t = dirty_.array[i]->unsafeGetTensorImpl();
        t->bump_version();
        dirty.insert(t);
    }

    // Copied these functions from custom_function.h
    torch::autograd::_jvp_fn_t jvp_fn = [](const variable_list& inputs,
        const variable_list& gI) -> variable_list {
            TORCH_CHECK(
                false,
                "jvp is not implemented for the c++ API of custom Function yet.",
                "Please open a feature request on GitHub if you need this.");
        };

    auto view_as_self_fn = [](const at::Tensor& x) -> at::Tensor {
        return x.view_as(x);
        };

    auto res = torch::autograd::_wrap_outputs(vars, nonDiff, dirty, outputs, node.weak_ptr == nullptr || node.weak_ptr->expired() ? nullptr : node.weak_ptr->lock(), jvp_fn, {}, view_as_self_fn);
    auto sz = res.size();

    Tensor* result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = res[i].has_value() ? ResultTensor(res[i].value()) : nullptr;
    )
}

SavedVariable THSAutograd_SavedVariable_ctor(Tensor variable, CSharpNodePtr node, bool is_inplace_on_view)
{
    CATCH_RETURN_RES(SavedVariable, nullptr,
        bool is_output = node.weak_ptr->lock().get() == variable->grad_fn().get();
        res = new std::shared_ptr<torch::autograd::SavedVariable>(new torch::autograd::SavedVariable(*variable, is_output, is_inplace_on_view))
    );
}

void THSAutograd_SavedVariable_dispose(SavedVariable var) {
    CATCH(
        delete var;
    )
}

Tensor THSAutograd_SavedVariable_unpack(SavedVariable saved_variable, CSharpNodePtr saved_for) {
    CATCH_RETURN_Tensor(
        res = ResultTensor((*saved_variable)->unpack(saved_for.weak_ptr == nullptr || saved_for.weak_ptr->expired() ? nullptr : saved_for.weak_ptr->lock()));
    )
}

void THSAutograd_SavedVariable_reset_data(SavedVariable saved_variable) {
    CATCH((*saved_variable)->reset_data();)
}