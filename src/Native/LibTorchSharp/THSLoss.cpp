// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>


template<typename T>
void ApplyReduction(T& opts, const int64_t reduction)
{
    if (reduction == 0)
        opts = opts.reduction(torch::kNone);
    if (reduction == 1)
        opts = opts.reduction(torch::kMean);
    if (reduction == 2)
        opts = opts.reduction(torch::kSum);
}

Tensor THSNN_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t ignore_index, const bool has_ii, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::CrossEntropyFuncOptions();
    ApplyReduction(opts, reduction);
    if (has_ii)
        opts = opts.ignore_index(ignore_index);
    if (weight != NULL)
        opts = opts.weight(*weight);
    res = ResultTensor(torch::nn::functional::cross_entropy(*input, *target, opts));
    )
}

Tensor THSNN_binary_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::BinaryCrossEntropyFuncOptions();
    ApplyReduction(opts, reduction);
    if (weight != NULL)
        opts = opts.weight(*weight);
    res = ResultTensor(torch::nn::functional::binary_cross_entropy(*input, *target, opts));
    )
}

Tensor THSNN_binary_cross_entropy_with_logits(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction, const Tensor pos_weights_wrapper)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::BCEWithLogitsLossOptions();
    ApplyReduction(opts, reduction);
    if (pos_weights_wrapper != nullptr)
        opts = opts.pos_weight(*pos_weights_wrapper);
    if (weight != nullptr)
        opts = opts.weight(*weight);
    res = ResultTensor(torch::nn::functional::binary_cross_entropy_with_logits(*input, *target, opts));
    )
}

Tensor THSNN_l1_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MSELossFuncOptions();
    ApplyReduction(opts, reduction);

    res = ResultTensor(torch::nn::functional::mse_loss(*input, *target, opts));
    )
}

Tensor THSNN_mse_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MSELossFuncOptions();
    ApplyReduction(opts, reduction);

    res = ResultTensor(torch::nn::functional::mse_loss(*input, *target, opts));
    )
}

Tensor THSNN_nll_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::NLLLossFuncOptions();
    ApplyReduction(opts, reduction);
    if (weight != NULL)
        opts = opts.weight(*weight);

    res = ResultTensor(torch::nn::functional::nll_loss(*input, *target, opts));
    )
}

Tensor THSNN_poisson_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::PoissonNLLLossFuncOptions().log_input(logInput).full(full).eps(eps);
    ApplyReduction(opts, reduction);

    res = ResultTensor(torch::nn::functional::poisson_nll_loss(*input, *target, opts));
    )
}

Tensor THSNN_kl_div_loss(const Tensor input, const Tensor target, const int64_t reduction, const bool log_target)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::KLDivFuncOptions().log_target(log_target);
    ApplyReduction(opts, reduction);

    res = ResultTensor(torch::nn::functional::kl_div(*input, *target, opts));
    )
}

Tensor THSNN_smooth_l1_loss(const Tensor input, const Tensor target, const int64_t reduction, const double beta)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::SmoothL1LossFuncOptions();
    ApplyReduction(opts, reduction);

    res = ResultTensor(torch::nn::functional::smooth_l1_loss(*input, *target, opts));
    )
}

Tensor THSNN_soft_margin_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::SoftMarginLossFuncOptions();
    ApplyReduction(opts, reduction);

    res = ResultTensor(torch::nn::functional::soft_margin_loss(*input, *target, opts));
    )
}