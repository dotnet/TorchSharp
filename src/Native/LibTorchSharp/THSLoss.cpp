// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
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

Tensor THSNN_binary_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::BinaryCrossEntropyFuncOptions();
        ApplyReduction(opts, reduction);
        if (weight != nullptr)
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

Tensor THSNN_cosine_embedding_loss(const Tensor input1, const Tensor input2, const Tensor target, const double margin, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::CosineEmbeddingLossFuncOptions().margin(margin);
        ApplyReduction(opts, reduction);
        res = ResultTensor(torch::nn::functional::cosine_embedding_loss(*input1, *input2, *target, opts));
    )
}

Tensor THSNN_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t ignore_index, const bool has_ii, const int64_t reduction, const double smoothing)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::CrossEntropyFuncOptions();
        ApplyReduction(opts, reduction);
        opts.label_smoothing(smoothing);
        if (has_ii)
            opts = opts.ignore_index(ignore_index);
        if (weight != nullptr)
            opts = opts.weight(*weight);
        res = ResultTensor(torch::nn::functional::cross_entropy(*input, *target, opts));
    )
}

Tensor THSNN_hinge_embedding_loss(const Tensor input, const Tensor target, const double margin, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::HingeEmbeddingLossFuncOptions().margin(margin);
        ApplyReduction(opts, reduction);
        res = ResultTensor(torch::nn::functional::hinge_embedding_loss(*input, *target, opts));
    )
}

Tensor THSNN_huber_loss(const Tensor input, const Tensor target, const double delta, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::HuberLossFuncOptions().delta(delta);
        ApplyReduction(opts, reduction);
        res = ResultTensor(torch::nn::functional::huber_loss(*input, *target, opts));
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

Tensor THSNN_l1_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::L1LossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::l1_loss(*input, *target, opts));
    )
}

Tensor THSNN_margin_ranking_loss(const Tensor input1, const Tensor input2, const Tensor target, const double margin, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MarginRankingLossFuncOptions().margin(margin);
        ApplyReduction(opts, reduction);
        res = ResultTensor(torch::nn::functional::margin_ranking_loss(*input1, *input2, *target, opts));
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
        if (weight != nullptr)
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

Tensor THSNN_smooth_l1_loss(const Tensor input, const Tensor target, const int64_t reduction, const double beta)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::SmoothL1LossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::smooth_l1_loss(*input, *target, opts, beta));
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

Tensor THSNN_ctc_loss(const Tensor log_probs, const Tensor targets, const Tensor input_lengths, const Tensor target_lengths, int64_t blank, bool zero_infinity, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::CTCLossFuncOptions().blank(blank).zero_infinity(zero_infinity);
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::ctc_loss(*log_probs, *targets, *input_lengths, *target_lengths, opts));
    )
}

Tensor THSNN_multilabel_margin_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MultilabelMarginLossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::multilabel_margin_loss(*input, *target, opts));
    )
}

Tensor THSNN_multilabel_soft_margin_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MultilabelSoftMarginLossFuncOptions();
        ApplyReduction(opts, reduction);
        if (weight != nullptr)
            opts = opts.weight(*weight);

        res = ResultTensor(torch::nn::functional::multilabel_soft_margin_loss(*input, *target, opts));
    )
}

Tensor THSNN_multi_margin_loss(const Tensor input, const Tensor target, const int64_t p, const double margin, const Tensor weight, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::MultiMarginLossFuncOptions()
        .p(p)
        .margin(margin);
        ApplyReduction(opts, reduction);
        if (weight != nullptr)
            opts = opts.weight(*weight);

        res = ResultTensor(torch::nn::functional::multi_margin_loss(*input, *target, opts));
    )
}

Tensor THSNN_triplet_margin_loss(const Tensor anchor, const Tensor positive, const Tensor negative, double margin, int64_t p, double eps, bool swap, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::TripletMarginLossFuncOptions()
        .p(p)
        .eps(eps)
        .margin(margin)
        .swap(swap);
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::triplet_margin_loss(*anchor, *positive, *negative, opts));
    )
}

Tensor THSNN_triplet_margin_with_distance_loss(const Tensor anchor, const Tensor positive, const Tensor negative, Tensor(*distance_function)(const Tensor x, const Tensor y), double margin, bool swap, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::TripletMarginWithDistanceLossFuncOptions()
        .margin(margin)
        .swap(swap);

        ApplyReduction(opts, reduction);

        if (distance_function != nullptr) {
            opts = opts.distance_function(
                [=](const at::Tensor& x, const at::Tensor& y) -> const at::Tensor& {
                auto x1 = ResultTensor(x);
                auto y1 = ResultTensor(y);
                return *distance_function(x1, y1);
            });
        }

        res = ResultTensor(torch::nn::functional::triplet_margin_with_distance_loss(*anchor, *positive, *negative, opts));
    )
}
