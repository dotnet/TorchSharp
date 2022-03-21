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

Tensor THSNN_cosine_embedding_loss(const Tensor input1, const Tensor input2, const Tensor target, const double margin, const int64_t reduction)
{
    CATCH_RETURN_Tensor(
        auto opts = torch::nn::functional::CosineEmbeddingLossFuncOptions().margin(margin);
        ApplyReduction(opts, reduction);
        res = ResultTensor(torch::nn::functional::cosine_embedding_loss(*input1, *input2, *target, opts));
    )
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
        auto opts = torch::nn::functional::MSELossFuncOptions();
        ApplyReduction(opts, reduction);

        res = ResultTensor(torch::nn::functional::mse_loss(*input, *target, opts));
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
        if (weight != NULL)
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
        if (weight != NULL)
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

NNModule THSNN_L1Loss_ctor(const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::L1LossOptions();
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::L1LossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_L1Loss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::L1Loss>()->forward(*tensor, *target));
}

NNModule THSNN_MSELoss_ctor(const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MSELossOptions();
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::MSELossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MSELoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::MSELoss>()->forward(*tensor, *target));
}

NNModule THSNN_CrossEntropyLoss_ctor(const Tensor weight, const int64_t ignore_index, const bool has_ii, const double label_smoothing, const bool has_ls, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::CrossEntropyLossOptions();
        ApplyReduction(opts, reduction);
        if (has_ii)
            opts = opts.ignore_index(ignore_index);
        if (has_ls)
            opts = opts.label_smoothing(label_smoothing);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = create_module<torch::nn::CrossEntropyLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_CrossEntropyLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::CrossEntropyLoss>()->forward(*tensor, *target));
}

NNModule THSNN_CTCLoss_ctor(const int64_t blank, const int64_t reduction, bool zero_infinity, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::CTCLossOptions().blank(blank).zero_infinity(zero_infinity);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::CTCLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_CTCLoss_forward(const NNModule module, const Tensor log_probs, const Tensor targets, const Tensor input_lengths, const Tensor target_lengths)
{
    CATCH_TENSOR((*module)->as<torch::nn::CTCLoss>()->forward(*log_probs, *targets, *input_lengths, *target_lengths));
}

NNModule THSNN_NLLLoss_ctor(const Tensor weight, const int64_t ignore_index, const bool has_ii, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::NLLLossOptions();
        ApplyReduction(opts, reduction);
        if (has_ii)
            opts = opts.ignore_index(ignore_index);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = create_module<torch::nn::NLLLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_NLLLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::NLLLoss>()->forward(*tensor, *target));
}

NNModule THSNN_PoissonNLLLoss_ctor(const bool log_input, const bool full, double eps, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::PoissonNLLLossOptions().log_input(log_input).eps(eps).full(full);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::PoissonNLLLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_PoissonNLLLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::PoissonNLLLoss>()->forward(*tensor, *target));
}

NNModule THSNN_KLDivLoss_ctor(const bool log_target, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::KLDivLossOptions().log_target(log_target);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::KLDivLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_KLDivLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::KLDivLoss>()->forward(*tensor, *target));
}

NNModule THSNN_BCELoss_ctor(const Tensor weight, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BCELossOptions();
        ApplyReduction(opts, reduction);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = create_module<torch::nn::BCELossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BCELoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::BCELoss>()->forward(*tensor, *target));
}

NNModule THSNN_BCEWithLogitsLoss_ctor(const Tensor weight, const int64_t reduction, const Tensor pos_weight, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BCEWithLogitsLossOptions();
        ApplyReduction(opts, reduction);
        if (weight != NULL)
            opts = opts.weight(*weight);
        if (pos_weight != NULL)
            opts = opts.pos_weight(*pos_weight);
        res = create_module<torch::nn::BCEWithLogitsLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BCEWithLogitsLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::BCEWithLogitsLoss>()->forward(*tensor, *target));
}

NNModule THSNN_MarginRankingLoss_ctor(const double margin, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MarginRankingLossOptions().margin(margin);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::MarginRankingLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MarginRankingLoss_forward(const NNModule module, const Tensor input1, const Tensor input2, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::MarginRankingLoss>()->forward(*input1, *input2, *target));
}

NNModule THSNN_HingeEmbeddingLoss_ctor(const double margin, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::HingeEmbeddingLossOptions().margin(margin);
    ApplyReduction(opts, reduction);
    res = create_module<torch::nn::HingeEmbeddingLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_HingeEmbeddingLoss_forward(const NNModule module, const Tensor input, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::HingeEmbeddingLoss>()->forward(*input, *target));
}

NNModule THSNN_HuberLoss_ctor(const double delta, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::HuberLossOptions().delta(delta);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::HuberLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_HuberLoss_forward(const NNModule module, const Tensor input, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::HuberLoss>()->forward(*input, *target));
}

NNModule THSNN_MultiLabelMarginLoss_ctor(const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MultiLabelMarginLossOptions();
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::MultiLabelMarginLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MultiLabelMarginLoss_forward(const NNModule module, const Tensor input, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::MultiLabelMarginLoss>()->forward(*input, *target));
}

NNModule THSNN_SmoothL1Loss_ctor(const double beta, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SmoothL1LossOptions().beta(beta);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::SmoothL1LossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_SmoothL1Loss_forward(const NNModule module, const Tensor input, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::SmoothL1Loss>()->forward(*input, *target));
}

NNModule THSNN_SoftMarginLoss_ctor(const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SoftMarginLossOptions();
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::SoftMarginLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_SoftMarginLoss_forward(const NNModule module, const Tensor input, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::SoftMarginLoss>()->forward(*input, *target));
}

NNModule THSNN_MultiLabelSoftMarginLoss_ctor(const Tensor weight, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MultiLabelSoftMarginLossOptions();
        ApplyReduction(opts, reduction);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = create_module<torch::nn::MultiLabelSoftMarginLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MultiLabelSoftMarginLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::MultiLabelSoftMarginLoss>()->forward(*tensor, *target));
}

NNModule THSNN_CosineEmbeddingLoss_ctor(const double margin, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::CosineEmbeddingLossOptions().margin(margin);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::CosineEmbeddingLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_CosineEmbeddingLoss_forward(const NNModule module, const Tensor input1, const Tensor input2, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::CosineEmbeddingLoss>()->forward(*input1, *input2, *target));
}

NNModule THSNN_MultiMarginLoss_ctor(const int64_t p, double margin, const Tensor weight, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MultiMarginLossOptions().p(0).margin(margin);
        ApplyReduction(opts, reduction);
        if (weight != NULL)
            opts = opts.weight(*weight);
        res = create_module<torch::nn::MultiMarginLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MultiMarginLoss_forward(const NNModule module, const Tensor tensor, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::MultiMarginLoss>()->forward(*tensor, *target));
}

NNModule THSNN_TripletMarginLoss_ctor(const int64_t p, double margin, const bool swap, const double eps, const int64_t reduction, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::TripletMarginLossOptions().p(0).margin(margin).swap(swap).eps(eps);
        ApplyReduction(opts, reduction);
        res = create_module<torch::nn::TripletMarginLossImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_TripletMarginLoss_forward(const NNModule module, const Tensor anchor, const Tensor positive, const Tensor negative, const Tensor target)
{
    CATCH_TENSOR((*module)->as<torch::nn::TripletMarginLoss>()->forward(*anchor, *positive, *negative));
}
