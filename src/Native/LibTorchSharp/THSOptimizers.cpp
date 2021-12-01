// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

void THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length))
{
    auto parameters = (*optimizer)->parameters();
    Tensor* result = allocator(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result[i] = ResultTensor(parameters[i]);
    }
}

Tensor THSNN_Optimizer_step(const Optimizer optimizer, Tensor(*loss_closure)())
{
    CATCH_TENSOR((loss_closure == nullptr) ? (*optimizer)->step() : (*optimizer)->step([loss_closure]() -> at::Tensor { return *(loss_closure()); }));
}

void THSNN_Optimizer_zero_grad(const Optimizer optimizer)
{
    (*optimizer)->zero_grad();
    auto defaults = (*optimizer)->defaults();
}

Optimizer THSNN_Adagrad_ctor(const Tensor* parameters, const int length, const double learning_rate, const double lr_decay, const double weight_decay, const double initial_accumulator_value, const double eps)
{
    auto params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::AdagradOptions(learning_rate)
        .lr_decay(lr_decay)
        .weight_decay(weight_decay)
        .initial_accumulator_value(initial_accumulator_value)
        .eps(eps);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adagrad>(torch::optim::Adagrad(params, options)));
}

Optimizer THSNN_Adam_ctor(const Tensor* parameters, const int length, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::AdamOptions(learning_rate)
        .betas(std::make_tuple(beta1, beta2))
        .eps(eps)
        .weight_decay(weight_decay)
        .amsgrad(amsgrad);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, options)));
}

Optimizer THSNN_AdamW_ctor(const Tensor* parameters, const int length, const double learning_rate, const double beta1, const double beta2, const double eps, const double weight_decay, const bool amsgrad)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::AdamWOptions(learning_rate)
        .betas(std::make_tuple(beta1, beta2))
        .eps(eps)
        .weight_decay(weight_decay)
        .amsgrad(amsgrad);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::AdamW>(torch::optim::AdamW(params, options)));
}

Optimizer THSNN_LBFGS_ctor(const Tensor* parameters, const int length, const double learning_rate, const int64_t max_iter, const int64_t max_eval, const double tolerange_grad, const double tolerance_change, const int64_t history_size)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::LBFGSOptions(learning_rate)
        .max_iter(max_iter)
        .max_eval(max_eval)
        .tolerance_grad(tolerange_grad)
        .tolerance_change(tolerance_change)
        .history_size(history_size);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::LBFGS>(torch::optim::LBFGS(params, options)));
}

Optimizer THSNN_RMSprop_ctor(const Tensor* parameters, const int length, const double learning_rate, const double alpha, const double eps, const double weight_decay, const double momentum, const bool centered)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);

    auto options = torch::optim::RMSpropOptions(learning_rate)
        .alpha(alpha)
        .eps(eps)
        .weight_decay(weight_decay)
        .momentum(momentum)
        .centered(centered);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::RMSprop>(torch::optim::RMSprop(params, options)));
}

Optimizer THSNN_SGD_ctor(const Tensor* parameters, const int length, const double learning_rate, const double momentum, const double dampening, const double weight_decay, const bool nesterov)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto opts = torch::optim::SGDOptions(learning_rate)
        .momentum(momentum)
        .dampening(dampening)
        .weight_decay(weight_decay)
        .nesterov(nesterov);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::SGD>(torch::optim::SGD(params, opts)));
}


// Scheduler integration

template<typename OptionsType>
void SetLearningRate(const Optimizer optimizer, const double lr)
{
    auto options = dynamic_cast<OptionsType*>(&(*optimizer)->defaults());
    options->lr(lr);

    auto& pgs = (*optimizer)->param_groups();
    for (auto pg = pgs.begin(); pg < pgs.end(); ++pg)
    {
        options = dynamic_cast<OptionsType*>(&(pg->options()));
        options->lr(lr);
    }
}

template<typename OptionsType>
void SetMomentum(const Optimizer optimizer, const double momentum)
{
    auto options = dynamic_cast<OptionsType*>(&(*optimizer)->defaults());
    options->momentum(momentum);

    auto& pgs = (*optimizer)->param_groups();
    for (auto pg = pgs.begin(); pg < pgs.end(); ++pg)
    {
        options = dynamic_cast<OptionsType*>(&(pg->options()));
        options->momentum(momentum);
    }
}

template<typename OptionsType>
void SetBetas(const Optimizer optimizer, const double beta1, const double beta2)
{
    auto betas = std::make_tuple(beta1, beta2);

    auto options = dynamic_cast<OptionsType*>(&(*optimizer)->defaults());
    options->betas(betas);

    auto& pgs = (*optimizer)->param_groups();
    for (auto pg = pgs.begin(); pg < pgs.end(); ++pg)
    {
        options = dynamic_cast<OptionsType*>(&(pg->options()));
        options->betas(betas);
    }
}

void THSNN_Adagrad_set_lr(const Optimizer optimizer, const double lr)
{
    SetLearningRate<torch::optim::AdagradOptions>(optimizer, lr);
}

void THSNN_Adam_set_lr(const Optimizer optimizer, const double lr)
{
    SetLearningRate<torch::optim::AdamOptions>(optimizer, lr);
}

void THSNN_AdamW_set_lr(const Optimizer optimizer, const double lr)
{
    SetLearningRate<torch::optim::AdamWOptions>(optimizer, lr);
}

void THSNN_RMSprop_set_lr(const Optimizer optimizer, const double lr)
{
    SetLearningRate<torch::optim::RMSpropOptions>(optimizer, lr);
}

void THSNN_LBFGS_set_lr(const Optimizer optimizer, const double lr)
{
    SetLearningRate<torch::optim::LBFGSOptions>(optimizer, lr);
}

void THSNN_SGD_set_lr(const Optimizer optimizer, const double lr)
{
    SetLearningRate<torch::optim::SGDOptions>(optimizer, lr);
}

void THSNN_Optimizer_dispose(const Optimizer optimizer)
{
    delete optimizer; // NOTE: this reduces the ref count on the shared_ptr
}

void THSNN_Adam_set_betas(const Optimizer optimizer, double beta1, double beta2)
{
    SetBetas<torch::optim::AdamOptions>(optimizer, beta1, beta2);
}

void THSNN_AdamW_set_betas(const Optimizer optimizer, double beta1, double beta2)
{
    SetBetas<torch::optim::AdamWOptions>(optimizer, beta1, beta2);
}

void THSNN_RMSprop_set_momentum(const Optimizer optimizer, double momentum)
{
    SetMomentum<torch::optim::RMSpropOptions>(optimizer, momentum);
}

void THSNN_SGD_set_momentum(const Optimizer optimizer, double momentum)
{
    SetMomentum<torch::optim::SGDOptions>(optimizer, momentum);
}
