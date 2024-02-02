// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include <string>

#include "torch/torch.h"

extern thread_local char *torch_last_err;

typedef torch::Tensor *Tensor;
typedef torch::Scalar *Scalar;
typedef torch::Generator* Generator;
typedef c10::Storage* Storage;
typedef torch::nn::utils::rnn::PackedSequence* PackedSequence;

typedef std::shared_ptr<torch::autograd::SavedVariable> * SavedVariable;
typedef std::shared_ptr<torch::nn::Module> * NNModule;
typedef std::shared_ptr<torch::nn::AnyModule> * NNAnyModule;
typedef std::shared_ptr<torch::optim::Optimizer> * Optimizer;
typedef std::shared_ptr<torch::jit::CompilationUnit> * JITCompilationUnit;
typedef std::shared_ptr<torch::jit::Module>* JITModule;
typedef std::shared_ptr<torch::jit::Method>* JITMethod;
typedef std::shared_ptr<torch::jit::Function> * JITFunction;
typedef std::shared_ptr<c10::Type> * JITType;
typedef std::shared_ptr<c10::TensorType>* JITTensorType;

struct TensorArray {
    Tensor *array;
    int64_t size;
};



//typedef std::shared_ptr<torch::jit::DimensionedTensorType>* JITDimensionedTensorType;

#define THS_API TH_API

#define CATCH(x) \
  try { \
    torch_last_err = 0; \
    x \
  } catch (const c10::Error e) { \
      torch_last_err = strdup(e.what()); \
  } catch (const std::runtime_error e) { \
      torch_last_err = strdup(e.what()); \
  }

#define CATCH_RETURN_RES(ty, dflt, stmt) \
    ty res = dflt; \
    CATCH(  \
        stmt;  \
    );  \
    return res;

#define CATCH_RETURN(ty, dflt, expr) CATCH_RETURN_RES(ty, dflt, res = expr)
#define CATCH_RETURN_NNModule(stmt) CATCH_RETURN_RES(NNModule, nullptr, stmt)
#define CATCH_RETURN_Tensor(stmt) CATCH_RETURN_RES(Tensor, nullptr, stmt)

// Return undefined tensors as nullptr to C#
inline Tensor ResultTensor(const at::Tensor & res)
{
    if (res.defined())
        return new torch::Tensor(res);
    else
        return nullptr;
}

#define CATCH_TENSOR(expr) \
    at::Tensor res = at::Tensor(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);

#define CATCH_TENSORS_2(expr) \
    at::Tensor fst = at::Tensor();  \
    at::Tensor snd = at::Tensor();  \
    CATCH(  \
        std::tie(fst,snd) = expr;  \
    );  \
    res1 = ResultTensor(fst); \
    res2 = ResultTensor(snd);     

#define CATCH_SCALAR(expr) \
    at::Scalar res = at::Scalar(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);


// Utility method used to built sharable strings.
const char * make_sharable_string(const std::string str);

// Method concerting arrays of tensor pointers into arrays of tensors.
template<class T>
std::vector<T> toTensors(torch::Tensor ** tensorPtrs, const int length)
{
    std::vector<T> tensors;

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            tensors.push_back(*tensorPtrs[i]);
        }
    }
    return tensors;
}

// Utilities for NN namespace.

template <typename T>
Tensor get_weight(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight);
}

template <typename T>
void set_weight(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight = *weights;
    );
}

template <typename T>
Tensor get_bias(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias);
}

template <typename T>
void set_bias(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias = *bias;
    );
}

template <typename T>
Tensor get_weight_ih(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight_ih);
}

template <typename T>
Tensor get_weight_hh(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight_hh);
}

template <typename T>
void set_weight_ih(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight_ih = *weights;
    );
}

template <typename T>
void set_weight_hh(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight_hh = *weights;
    );
}

template <typename T>
Tensor get_bias_ih(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias_ih);
}

template <typename T>
Tensor get_bias_hh(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias_hh);
}

template <typename T>
void set_bias_ih(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias_ih = *bias;
    );
}

template <typename T>
void set_bias_hh(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias_hh = *bias;
    );
}

#define WIH_BASE 0
#define WHH_BASE 1
#define BIH_BASE 2
#define BHH_BASE 3

template <typename T>
Tensor get_weight_ih(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[WIH_BASE + idx * 4]);
}

template <typename T>
Tensor get_weight_hh(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[WHH_BASE + idx * 4]);
}

template <typename T>
void set_weight_ih(const NNModule module, const Tensor weights, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[WIH_BASE + idx * 4] = *weights;
    );
}

template <typename T>
void set_weight_hh(const NNModule module, const Tensor weights, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[WHH_BASE + idx * 4] = *weights;
    );
}

template <typename T>
Tensor get_bias_ih(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[BIH_BASE + idx * 4]);
}

template <typename T>
Tensor get_bias_hh(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[BHH_BASE + idx * 4]);
}

template <typename T>
void set_bias_ih(const NNModule module, const Tensor bias, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[BIH_BASE + idx * 4] = *bias;
    );
}

template <typename T>
void set_bias_hh(const NNModule module, const Tensor bias, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[BHH_BASE + idx * 4] = *bias;
    );
}

template<typename TImpl>
NNModule create_module(NNAnyModule* outAsAnyModule)
{
    auto mod = std::make_shared<TImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != nullptr)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<TImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    return new std::shared_ptr<torch::nn::Module>(mod);
}

template<typename TImpl, typename TOptions>
NNModule create_module(const TOptions& opts, NNAnyModule* outAsAnyModule)
{
    auto mod = std::make_shared<TImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != nullptr)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<TImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    return new std::shared_ptr<torch::nn::Module>(mod);
}

inline
torch::nn::init::NonlinearityType get_nl_type(const int64_t nl)
{
    switch (nl)
    {
    default:
    case 0:  return torch::kLinear;
    case 1:  return torch::kConv1D;
    case 2:  return torch::kConv2D;
    case 3:  return torch::kConv3D;
    case 4:  return torch::kConvTranspose1D;
    case 5:  return torch::kConvTranspose2D;
    case 6:  return torch::kConvTranspose3D;
    case 7:  return torch::kSigmoid;
    case 8:  return torch::kTanh;
    case 9:  return torch::kReLU;
    case 10: return torch::kLeakyReLU;
    }
}