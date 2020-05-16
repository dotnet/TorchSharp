// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include <string>

#include "TH/THGeneral.h"
#include "torch/torch.h"

extern thread_local char *torch_last_err;

typedef torch::Tensor *Tensor;
typedef torch::Scalar *Scalar;
typedef std::shared_ptr<torch::nn::Module>* NNModule;
typedef std::shared_ptr<torch::nn::AnyModule> * NNAnyModule;
typedef std::shared_ptr<torch::nn::SequentialImpl> *NNSequential;
//typedef std::shared_ptr<torch::nn::SequentialImpl>* NNSequential;
typedef std::shared_ptr<torch::optim::Optimizer> * Optimizer;
//typedef std::shared_ptr<torch::jit::script::Module> * JITModule;
//typedef std::shared_ptr<c10::Type> * JITType;
//typedef std::shared_ptr<torch::jit::DimensionedTensorType>* JITDimensionedTensorType;

#define THS_API TH_API

#define CATCH(x) \
  try { \
    torch_last_err = 0; \
    x \
  } catch (const c10::Error e) { \
      torch_last_err = strdup(e.what()); \
  }

#define CATCH_RETURN_RES(ty, stmt) \
    ty res; \
    CATCH(  \
        stmt;  \
    );  \
    return res;

#define CATCH_RETURN(ty, expr) CATCH_RETURN_RES(ty, res = expr)

#define CATCH_RETURN_TENSOR(expr) \
    at::Tensor res; \
    CATCH(  \
        res = expr;  \
    );  \
    return new torch::Tensor(res);


// Utility method used to built sharable strings.
const char * make_sharable_string(const std::string str);

// Method concerting arrays of tensor pointers into arrays of tensors.
template<class T>
std::vector<T> toTensors(torch::Tensor ** tensorPtrs, const int length)
{
    std::vector<T> tensors;

    for (int i = 0; i < length; i++)
    {
        tensors.push_back(*tensorPtrs[i]);
    }

    return tensors;
}
