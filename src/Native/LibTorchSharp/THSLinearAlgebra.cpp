// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSLinalg_cholesky(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::cholesky(*tensor));
}

Tensor THSLinalg_det(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::det(*tensor));
}

Tensor THSLinalg_slogdet(const Tensor tensor, Tensor* logabsdet)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::slogdet(*tensor););
    *logabsdet = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_eig(const Tensor tensor, Tensor* eigenvectors)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::eig(*tensor););
    *eigenvectors = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_eigh(const Tensor tensor, const char UPLO, Tensor* eigenvectors)
{
    std::string _uplo;
    _uplo.push_back(UPLO);
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::eigh(*tensor, _uplo););
    *eigenvectors = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_eigvals(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::eigvals(*tensor));
}

Tensor THSLinalg_eigvalsh(const Tensor tensor, const char UPLO)
{
    std::string _uplo;
    _uplo.push_back(UPLO);
    CATCH_TENSOR(torch::linalg::eigvalsh(*tensor, _uplo));
}

Tensor THSLinalg_inv(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::inv(*tensor));
}

Tensor THSLinalg_lstsq_none(const Tensor A, const Tensor B, Tensor* residuals, Tensor* rank, Tensor* singular_values)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::lstsq(*A, *B, c10::nullopt, c10::nullopt););
    *residuals = ResultTensor(std::get<1>(res));
    *rank = ResultTensor(std::get<2>(res));
    *singular_values = ResultTensor(std::get<3>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_lstsq_rcond(const Tensor A, const Tensor B, const double rcond, Tensor* residuals, Tensor* rank, Tensor* singular_values)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::lstsq(*A, *B, rcond, c10::nullopt););
    *residuals = ResultTensor(std::get<1>(res));
    *rank = ResultTensor(std::get<2>(res));
    *singular_values = ResultTensor(std::get<3>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_matrix_rank(const Tensor tensor, const double tol, const bool has_tol, const bool hermitian)
{
    if (has_tol)
    {
        CATCH_TENSOR(torch::linalg::matrix_rank(*tensor, tol, hermitian));
    }
    else
    {
        CATCH_TENSOR(torch::linalg::matrix_rank(*tensor, c10::nullopt, hermitian));
    }
}

Tensor THSLinalg_matrix_power(const Tensor tensor, const int64_t n)
{
    CATCH_TENSOR(torch::linalg::matrix_power(*tensor, n));
}

Tensor THSLinalg_multi_dot(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::linalg::multi_dot(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSLinalg_norm_str(const Tensor tensor, const char* p, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::norm(*tensor, p, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_norm_float(const Tensor tensor, const double p, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::norm(*tensor, p, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_norm_int(const Tensor tensor, const int p, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::norm(*tensor, p, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_norm_opt(const Tensor tensor, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::norm(*tensor, c10::nullopt, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_pinv(const Tensor tensor, const double rcond, const bool hermitian)
{
    CATCH_TENSOR(torch::linalg::pinv(*tensor, rcond, hermitian));
}

Tensor THSLinalg_qr(const Tensor tensor, const char mode, Tensor* R)
{
    std::tuple<at::Tensor, at::Tensor> res;
    if (mode == 0) {
        CATCH(res = torch::linalg_qr(*tensor, "reduced"););
    }
    else if (mode == 1) {
        CATCH(res = torch::linalg_qr(*tensor, "complete"););
    }
    else {
        CATCH(res = torch::linalg_qr(*tensor, "r"););
    }
    *R = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));

}

Tensor THSLinalg_solve(const Tensor tensor, Tensor other)
{
    CATCH_TENSOR(torch::linalg::solve(*tensor, *other));
}

Tensor THSLinalg_svd(const Tensor tensor, const bool full_matrices, Tensor* S, Tensor* Vh)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg_svd(*tensor, full_matrices););
    *S = ResultTensor(std::get<1>(res));
    *Vh = ResultTensor(std::get<2>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_tensorinv(const Tensor tensor, const int64_t ind)
{
    CATCH_TENSOR(torch::linalg::tensorinv(*tensor, ind));
}

Tensor THSLinalg_tensorsolve(const Tensor tensor, Tensor other, const int64_t* dim, const int dim_length)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::tensorsolve(*tensor, *other, dims));
}

Tensor THSTensor_cholesky(const Tensor tensor, const bool upper)
{
    CATCH_TENSOR(tensor->cholesky(upper));
}

Tensor THSTensor_cholesky_inverse(const Tensor tensor, const bool upper)
{
    CATCH_TENSOR(tensor->cholesky_inverse(upper));
}

Tensor THSTensor_cholesky_solve(const Tensor tensor, const Tensor tensor2, const bool upper)
{
    CATCH_TENSOR(tensor->cholesky_solve(*tensor2, upper));
}

Tensor THSTensor_diag(const Tensor tensor, const int64_t diagonal)
{
    CATCH_TENSOR(tensor->diag(diagonal));
}

Tensor THSTensor_diagflat(const Tensor tensor, const int64_t offset)
{
    CATCH_TENSOR(tensor->diagflat(offset));
}

Tensor THSTensor_diagonal(const Tensor tensor, const int64_t offset, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->diagonal(offset, dim1, dim2));
}

Tensor THSTensor_kron(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->kron(*right));
}

Tensor THSTensor_lcm(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->lcm(*other));
}

Tensor THSTensor_lcm_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->lcm_(*other));
}

Tensor THSTensor_lgamma(const Tensor tensor)
{
    CATCH_TENSOR(tensor->lgamma());
}

Tensor THSTensor_norm(const Tensor tensor, float p)
{
    CATCH_TENSOR(tensor->norm(p));
}

Tensor THSTensor_norm_along_dimension(const Tensor tensor, const int64_t dim, const bool keepdim, float p)
{
    CATCH_TENSOR(tensor->norm(p, dim, keepdim));
}

Tensor THSTensor_t(const Tensor tensor)
{
    CATCH_TENSOR(tensor->t());
}

