// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSLinalg_cholesky(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::cholesky(*tensor));
}

Tensor THSLinalg_cholesky_ex(const Tensor tensor, bool check_errors, Tensor* info)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg_cholesky_ex(*tensor, check_errors););
    *info = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_cond_int(const Tensor tensor, const int p)
{
    CATCH_TENSOR(torch::linalg_cond(*tensor, p));
}

Tensor THSLinalg_cond_float(const Tensor tensor, const double p)
{
    CATCH_TENSOR(torch::linalg_cond(*tensor, p));
}

Tensor THSLinalg_cond_str(const Tensor tensor, const char* p)
{
    CATCH_TENSOR(p != nullptr ? torch::linalg_cond(*tensor, p) : torch::linalg_cond(*tensor));
}

Tensor THSLinalg_cond_none(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg_cond(*tensor));
}

Tensor THSLinalg_cross(const Tensor input, const Tensor other, const int64_t dim)
{
    CATCH_TENSOR(torch::linalg_cross(*input, *other, dim));
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


Tensor THSTensor_eig(const Tensor tensor, bool vectors, Tensor* eigenvectors)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = tensor->eig(vectors););
    if (vectors) {
        *eigenvectors = ResultTensor(std::get<1>(res));
    }
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

Tensor THSLinalg_inv_ex(const Tensor tensor, bool check_errors, Tensor* info)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg_inv_ex(*tensor, check_errors););
    *info = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
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

Tensor THSLinalg_lu_factor(const Tensor A, const bool pivot, Tensor* pivots)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::lu_factor(*A, pivot););
    *pivots = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_matrix_norm(const Tensor tensor, const Scalar ord, const int64_t* dim, const int dim_length, const bool keepdim)
{
    auto dims = c10::ArrayRef<int64_t>(dim, dim_length);
    CATCH_TENSOR(torch::linalg::matrix_norm(*tensor, *ord, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_matrix_norm_fronuc(const Tensor tensor, const int8_t fronuc, const int64_t* dim, const int dim_length, const bool keepdim)
{
    auto dims = c10::ArrayRef<int64_t>(dim, dim_length);
    CATCH_TENSOR(torch::linalg::matrix_norm(*tensor, (fronuc == 0) ? "fro" : "nuc", dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_vector_norm(const Tensor tensor, const Scalar ord, const int64_t* dim, const int dim_length, const bool keepdim)
{
    auto dims = c10::ArrayRef<int64_t>(dim, dim_length);
    CATCH_TENSOR(torch::linalg::vector_norm(*tensor, *ord, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_matrix_rank(const Tensor tensor, const double atol, const bool has_atol, const double rtol, const bool has_rtol, const bool hermitian)
{
    auto atol_ = has_atol ? atol : c10::optional<double>();
    auto rtol_ = has_rtol ? rtol : c10::optional<double>();

    CATCH_TENSOR(torch::linalg::matrix_rank(*tensor, atol_, rtol_, hermitian));
}

Tensor THSLinalg_matrix_rank_tensor(const Tensor tensor, const Tensor atol, const Tensor rtol, const bool hermitian)
{
    const c10::optional<at::Tensor> atol_ = atol != nullptr ? *atol : c10::optional<at::Tensor>();
    const c10::optional<at::Tensor> rtol_ = rtol != nullptr ? *rtol : c10::optional<at::Tensor>();

    CATCH_TENSOR(torch::linalg::matrix_rank(*tensor, atol_, rtol_, hermitian));
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

Tensor THSLinalg_pinv(const Tensor tensor, const double atol, const bool has_atol, const double rtol, const bool has_rtol, const bool hermitian)
{
    auto atol_ = has_atol ? atol : c10::optional<double>();
    auto rtol_ = has_rtol ? rtol : c10::optional<double>();

    CATCH_TENSOR(torch::linalg_pinv(*tensor, atol_, rtol_, hermitian));
}

Tensor THSLinalg_pinv_tensor(const Tensor tensor, const Tensor atol, const Tensor rtol, const bool hermitian)
{
    const c10::optional<at::Tensor> atol_ = atol != nullptr ? *atol : c10::optional<at::Tensor>();
    const c10::optional<at::Tensor> rtol_ = rtol != nullptr ? *rtol : c10::optional<at::Tensor>();

    CATCH_TENSOR(torch::linalg_pinv(*tensor, atol_, rtol_, hermitian));
}

Tensor THSLinalg_pinverse(const Tensor tensor, const double rcond, const bool hermitian)
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
    CATCH(res = torch::linalg::svd(*tensor, full_matrices););
    *S = ResultTensor(std::get<1>(res));
    *Vh = ResultTensor(std::get<2>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_svdvals(const Tensor tensor)
{
    CATCH_TENSOR(res = torch::linalg::svdvals(*tensor));
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

Tensor THSTensor_kthvalue(const Tensor input, int64_t k, int64_t dim, bool keepdim, Tensor* out)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = input->kthvalue(k, dim, keepdim););
    *out = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));

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

Tensor THSTensor_lu(const Tensor tensor, bool pivot, bool get_infos, Tensor* infos, Tensor* pivots)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg_lu_factor_ex(*tensor, pivot, false););
    *infos = get_infos ? ResultTensor(std::get<2>(res)) : nullptr;
    *pivots = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_lu_solve(const Tensor tensor, const Tensor LU_data, const Tensor LU_pivots)
{
    CATCH_TENSOR(tensor->lu_solve(*LU_data, *LU_pivots));
}

Tensor THSTensor_lu_unpack(const Tensor LU_data, const Tensor LU_pivots, bool unpack_data, bool unpack_pivots, Tensor* L, Tensor* U)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;
    CATCH(res = torch::lu_unpack(*LU_data, *LU_pivots, unpack_data, unpack_pivots););
    *U = ResultTensor(std::get<2>(res));
    *L = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_norm_along_dimension(const Tensor tensor, const int64_t dim, const bool keepdim, float p)
{
    CATCH_TENSOR(tensor->norm(p, dim, keepdim));
}

Tensor THSTensor_t(const Tensor tensor)
{
    CATCH_TENSOR(tensor->t());
}

