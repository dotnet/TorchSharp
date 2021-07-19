// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

int THSTensor_allclose(const Tensor left, const Tensor right, double rtol, double atol, bool equal_nan)
{
    CATCH_RETURN(int, 0, left->allclose(*right, rtol, atol, equal_nan));
}

Tensor THSTensor_all(const Tensor tensor)
{
    CATCH_TENSOR(tensor->all());
}

Tensor THSTensor_all_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->all(dim, keepdim));
}

Tensor THSTensor_amax(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim)
{
    CATCH_TENSOR(tensor->amax(c10::IntArrayRef(dimensions, length), keepdim));
}

Tensor THSTensor_amin(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim)
{
    CATCH_TENSOR(tensor->amin(c10::IntArrayRef(dimensions, length), keepdim));
}

Tensor THSTensor_angle(const Tensor tensor)
{
    CATCH_TENSOR(tensor->angle());
}

Tensor THSTensor_any(const Tensor tensor)
{
    CATCH_TENSOR(tensor->any());
}

Tensor THSTensor_any_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->any(dim, keepdim));
}
Tensor THSTensor_argmax(const Tensor tensor)
{
    CATCH_TENSOR(tensor->argmax());
}

Tensor THSTensor_argmax_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->argmax(dim, keepdim));
}

Tensor THSTensor_argmin(const Tensor tensor)
{
    CATCH_TENSOR(tensor->argmin());
}

Tensor THSTensor_argmin_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->argmin(dim, keepdim));
}

Tensor THSTensor_atleast_1d(const Tensor tensor)
{
    CATCH_TENSOR(torch::atleast_1d(*tensor));
}

Tensor THSTensor_argsort(const Tensor tensor, const int64_t dim, bool descending)
{
    CATCH_TENSOR(tensor->argsort(dim, descending));
}

Tensor THSTensor_atleast_2d(const Tensor tensor)
{
    CATCH_TENSOR(torch::atleast_2d(*tensor));
}

Tensor THSTensor_atleast_3d(const Tensor tensor)
{
    CATCH_TENSOR(torch::atleast_3d(*tensor));
}

void THSTensor_backward(Tensor tensor)
{
    CATCH(
        tensor->backward();
    )
}

Tensor THSTensor_bincount(const Tensor tensor, const Tensor weights, const int64_t minlength)
{
    CATCH_TENSOR(tensor->bincount((weights ? *weights : at::Tensor()), minlength));
}

Tensor THSTensor_block_diag(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::block_diag(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_broadcast_to(const Tensor tensor, const int64_t* shape, const int shape_len)
{
    CATCH_TENSOR(tensor->broadcast_to(at::ArrayRef<int64_t>(shape, shape_len)));
}

EXPORT_API(Tensor) THSTensor_bucketize(const Tensor tensor, const Tensor boundaries, const bool out_int32, const bool right)
{
    CATCH_TENSOR(torch::bucketize(*tensor, *boundaries, out_int32, right));
}

double THSTensor_clip_grad_norm_(const Tensor* tensors, const int length, const double max_norm, const double norm_type)
{
    double res = 0.0;
    CATCH(
        res = torch::nn::utils::clip_grad_norm_(toTensors<at::Tensor>((torch::Tensor**)tensors, length), max_norm, norm_type);
    );
    return res;
}

Tensor THSTensor_cat(const Tensor* tensors, const int length, const int64_t dim)
{
    CATCH_TENSOR(torch::cat(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_celu(const Tensor tensor)
{
    CATCH_TENSOR(torch::celu(*tensor));
}

Tensor THSTensor_celu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::celu_(*tensor));
}

void THSTensor_chunk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t chunks, const int64_t dim)
{
    CATCH(
        auto res = tensor->chunk(chunks, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_clamp(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp(*min, *max));
}

Tensor THSTensor_clamp_(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp_(*min, *max));
}

Tensor THSTensor_clamp_max(const Tensor tensor, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp_max(*max));
}

Tensor THSTensor_clamp_max_(const Tensor tensor, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp_max_(*max));
}

Tensor THSTensor_clamp_min(const Tensor tensor, const Scalar min)
{
    CATCH_TENSOR(tensor->clamp_min(*min));
}

Tensor THSTensor_clamp_min_(const Tensor tensor, const Scalar min)
{
    CATCH_TENSOR(tensor->clamp_min_(*min));
}

Tensor THSTensor_clone(const Tensor tensor)
{
    CATCH_TENSOR(tensor->clone());
}

Tensor THSTensor_copy_(const Tensor input, const Tensor other, const bool non_blocking)
{
    CATCH_TENSOR(input->copy_(*other, non_blocking));
}

Tensor THSTensor_complex(const Tensor real, const Tensor imag)
{
    CATCH_TENSOR(torch::complex(*real, *imag));
}

Tensor THSTensor_polar(const Tensor abs, const Tensor angle)
{
    CATCH_TENSOR(torch::polar(*abs, *angle));
}

Tensor THSTensor_contiguous(const Tensor tensor)
{
    CATCH_TENSOR(tensor->contiguous());
}

Tensor THSTensor_copysign(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(input->copysign(*other));
}

Tensor THSTensor_cpu(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cpu());
}

Tensor THSTensor_cuda(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cuda());
}

void THSTensor_cummax(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim)
{
    auto max = tensor->cummax(dim);
    Tensor* result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
}

void THSTensor_cummin(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim)
{
    auto max = tensor->cummin(dim);
    Tensor* result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
}

Tensor THSTensor_cumprod(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(has_type ? tensor->cumprod(dim, (c10::ScalarType)dtype) : tensor->cumprod(dim));
}

Tensor THSTensor_cumsum(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(has_type ? tensor->cumsum(dim, (c10::ScalarType)dtype) : tensor->cumsum(dim));
}

void* THSTensor_data(const Tensor tensor)
{
    CATCH_RETURN(void*, NULL, tensor->data_ptr());
}

float THSTensor_data_idx_float16(const Tensor tensor, const int64_t i)
{
    CATCH_RETURN(float, NULL, (float)(tensor->data_ptr<c10::Half>())[i]);
}

float THSTensor_data_idx_bfloat16(const Tensor tensor, const int64_t i)
{
    CATCH_RETURN(float, NULL, (float)(tensor->data_ptr<c10::BFloat16>())[i]);
}

const char* THSTensor_device_str(const Tensor tensor)
{
    auto device = tensor->device();

    return make_sharable_string(device.str());
}

int THSTensor_device_index(const Tensor tensor)
{
    auto device = tensor->device();
    return device.index();
}

int THSTensor_device_type(const Tensor tensor)
{
    auto device = tensor->device();
    return (int)device.type();
}

Tensor THSTensor_diff(const Tensor tensor, const int64_t n, const int64_t dim, const Tensor prepend, const Tensor append)
{
    c10::optional<at::Tensor> prep = prepend != nullptr ? *prepend : c10::optional<at::Tensor>(c10::nullopt);
    c10::optional<at::Tensor> app = append != nullptr ? *append : c10::optional<at::Tensor>(c10::nullopt);
    CATCH_TENSOR(tensor->diff(n, dim, prep, app));
}

void THSTensor_dispose(const Tensor tensor)
{
    delete tensor;
}

Tensor THSTensor_digamma(const Tensor tensor)
{
    CATCH_TENSOR(tensor->digamma());
}

Tensor THSTensor_digamma_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->digamma_());
}

Tensor THSTensor_dist(const Tensor tensor, const Tensor other, const float p)
{
    CATCH_TENSOR(tensor->dist(*other, p));
}

int64_t THSTensor_element_size(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->element_size());
}

Tensor THSTensor_elu(const Tensor tensor, const Scalar alpha, const Scalar scale, const Scalar input_scale)
{
    CATCH_TENSOR(torch::elu(*tensor, *alpha, *scale, *input_scale));
}

Tensor THSTensor_elu_(const Tensor tensor, const Scalar alpha, const Scalar scale, const Scalar input_scale)
{
    CATCH_TENSOR(torch::elu_(*tensor, *alpha, *scale, *input_scale));
}

Tensor THSTensor_expand(const Tensor tensor, const int64_t* sizes, const int length, bool implicit)
{
    CATCH_TENSOR(tensor->expand(at::ArrayRef<int64_t>(sizes, length), implicit));
}

Tensor THSTensor_repeat(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->repeat(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_movedim(const Tensor tensor, const int64_t* src, const int src_len, const int64_t* dst, const int dst_len)
{
    CATCH_TENSOR(tensor->movedim(at::ArrayRef<int64_t>(src, src_len), at::ArrayRef<int64_t>(dst, dst_len)));
}

Tensor THSTensor_count_nonzero(const Tensor tensor, const int64_t* dim, const int dim_len)
{
    CATCH_TENSOR(tensor->count_nonzero(at::ArrayRef<int64_t>(dim, dim_len)));
}

Tensor THSTensor_nonzero(const Tensor tensor)
{
    CATCH_TENSOR(tensor->nonzero());
}


Tensor THSTensor_flip(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->flip(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_fliplr(const Tensor tensor)
{
    CATCH_TENSOR(tensor->fliplr());
}

Tensor THSTensor_flipud(const Tensor tensor)
{
    CATCH_TENSOR(tensor->flipud());
}

Tensor THSTensor_fill_(const Tensor tensor, const Scalar value)
{
    CATCH_TENSOR(tensor->fill_(*value));
}

Tensor THSTensor_frexp(const Tensor tensor, Tensor* exponent)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::frexp(*tensor););
    *exponent = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_gather(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index)
{
    CATCH_TENSOR(torch::gather(*tensor, dim, *index));
}

Tensor THSTensor_gelu(const Tensor tensor)
{
    CATCH_TENSOR(torch::gelu(*tensor));
}

Tensor THSTensor_get1(const Tensor tensor, int64_t index)
{
    CATCH_TENSOR((*tensor)[index]);
}

Tensor THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2)
{
    CATCH_TENSOR((*tensor)[index1][index2]);
}

Tensor THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3]);
}

Tensor THSTensor_get4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3][index4]);
}

Tensor THSTensor_get5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3][index4][index5]);
}

Tensor THSTensor_get6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3][index4][index5][index6]);
}

Tensor THSTensor_gcd(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->gcd(*other));
}

Tensor THSTensor_gcd_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->gcd_(*other));
}

Tensor THSTensor_grad(const Tensor tensor)
{
    Tensor res;
    CATCH(
        torch::Tensor grad = tensor->grad();
    res = grad.defined() ? new torch::Tensor(grad) : NULL;
    );
    return res;
}

Tensor THSTensor_hardsigmoid(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardsigmoid(*tensor));
}

Tensor THSTensor_hardsigmoid_(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardsigmoid_(*tensor));
}

Tensor THSTensor_hardswish(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardswish(*tensor));
}

Tensor THSTensor_hardswish_(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardswish_(*tensor));
}

Tensor THSTensor_hardtanh(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(torch::hardtanh(*tensor, *min, *max));
}

Tensor THSTensor_hardtanh_(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(torch::hardtanh_(*tensor, *min, *max));
}

Tensor THSTensor_heaviside(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(torch::heaviside(*left, *right));
}

Tensor THSTensor_hypot(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(torch::hypot(*left, *right));
}

Tensor THSTensor_i0(const Tensor tensor)
{
    CATCH_TENSOR(torch::i0(*tensor));
}

Tensor THSTensor_igamma(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(torch::igamma(*tensor, *other));
}

Tensor THSTensor_igammac(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(torch::igammac(*tensor, *other));
}

Tensor THSTensor_isclose(const Tensor tensor, const Tensor other, const double rtol, const double atol, const bool equal_nan)
{
    CATCH_TENSOR(torch::isclose(*tensor, *other, rtol, atol, equal_nan));
}


Tensor THSTensor_isinf(const Tensor tensor)
{
    CATCH_TENSOR(torch::isinf(*tensor));
}

Tensor THSTensor_isfinite(const Tensor tensor)
{
    CATCH_TENSOR(torch::isfinite(*tensor));
}

Tensor THSTensor_isneginf(const Tensor tensor)
{
    CATCH_TENSOR(torch::isneginf(*tensor));
}

Tensor THSTensor_isposinf(const Tensor tensor)
{
    CATCH_TENSOR(torch::isposinf(*tensor));
}

Tensor THSTensor_isreal(const Tensor tensor)
{
    CATCH_TENSOR(torch::isreal(*tensor));
}

void completeTensorIndices(const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    at::indexing::TensorIndex* indicesArray,
    const int indicesLength)
{
    // The indexStart encodes the kind of slice being performed for each dimension
    // range INT64_MIN..INT64_MIN+5 is for various singleton cases
    // range INT64_MIN+6 is for slice with absent start
    // range INT64_MIN+7 ... INT64_MIN/4 is for start of slice centered around INT64_MIN/2
    // range INT64_MIN/4+1 ... INT64_MAX is for single (normally a positive integer)
    for (int i = 0; i < indicesLength; i++)
    {
        auto n = indexStarts[i];
        if (n == INT64_MIN) // TensorIndex 'Null'
        {
            at::indexing::TensorIndex idx(c10::nullopt);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 1) // TensorIndex 'False'
        {
            at::indexing::TensorIndex idx(false);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 2) // TensorIndex 'True'
        {
            at::indexing::TensorIndex idx(true);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 3) // TensorIndex '...'
        {
            at::indexing::TensorIndex idx(at::indexing::Ellipsis);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 4) // TensorIndex 'None'
        {
            at::indexing::TensorIndex idx(at::indexing::None);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 5) // TensorIndex by tensor
        {
            at::indexing::TensorIndex idx(*indexTensors[i]);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n > INT64_MIN / 4) // TensorIndex by integer
        {
            at::indexing::TensorIndex idx(n);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else // TensorIndex by Slice
        {
            // slice
            auto start = (n == INT64_MIN + 6) ? c10::optional<int64_t>() : c10::optional<int64_t>(n - INT64_MIN / 2);
            auto end = (indexEnds == NULL || indexEnds[i] == INT64_MIN) ? c10::optional<int64_t>() : c10::optional<int64_t>(indexEnds[i]);
            auto step = (indexSteps == NULL || indexSteps[i] == INT64_MIN) ? c10::optional<int64_t>() : c10::optional<int64_t>(indexSteps[i]);
            at::indexing::TensorIndex idx(at::indexing::Slice(start, end, step));
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
    }
}

Tensor THSTensor_index(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength)
{
    at::indexing::TensorIndex* indicesArray = (at::indexing::TensorIndex*)alloca(indicesLength * sizeof(at::indexing::TensorIndex));
    memset(indicesArray, 0, indicesLength * sizeof(at::indexing::TensorIndex));
    // The indexStart encodes the kind of slice being performed for each dimension
    completeTensorIndices(indexStarts, indexEnds, indexSteps, indexTensors, indicesArray, indicesLength);
    auto indices = at::ArrayRef<at::indexing::TensorIndex>(indicesArray, indicesLength);
    CATCH_TENSOR(tensor->index(indices));
}

Tensor THSTensor_index_put_(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength,
    const Tensor value)
{
    at::indexing::TensorIndex* indicesArray = (at::indexing::TensorIndex*)alloca(indicesLength * sizeof(at::indexing::TensorIndex));
    memset(indicesArray, 0, indicesLength * sizeof(at::indexing::TensorIndex));
    completeTensorIndices(indexStarts, indexEnds, indexSteps, indexTensors, indicesArray, indicesLength);
    auto indices = at::ArrayRef<at::indexing::TensorIndex>(indicesArray, indicesLength);
    CATCH_TENSOR(tensor->index_put_(indices, *value));
}

Tensor THSTensor_index_put_scalar_(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength,
    const Scalar value)
{
    at::indexing::TensorIndex* indicesArray = (at::indexing::TensorIndex*)alloca(indicesLength * sizeof(at::indexing::TensorIndex));
    memset(indicesArray, 0, indicesLength * sizeof(at::indexing::TensorIndex));
    completeTensorIndices(indexStarts, indexEnds, indexSteps, indexTensors, indicesArray, indicesLength);
    auto indices = at::ArrayRef<at::indexing::TensorIndex>(indicesArray, indicesLength);
    CATCH_TENSOR(tensor->index_put_(indices, *value));
}

Tensor THSTensor_index_select(Tensor tensor, int64_t dim, Tensor index)
{
    CATCH_TENSOR(tensor->index_select(dim, *index));
}

Tensor THSTensor_indices(Tensor tensor)
{
    CATCH_TENSOR(tensor->_indices());
}

Tensor THSTensor_inner(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->inner(*right));
}

Tensor THSTensor_inverse(const Tensor tensor)
{
    CATCH_TENSOR(tensor->inverse());
}

int THSTensor_is_sparse(const Tensor tensor)
{
    CATCH_RETURN(int, 0, tensor->is_sparse());
}

Scalar THSTensor_item(const Tensor tensor)
{
    CATCH_RETURN(Scalar, NULL, new torch::Scalar(tensor->item()));
}

Tensor THSTensor_leaky_relu(const Tensor tensor, const Scalar negative_slope)
{
    CATCH_TENSOR(torch::leaky_relu(*tensor, *negative_slope));
}

Tensor THSTensor_leaky_relu_(const Tensor tensor, const Scalar negative_slope)
{
    CATCH_TENSOR(torch::leaky_relu_(*tensor, *negative_slope));
}

Tensor THSTensor_load(const char* location)
{
    CATCH_RETURN_Tensor(
        torch::Tensor tensor;
        torch::load(tensor, location);
        res = ResultTensor(tensor);
    );
}

Tensor THSTensor_log_sigmoid(const Tensor tensor)
{
    CATCH_TENSOR(torch::log_sigmoid(*tensor));
}

//Tensor THSTensor_log_sigmoid_backward(const Tensor tensor)
//{
//    CATCH_TENSOR(torch::log_sigmoid_backward(*tensor));
//}

Tensor THSTensor_lerp(const Tensor tensor, const Tensor end, const Tensor weight)
{
    CATCH_TENSOR(tensor->lerp(*end, *weight));
}

Tensor THSTensor_lerp_(const Tensor tensor, const Tensor end, const Tensor weight)
{
    CATCH_TENSOR(tensor->lerp_(*end, *weight));
}

Tensor THSTensor_masked_fill(const Tensor tensor, const Tensor mask, const Scalar value)
{
    CATCH_TENSOR(tensor->masked_fill(*mask, *value));
}

Tensor THSTensor_masked_fill_(const Tensor tensor, const Tensor mask, const Scalar value)
{
    CATCH_TENSOR(tensor->masked_fill_(*mask, *value));
}

Tensor THSTensor_masked_scatter(const Tensor tensor, const Tensor mask, const Tensor value)
{
    CATCH_TENSOR(tensor->masked_scatter(*mask, *value));
}

Tensor THSTensor_masked_scatter_(const Tensor tensor, const Tensor mask, const Tensor value)
{
    CATCH_TENSOR(tensor->masked_scatter_(*mask, *value));
}

Tensor THSTensor_masked_select(const Tensor tensor, const Tensor mask)
{
    CATCH_TENSOR(tensor->masked_select(*mask));
}

Tensor THSTensor_max(const Tensor tensor)
{
    CATCH_TENSOR(tensor->max());
}

Tensor THSTensor_max_elementwise(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->max(*other));
}

Tensor THSTensor_maximum(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->maximum(*other));
}

void THSTensor_max_along_dimension(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keepdim)
{
    CATCH(
        auto max = tensor->max(dim, keepdim);
        Tensor * result = allocator(2);
        result[0] = new torch::Tensor(std::get<0>(max));
        result[1] = new torch::Tensor(std::get<1>(max));
    )
}

Tensor THSTensor_mean(const Tensor tensor)
{
    CATCH_TENSOR(tensor->mean());
}

Tensor THSTensor_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(
        has_type ?
        tensor->mean(at::ArrayRef<int64_t>(dimensions, length), keepdim, (c10::ScalarType)dtype)
        :
        tensor->mean(at::ArrayRef<int64_t>(dimensions, length), keepdim))
}

Tensor THSTensor_median(const Tensor tensor)
{
    CATCH_TENSOR(tensor->median());
}

void THSTensor_mode(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keep_dim)
{
    CATCH(
        auto res = tensor->mode(dim, keep_dim);
        const size_t sz = 2;
        Tensor * result = allocator(2);
        result[0] = new torch::Tensor(std::get<0>(res));
        result[1] = new torch::Tensor(std::get<1>(res));
    )
}

Tensor THSTensor_quantile(const Tensor tensor, const Tensor q, const int64_t dim, const bool keep_dim)
{
    CATCH_TENSOR(dim == -1 ? tensor->quantile(*q, c10::nullopt, keep_dim) : tensor->quantile(*q, dim, keep_dim));
}

Tensor THSTensor_nanquantile(const Tensor tensor, const Tensor q, const int64_t dim, const bool keep_dim)
{
    CATCH_TENSOR(dim == -1 ? tensor->nanquantile(*q, c10::nullopt, keep_dim) : tensor->nanquantile(*q, dim, keep_dim));
}

Tensor THSTensor_min(const Tensor tensor)
{
    CATCH_TENSOR(tensor->min());
}

Tensor THSTensor_minimmum(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->minimum(*other));
}

Tensor THSTensor_min_elementwise(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->min(*other));
}

void THSTensor_min_along_dimension(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keepdim)
{
    CATCH(
        auto max = tensor->min(dim, keepdim);
        Tensor * result = allocator(2);
        result[0] = new torch::Tensor(std::get<0>(max));
        result[1] = new torch::Tensor(std::get<1>(max));
    )
}

//Tensor THSTensor_median_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
//{
//    CATCH_TENSOR(tensor->median(dim, keepdim));
//}

Tensor THSTensor_msort(const Tensor tensor)
{
    CATCH_TENSOR(tensor->msort());
}

Tensor THSTensor_nansum(const Tensor input)
{
    CATCH_TENSOR(torch::nansum(*input));
}

Tensor THSTensor_nanmedian(const Tensor input)
{
    CATCH_TENSOR(torch::nanmedian(*input));
}

Tensor THSTensor_nan_to_num(const Tensor input, double* _nan, double* _posinf, double* _neginf)
{
    c10::optional<double> nan = (_nan != nullptr) ? *_nan : c10::optional<double>(c10::nullopt);
    c10::optional<double> posinf = (_posinf != nullptr) ? *_posinf : c10::optional<double>(c10::nullopt);
    c10::optional<double> neginf = (_neginf != nullptr) ? *_neginf : c10::optional<double>(c10::nullopt);
    CATCH_TENSOR(torch::nan_to_num(*input, nan, posinf, neginf));
}

Tensor THSTensor_narrow(const Tensor tensor, int64_t dim, int64_t start, int64_t length)
{
    CATCH_TENSOR(tensor->narrow(dim, start, length))
}

Tensor THSTensor_nextafter(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::nextafter(*input, *other));
}

int64_t THSTensor_ndimension(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->ndimension());
}

int64_t THSTensor_numel(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->numel());
}


Tensor THSTensor_outer(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->outer(*right));
}


Tensor THSTensor_permute(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->permute(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_polygamma(const Tensor tensor, int64_t n)
{
    CATCH_TENSOR(tensor->polygamma(n));
}

Tensor THSTensor_polygamma_(const Tensor tensor, int64_t n)
{
    CATCH_TENSOR(tensor->polygamma_(n));
}

Tensor THSTensor_positive(const Tensor tensor)
{
    CATCH_TENSOR(tensor->positive());
}

Tensor THSTensor_prelu(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->prelu(*right));
}

//Tensor THSTensor_prelu_backward(const Tensor grad_output, const Tensor self, const Tensor weight)
//{
//    CATCH_TENSOR(torch::prelu_backward(*grad_output, *self, *weight));
//}

Tensor THSTensor_ravel(const Tensor tensor)
{
    CATCH_TENSOR(torch::ravel(*tensor));
}

Tensor THSTensor_relu(const Tensor tensor)
{
    CATCH_TENSOR(torch::relu(*tensor));
}

Tensor THSTensor_relu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::relu_(*tensor));
}

Tensor THSTensor_relu6(const Tensor tensor)
{
    CATCH_TENSOR(torch::nn::functional::relu6(*tensor));
}

Tensor THSTensor_relu6_(const Tensor tensor)
{
    CATCH_TENSOR(torch::nn::functional::relu6(*tensor, torch::nn::functional::ReLU6FuncOptions().inplace(true)));
}

Tensor THSTensor_renorm(const Tensor tensor, const float p, const int64_t dim, const float maxnorm)
{
    CATCH_TENSOR(tensor->renorm(p, dim, maxnorm));
}

int THSTensor_requires_grad(const Tensor tensor)
{
    CATCH_RETURN(int, 0, tensor->requires_grad());
}

Tensor THSTensor_reshape(const Tensor tensor, const int64_t* shape, const int length)
{
    CATCH_TENSOR(tensor->reshape(at::ArrayRef<int64_t>(shape, length)));
}

void THSTensor_save(const Tensor tensor, const char* location)
{
    CATCH(
        torch::save(*tensor, location);
    );
}

Tensor THSTensor_scatter(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index,
    const Tensor source)
{
    CATCH_TENSOR(torch::scatter(*tensor, dim, *index, *source));
}

Tensor THSTensor_selu(const Tensor tensor)
{
    CATCH_TENSOR(torch::selu(*tensor));
}

Tensor THSTensor_selu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::selu_(*tensor));
}

Tensor THSTensor_silu(const Tensor tensor)
{
    CATCH_TENSOR(torch::silu(*tensor));
}

Tensor THSTensor_silu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::silu_(*tensor));
}

void THSTensor_set1(const Tensor tensor, int64_t index, Scalar value)
{
    CATCH(
        (*tensor)[index] = *value;
    )
}

void THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2] = *value;
    )
}

void THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3] = *value;
    )
}

void THSTensor_set4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4] = *value;
    )
}

void THSTensor_set5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4][index5] = *value;
    )
}

void THSTensor_set6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4][index5][index6] = *value;
    )
}

Tensor THSTensor_slice(const Tensor tensor, int64_t dim, int64_t start, int64_t finish, int64_t step)
{
    CATCH_TENSOR(tensor->slice(dim, start, finish, step));
}

Tensor THSTensor_softplus(const Tensor tensor)
{
    CATCH_TENSOR(torch::softplus(*tensor));
}

Tensor THSTensor_sort(const Tensor tensor, const int64_t dim, const bool descending, const bool stable, Tensor* indices)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = tensor->sort(stable, dim, descending););
    *indices = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_sparse(
    Tensor indices,
    Tensor values,
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::sparse_coo_tensor(*indices, *values, at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_sigmoid(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sigmoid());
}

Tensor THSTensor_sigmoid_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sigmoid_());
}

void THSTensor_split_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t split_size,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->split(split_size, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_split_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->split_with_sizes(at::ArrayRef<int64_t>(sizes, length), dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_tensor_split_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t n,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->tensor_split(n, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_tensor_split_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->tensor_split(at::ArrayRef<int64_t>(sizes, length), dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_tensor_split_with_tensor_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const Tensor sizes,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->tensor_split(*sizes, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_vsplit_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t n)
{
    CATCH(
        auto res = tensor->vsplit(n);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_vsplit_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length)
{
    CATCH(
        auto res = tensor->vsplit(at::ArrayRef<int64_t>(sizes, length));
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_hsplit_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t n)
{
    CATCH(
        auto res = tensor->hsplit(n);
    const size_t sz = res.size();
    Tensor * result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_hsplit_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length)
{
    CATCH(
        auto res = tensor->hsplit(at::ArrayRef<int64_t>(sizes, length));
    const size_t sz = res.size();
    Tensor * result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_dsplit_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t n)
{
    CATCH(
        auto res = tensor->dsplit(n);
    const size_t sz = res.size();
    Tensor * result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_dsplit_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length)
{
    CATCH(
        auto res = tensor->dsplit(at::ArrayRef<int64_t>(sizes, length));
    const size_t sz = res.size();
    Tensor * result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_squeeze_no_dim(Tensor tensor)
{
    CATCH_TENSOR(tensor->squeeze());
}

Tensor THSTensor_squeeze(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->squeeze(dim));
}

int64_t THSTensor_stride(const Tensor tensor, const int64_t dim)
{
    CATCH_RETURN(int64_t, 0, tensor->stride(dim));
}

void THSTensor_strides(const Tensor tensor, int64_t* (*allocator)(size_t length))
{
    CATCH(
        auto res = tensor->strides();
        const size_t sz = res.size();
        int64_t* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = res[i];
    );
}

int64_t THSTensor_size(const Tensor tensor, const int64_t dim)
{
    CATCH_RETURN(int64_t, 0, tensor->size(dim));
}

void THSTensor_sizes(const Tensor tensor, int64_t* (*allocator)(size_t length))
{
    CATCH(
        auto res = tensor->sizes();
        const size_t sz = res.size();
        int64_t * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = res[i];
    );
}

Tensor THSTensor_sub_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->sub(*right));
}

Tensor THSTensor_set_requires_grad(const Tensor tensor, const bool requires_grad)
{
    CATCH_TENSOR(tensor->set_requires_grad(requires_grad));
}

Tensor THSTensor_stack(const Tensor* tensors, const int length, const int64_t dim)
{
    CATCH_TENSOR(torch::stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_hstack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::hstack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_vstack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::vstack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_dstack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::dstack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_column_stack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::column_stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_row_stack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::row_stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_std(const Tensor tensor)
{
    CATCH_TENSOR(tensor->std());
}

Tensor THSTensor_std_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim)
{
    CATCH_TENSOR(tensor->std(at::ArrayRef<int64_t>(dimensions, length), unbiased, keepdim));
}

Tensor THSTensor_sum(const Tensor tensor, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(has_type ? tensor->sum((c10::ScalarType)dtype) : tensor->sum())
}

Tensor THSTensor_sum_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(
        has_type ?
        tensor->sum(at::ArrayRef<int64_t>(dimensions, length), keepdim, (c10::ScalarType)dtype)
        :
        tensor->sum(at::ArrayRef<int64_t>(dimensions, length), keepdim))
}

Tensor THSTensor_take(const Tensor tensor, const Tensor indices)
{
    CATCH_TENSOR(tensor->take(*indices));
}

Tensor THSTensor_take_along_dim_dflt(const Tensor tensor, const Tensor indices)
{
    CATCH_TENSOR(tensor->take_along_dim(*indices));
}

Tensor THSTensor_take_along_dim(const Tensor tensor, const Tensor indices, const int64_t dim)
{
    CATCH_TENSOR(tensor->take_along_dim(*indices, dim));
}

Tensor THSTensor_tile(const Tensor tensor, const int64_t* rep, const int rep_length)
{
    CATCH_TENSOR(tensor->tile(at::ArrayRef<int64_t>(rep, rep_length)));
}


void THSTensor_topk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int k, const int64_t dim, const bool largest, const bool sorted)
{
    CATCH(
        auto topk = tensor->topk(k, dim, largest, sorted);
        Tensor * result = allocator(2);
        result[0] = new torch::Tensor(std::get<0>(topk));
        result[1] = new torch::Tensor(std::get<1>(topk));
    )
}

Tensor THSTensor_trunc(const Tensor tensor)
{
    CATCH_TENSOR(tensor->trunc());
}

Tensor THSTensor_trunc_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->trunc_());
}

int8_t THSTensor_type(const Tensor tensor)
{
    CATCH_RETURN(int8_t, 0, (int8_t)tensor->scalar_type());
}

Tensor THSTensor_to_dense(Tensor tensor)
{
    CATCH_TENSOR(tensor->to_dense());
}

Tensor THSTensor_to_device(const Tensor tensor, const int device_type, const int device_index)
{
    CATCH_RETURN_Tensor(
        auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
        res = ResultTensor(tensor->to(device)); 
    );
}

Tensor THSTensor_to_type(const Tensor tensor, int8_t scalar_type)
{
    CATCH_TENSOR(tensor->toType(at::ScalarType(scalar_type)));
}

Tensor THSTensor_to_type_and_device(const Tensor tensor, int8_t scalar_type, const int device_type, const int device_index)
{
    CATCH_RETURN_Tensor(
        auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
        res = ResultTensor(tensor->to(device, at::ScalarType(scalar_type)));
    );
}

Tensor THSTensor_triu(const Tensor tensor, const int64_t diagonal)
{
    CATCH_TENSOR(tensor->triu(diagonal));
}

Tensor THSTensor_tril(const Tensor tensor, const int64_t diagonal)
{
    CATCH_TENSOR(tensor->tril(diagonal));
}

Tensor THSTensor_transpose(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->transpose(dim1, dim2));
}

Tensor THSTensor_transpose_(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->transpose_(dim1, dim2));
}

Tensor THSTensor_view(const Tensor tensor, const int64_t* shape, const int length)
{
    CATCH_TENSOR(tensor->view(at::ArrayRef<int64_t>(shape, length)));
}

Tensor THSTensor_real(const Tensor tensor)
{
    CATCH_TENSOR(torch::real(*tensor));
}

Tensor THSTensor_imag(const Tensor tensor)
{
    CATCH_TENSOR(torch::imag(*tensor));
}

Tensor THSTensor_view_as_real(const Tensor tensor)
{
    CATCH_TENSOR(torch::view_as_real(*tensor));
}

Tensor THSTensor_view_as_complex(const Tensor tensor)
{
    CATCH_TENSOR(torch::view_as_complex(*tensor));
}

void THSTensor_unbind(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim)
{
    CATCH(
        auto res = tensor->unbind(dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_values(Tensor tensor)
{
    CATCH_TENSOR(tensor->_values());
}

Tensor THSTensor_vander(const Tensor tensor, const int64_t N, const bool increasing)
{
    CATCH_TENSOR(torch::vander(*tensor, N, increasing));
}

Tensor THSTensor_where(const Tensor condition, const Tensor x, const Tensor y)
{
    CATCH_TENSOR(x->where(*condition, *y))
}
