// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
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

Tensor THSTensor_alias(const Tensor tensor)
{
    CATCH_TENSOR(*tensor);
}

Tensor THSTensor_all_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->all(dim, keepdim));
}

Tensor THSTensor_amax(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim)
{
    CATCH_TENSOR(tensor->amax(c10::IntArrayRef(dimensions, length), keepdim));
}

Tensor THSTensor_amax_out(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, const Tensor out)
{
    CATCH_TENSOR(torch::amax_out(*out, *tensor, c10::IntArrayRef(dimensions, length), keepdim));
}

Tensor THSTensor_amin(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim)
{
    CATCH_TENSOR(tensor->amin(c10::IntArrayRef(dimensions, length), keepdim));
}

Tensor THSTensor_amin_out(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, const Tensor out)
{
    CATCH_TENSOR(torch::amin_out(*out, *tensor, c10::IntArrayRef(dimensions, length), keepdim));
}

Tensor THSTensor_aminmax(const Tensor tensor, const int64_t dim, bool keepdim, Tensor* max)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = dim == -1 ? tensor->aminmax(c10::nullopt, keepdim) : tensor->aminmax(dim, keepdim);)
        * max = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
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

Tensor THSTensor_adjoint(const Tensor tensor)
{
    CATCH_TENSOR(tensor->adjoint());
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

Tensor THSTensor_argwhere(const Tensor tensor)
{
    CATCH_TENSOR(tensor->argwhere());
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

void THSTensor_broadcast_tensors(const Tensor* tensors, const int length, Tensor* (*allocator)(size_t length))
{
    CATCH(
        auto res = torch::broadcast_tensors(toTensors<at::Tensor>((torch::Tensor**)tensors, length));
    const size_t sz = res.size();
    Tensor * result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = new torch::Tensor(res[i]);
    );
}

Tensor THSTensor_broadcast_to(const Tensor tensor, const int64_t* shape, const int shape_len)
{
    CATCH_TENSOR(tensor->broadcast_to(at::ArrayRef<int64_t>(shape, shape_len)));
}

Tensor THSTensor_bucketize(const Tensor tensor, const Tensor boundaries, const bool out_int32, const bool right)
{
    CATCH_TENSOR(torch::bucketize(*tensor, *boundaries, out_int32, right));
}

Tensor THSTensor_channel_shuffle(const Tensor tensor, const int64_t groups)
{
    CATCH_TENSOR(torch::channel_shuffle(*tensor, groups));
}

Tensor THSTensor_parameters_to_vector(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::nn::utils::parameters_to_vector(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

void THSTensor_vector_to_parameters(const Tensor vec, const Tensor* tensors, const int length)
{
    CATCH(torch::nn::utils::vector_to_parameters(*vec, toTensors<at::Tensor>((torch::Tensor**)tensors, length)););
}

Tensor THSTensor_cartesian_prod(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::cartesian_prod(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

double THSTensor_clip_grad_norm_(const Tensor* tensors, const int length, const double max_norm, const double norm_type)
{
    double res = 0.0;
    CATCH(
        res = torch::nn::utils::clip_grad_norm_(toTensors<at::Tensor>((torch::Tensor**)tensors, length), max_norm, norm_type);
    );
    return res;
}

void THSTensor_clip_grad_value_(const Tensor* tensors, const int length, const double value)
{
    std::vector<at::Tensor> vec;
    CATCH(
        for (auto i = 0; i < length; i++) {
            vec.push_back(*tensors[i]);
        }
    torch::nn::utils::clip_grad_value_(vec, value);
    );
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
    auto mn = min == nullptr ? c10::optional<c10::Scalar>() : *min;
    auto mx = max == nullptr ? c10::optional<c10::Scalar>() : *max;
    CATCH_TENSOR(tensor->clamp(mn, mx));
}

Tensor THSTensor_clamp_(const Tensor tensor, const Scalar min, const Scalar max)
{
    auto mn = min == nullptr ? c10::optional<c10::Scalar>() : *min;
    auto mx = max == nullptr ? c10::optional<c10::Scalar>() : *max;
    CATCH_TENSOR(tensor->clamp_(mn, mx));
}

Tensor THSTensor_clamp_tensor(const Tensor tensor, const Tensor min, const Tensor max)
{
    auto mn = min == nullptr ? c10::optional<at::Tensor>() : *min;
    auto mx = max == nullptr ? c10::optional<at::Tensor>() : *max;
    CATCH_TENSOR(tensor->clamp(mn, mx));
}

Tensor THSTensor_clamp_tensor_(const Tensor tensor, const Tensor min, const Tensor max)
{
    auto mn = min == nullptr ? c10::optional<at::Tensor>() : *min;
    auto mx = max == nullptr ? c10::optional<at::Tensor>() : *max;
    CATCH_TENSOR(tensor->clamp_(mn, mx));
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

Tensor THSTensor_combinations(const Tensor tensor, const int r, const bool with_replacement)
{
    CATCH_TENSOR(torch::combinations(*tensor, r, with_replacement));
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

int THSTensor_is_contiguous(const Tensor tensor)
{
    bool result = false;
    CATCH(result = tensor->is_contiguous(););
    return result;
}

int64_t THSTensor_is_nonzero(const Tensor tensor)
{
    bool result = false;
    CATCH(result = tensor->is_nonzero();)
    return result;
}

Tensor THSTensor_copysign(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(input->copysign(*other));
}

Tensor THSTensor_corrcoef(const Tensor tensor)
{
    CATCH_TENSOR(tensor->corrcoef());
}

bool THSTensor_is_cpu(const Tensor tensor)
{
    bool result = true;
    CATCH(result = tensor->is_cpu(););
    return result;
}

Tensor THSTensor_cpu(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cpu());
}

Tensor THSTensor_cuda(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cuda());
}

Tensor THSTensor_pin_memory(const Tensor tensor)
{
    CATCH_TENSOR(tensor->pin_memory());
}

int64_t THSTensor_is_pinned(const Tensor tensor)
{
    bool result = false;
    CATCH(result = tensor->is_pinned(););
    return result;
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

Tensor THSTensor_detach(const Tensor tensor)
{
    CATCH_TENSOR(tensor->detach());
}

Tensor THSTensor_detach_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->detach_());
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

Tensor THSTensor_diag_embed(const Tensor tensor, const int64_t offset, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->diag_embed(offset, dim1, dim2));
}

Tensor THSTensor_diff(const Tensor tensor, const int64_t n, const int64_t dim, const Tensor prepend, const Tensor append)
{
    c10::optional<at::Tensor> prep = prepend != nullptr ? *prepend : c10::optional<at::Tensor>(c10::nullopt);
    c10::optional<at::Tensor> app = append != nullptr ? *append : c10::optional<at::Tensor>(c10::nullopt);
    CATCH_TENSOR(tensor->diff(n, dim, prep, app));
}

void THSTensor_free(const Tensor tensor)
{
    // If we can figure out how to decref the native tensor, the logic should go here.
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

Tensor THSTensor_repeat_interleave(const Tensor tensor, const Tensor repeats, const int64_t dim, const int64_t output_size)
{
    auto _dim = dim == INT64_MIN ? c10::optional<int64_t>() : c10::optional<int64_t>(dim);
    auto _output_size = output_size == INT64_MIN ? c10::optional<int64_t>() : c10::optional<int64_t>(output_size);
    CATCH_TENSOR(tensor->repeat_interleave(*repeats, _dim, _output_size));
}

Tensor THSTensor_repeat_interleave_int64(const Tensor tensor, const int64_t repeats, const int64_t dim, const int64_t output_size)
{
    auto _dim = dim == INT64_MIN ? c10::optional<int64_t>() : c10::optional<int64_t>(dim);
    auto _output_size = output_size == INT64_MIN ? c10::optional<int64_t>() : c10::optional<int64_t>(output_size);
    CATCH_TENSOR(tensor->repeat_interleave(repeats, _dim, _output_size));
}

int THSTensor_result_type(const Tensor left, const Tensor right)
{
    CATCH_RETURN_RES(int, -1, res = (int)torch::result_type(*left, *right));
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

Tensor THSTensor_isin(const Tensor elements, const Tensor test_elements, bool assume_unique, bool invert)
{
    CATCH_TENSOR(torch::isin(*elements, *test_elements, assume_unique, invert));
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

// Wrapper for <code>Tensor isnan(const Tensor&amp; self)</code>
// See also: <a href="https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorCompare.cpp"/>TensorCompare.cpp</a>
Tensor THSTensor_isnan(const Tensor tensor)
{
    CATCH_TENSOR(torch::isnan(*tensor));
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

Tensor THSTensor_select(Tensor tensor, int64_t dim, int64_t index)
{
    CATCH_TENSOR(tensor->select(dim, index));
}

Tensor THSTensor_index_add(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source, const Scalar alpha)
{
    CATCH_TENSOR(tensor->index_add(dim, *index, *source, *alpha));
}

Tensor THSTensor_index_add_(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source, const Scalar alpha)
{
    CATCH_TENSOR(tensor->index_add_(dim, *index, *source, *alpha));
}

Tensor THSTensor_index_copy(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source)
{
    CATCH_TENSOR(tensor->index_copy(dim, *index, *source));
}

Tensor THSTensor_index_copy_(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source)
{
    CATCH_TENSOR(tensor->index_copy_(dim, *index, *source));
}

Tensor THSTensor_index_fill(const Tensor tensor, const int64_t dim, const Tensor index, const Scalar value)
{
    CATCH_TENSOR(tensor->index_fill(dim, *index, *value));
}

Tensor THSTensor_index_fill_(const Tensor tensor, const int64_t dim, const Tensor index, const Scalar value)
{
    CATCH_TENSOR(tensor->index_fill_(dim, *index, *value));
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

Tensor THSTensor_var(const Tensor tensor)
{
    CATCH_TENSOR(tensor->var());
}

Tensor THSTensor_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(
        has_type
        ? tensor->mean(at::ArrayRef<int64_t>(dimensions, length), keepdim, (c10::ScalarType)dtype)
        : tensor->mean(at::ArrayRef<int64_t>(dimensions, length), keepdim))
}

Tensor THSTensor_var_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim)
{
    tensor->var();
    CATCH_TENSOR(tensor->var(at::ArrayRef<int64_t>(dimensions, length), unbiased, keepdim))
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

Tensor THSTensor_min_elementwise(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->minimum(*other));
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

Tensor THSTensor_nanmean(const Tensor input, const int64_t* dims, const int dims_len, bool keepdim, int8_t scalar_type)
{
    CATCH_TENSOR(torch::nanmean(*input, at::ArrayRef<int64_t>(dims, dims_len), keepdim, at::ScalarType(scalar_type)));
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

Tensor THSTensor_ormqr(const Tensor input, const Tensor tau, const Tensor other, bool left, bool transpose)
{
    CATCH_TENSOR(torch::ormqr(*input, *tau, *other, left, transpose));
}

Tensor THSTensor_mH(const Tensor tensor)
{
    CATCH_TENSOR(tensor->mH());
}

Tensor THSTensor_mT(const Tensor tensor)
{
    CATCH_TENSOR(tensor->mT());
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

void THSTensor_retain_grad(const Tensor tensor)
{
    CATCH(tensor->retain_grad(););
}

int64_t THSTensor_is_leaf(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->is_leaf(););
}

Tensor THSTensor_reshape(const Tensor tensor, const int64_t* shape, const int length)
{
    CATCH_TENSOR(tensor->reshape(at::ArrayRef<int64_t>(shape, length)));
}

Tensor THSTensor_rot90(const Tensor tensor, const int64_t k, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->rot90(k, { dim1, dim2 }));
}

Tensor THSTensor_roll(const Tensor tensor, const int64_t* shifts, const int shLength, const int64_t* dims, const int dimLength)
{
    CATCH_TENSOR(
        dims != nullptr
        ? tensor->roll(at::ArrayRef<int64_t>(shifts, shLength), at::ArrayRef<int64_t>(dims, dimLength))
        : tensor->roll(at::ArrayRef<int64_t>(shifts, shLength)));
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

Tensor THSTensor_scatter_(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index,
    const Tensor source)
{
    CATCH_TENSOR(tensor->scatter_(dim, *index, *source));
}

Tensor THSTensor_select_scatter(
    const Tensor tensor,
    const Tensor source,
    const int64_t dim,
    const int64_t index)
{
    CATCH_TENSOR(torch::select_scatter(*tensor, *source, dim, index));
}

Tensor THSTensor_diagonal_scatter(
    const Tensor tensor,
    const Tensor source,
    const int64_t offset,
    const int64_t dim1,
    const int64_t dim2)
{
    CATCH_TENSOR(torch::diagonal_scatter(*tensor, *source, offset, dim1, dim2));
}

Tensor THSTensor_slice_scatter(
    const Tensor tensor,
    const Tensor source,
    const int64_t dim,
    const int64_t *start,
    const int64_t *end,
    const int64_t step)
{
    CATCH_TENSOR(torch::slice_scatter(*tensor, *source, dim, start == nullptr ? c10::optional<int64_t>() : c10::optional<int64_t>(*start), end == nullptr ? c10::optional<int64_t>() : c10::optional<int64_t>(*end), step));
}

Tensor THSTensor_scatter_add(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index,
    const Tensor source)
{
    CATCH_TENSOR(torch::scatter_add(*tensor, dim, *index, *source));
}

Tensor THSTensor_scatter_add_(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index,
    const Tensor source)
{
    CATCH_TENSOR(tensor->scatter_add_(dim, *index, *source));
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

void THSTensor_set1(const Tensor tensor, int64_t index, const Tensor value)
{
    CATCH(
        (*tensor)[index] = *value;
    )
}

void THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, const Tensor value)
{
    CATCH(
        (*tensor)[index1][index2] = *value;
    )
}

void THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, const Tensor value)
{
    CATCH(
        (*tensor)[index1][index2][index3] = *value;
    )
}

void THSTensor_set4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, const Tensor value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4] = *value;
    )
}

void THSTensor_set5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, const Tensor value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4][index5] = *value;
    )
}

void THSTensor_set6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6, const Tensor value)
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

Tensor THSTensor_cumulative_trapezoid_x(const Tensor y, const Tensor x, int64_t dim)
{
    CATCH_TENSOR(torch::cumulative_trapezoid(*y, *x, dim));
}

Tensor THSTensor_cumulative_trapezoid_dx(const Tensor y, const double dx, int64_t dim)
{
    CATCH_TENSOR(torch::cumulative_trapezoid(*y, dx, dim));
}

Tensor THSTensor_trapezoid_x(const Tensor y, const Tensor x, int64_t dim)
{
    CATCH_TENSOR(torch::trapezoid(*y, *x, dim));
}

Tensor THSTensor_trapezoid_dx(const Tensor y, const double dx, int64_t dim)
{
    CATCH_TENSOR(torch::trapezoid(*y, dx, dim));
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

Tensor THSTensor_squeeze_no_dim_(Tensor tensor)
{
    CATCH_TENSOR(tensor->squeeze_());
}

Tensor THSTensor_squeeze_(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->squeeze_(dim));
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
    int64_t * result = allocator(sz);
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

void THSTensor_meshgrid(const Tensor* tensors, const int length, const char* indexing, Tensor* (*allocator)(size_t length))
{
    std::string str = indexing;
    CATCH(
        auto res = torch::meshgrid(toTensors<at::Tensor>((torch::Tensor**)tensors, length), indexing);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_std(const Tensor tensor)
{
    CATCH_TENSOR(tensor->std());
}

Tensor THSTensor_std_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim)
{
    CATCH_TENSOR(tensor->std(at::ArrayRef<int64_t>(dimensions, length), unbiased, keepdim));
}

Tensor THSTensor_std_mean(const Tensor tensor, bool unbiased, Tensor* mean)
{
    std::tuple<at::Tensor, at::Tensor> res;

    CATCH(res = torch::std_mean(*tensor, unbiased););
    *mean = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

// <summary>
// Wrapper for <code>std::tuple&lt;Tensor, Tensor&gt; var_mean(const Tensor&amp; self, bool unbiased)</code>.
// See also: <a href="https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp>ReduceOps.cpp</a>
// </summary>
Tensor THSTensor_var_mean(const Tensor tensor, bool unbiased, Tensor* mean)
{
    std::tuple<at::Tensor, at::Tensor> res;

    CATCH(res = torch::var_mean(*tensor, unbiased););
    *mean = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_std_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim, Tensor* mean)
{
    std::tuple<at::Tensor, at::Tensor> res;

    CATCH(res = torch::std_mean(*tensor, at::ArrayRef<int64_t>(dimensions, length), unbiased, keepdim););
    *mean = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

// <summary>
// Wrapper for <code>std::tuple&lt;Tensor, Tensor&gt; var_mean(const Tensor&amp; self, IntArrayRef dim, bool unbiased, bool keepdim)</code>.
// See also: <a href="https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp>ReduceOps.cpp</a>
// </summary>
Tensor THSTensor_var_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim, Tensor* mean)
{
    std::tuple<at::Tensor, at::Tensor> res;

    CATCH(res = torch::var_mean(*tensor, at::ArrayRef<int64_t>(dimensions, length), unbiased, keepdim););
    *mean = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
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

Tensor THSTensor_set_(Tensor tensor, const Tensor source)
{
    CATCH_TENSOR(tensor->set_(*source));
}

Tensor THSTensor_to_device(const Tensor tensor, const int device_type, const int device_index, const bool copy)
{
    CATCH_RETURN_Tensor(
        auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
    res = ResultTensor(tensor->to(device, false, copy));
    );
}

Tensor THSTensor_to_type(const Tensor tensor, int8_t scalar_type, const bool copy)
{
    CATCH_TENSOR(tensor->to(at::ScalarType(scalar_type), false, copy));
}

Tensor THSTensor_to_type_and_device(const Tensor tensor, int8_t scalar_type, const int device_type, const int device_index, const bool copy)
{
    CATCH_RETURN_Tensor(
        auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
    res = ResultTensor(tensor->to(device, at::ScalarType(scalar_type), false, copy));
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

Tensor THSTensor_tril_indices(const int64_t row, const int64_t col, const int64_t offset, const int8_t scalar_type, const int device_type, const int device_index)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index));
    CATCH_TENSOR(torch::tril_indices(row, col, offset, options));
}

Tensor THSTensor_triu_indices(const int64_t row, const int64_t col, const int64_t offset, const int8_t scalar_type, const int device_type, const int device_index)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index));
    CATCH_TENSOR(torch::triu_indices(row, col, offset, options));
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

Tensor THSTensor_flatten(const Tensor tensor, const int64_t start, const int64_t end)
{
    CATCH_TENSOR(tensor->flatten(start, end));
}

Tensor THSTensor_unflatten(const Tensor tensor, const int64_t dimension, const int64_t* shape, const int length)
{
    CATCH_TENSOR(tensor->unflatten(dimension, at::ArrayRef<int64_t>(shape, length)));
}

Tensor THSTensor_unique(const Tensor tensor, const bool sorted, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;

    CATCH(res = torch::_unique2(*tensor, sorted, return_inverse, return_counts););

    if (return_inverse) {
        *inverse_indices = ResultTensor(std::get<1>(res));
    }
    if (return_counts) {
        *counts = ResultTensor(std::get<2>(res));
    }
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_unique_dim(const Tensor tensor, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;

    CATCH(res = torch::unique_dim(*tensor, dim, sorted, return_inverse, return_counts););

    if (return_inverse) {
        *inverse_indices = ResultTensor(std::get<1>(res));
    }
    if (return_counts) {
        *counts = ResultTensor(std::get<2>(res));
    }
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_unique_consecutive(const Tensor tensor, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;

    CATCH(res = torch::unique_consecutive(*tensor, return_inverse, return_counts););

    if (return_inverse) {
        *inverse_indices = ResultTensor(std::get<1>(res));
    }
    if (return_counts) {
        *counts = ResultTensor(std::get<2>(res));
    }
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_unique_dim_consecutive(const Tensor tensor, const int64_t dim, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts)
{
    std::tuple<at::Tensor, at::Tensor, at::Tensor> res;

    CATCH(res = torch::unique_dim_consecutive(*tensor, dim, return_inverse, return_counts););

    if (return_inverse) {
        *inverse_indices = ResultTensor(std::get<1>(res));
    }
    if (return_counts) {
        *counts = ResultTensor(std::get<2>(res));
    }
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_unfold(const Tensor tensor, const int64_t dimension, const int64_t size, const int64_t step)
{
    CATCH_TENSOR(tensor->unfold(dimension, size, step));
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

void THSTensor_where_list(
    const Tensor condition,
    Tensor* (*allocator)(size_t length))
{
    CATCH(
        auto res = at::_ops::where::call(*condition);
    const size_t sz = res.size();
    Tensor * result = allocator(sz);
    for (size_t i = 0; i < sz; i++)
        result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_searchsorted_t(const Tensor sorted_sequence, const Tensor values, const bool out_int32, const bool right, const Tensor sorter)
{    
    CATCH_TENSOR(
        sorter == nullptr
        ? torch::searchsorted(*sorted_sequence, *values, out_int32, right)
        : torch::searchsorted(*sorted_sequence, *values, out_int32, right, c10::nullopt, *sorter));
}

Tensor THSTensor_searchsorted_s(const Tensor sorted_sequence, const Scalar values, const bool out_int32, const bool right, const Tensor sorter)
{
    CATCH_TENSOR(
        sorter == nullptr
        ? torch::searchsorted(*sorted_sequence, *values, out_int32, right)
        : torch::searchsorted(*sorted_sequence, *values, out_int32, right, c10::nullopt, *sorter));
}

Tensor THSTensor_histogram_t(const Tensor input, const Tensor bins, const Tensor weight, const bool density, Tensor* r_bin_edges)
{
    std::tuple<at::Tensor, at::Tensor> res;

    c10::optional<at::Tensor> weight_ = weight == nullptr ? c10::optional<at::Tensor>(c10::nullopt) : *weight;

    CATCH(res = torch::histogram(*input, *bins, weight_, density););
    *r_bin_edges = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_histogram_i(const Tensor input, const int64_t bins, const double* range, const int length, const Tensor weight, const bool density, Tensor* r_bin_edges)
{
    std::tuple<at::Tensor, at::Tensor> res;

    auto range_ = range == nullptr ? c10::optional<at::ArrayRef<double>>() : c10::optional<at::ArrayRef<double>>(at::ArrayRef<double>(range, length));
    c10::optional<at::Tensor> weight_ = weight == nullptr ? c10::optional<at::Tensor>(c10::nullopt) : *weight;

    CATCH(res = torch::histogram(*input, bins, range_, weight_, density););
    *r_bin_edges = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_histogram_out_t(const Tensor input, const Tensor bins, const Tensor weight, const bool density, Tensor* hist, Tensor* bin_edges, Tensor* r_bin_edges)
{
    std::tuple<at::Tensor, at::Tensor> res;

    c10::optional<at::Tensor> weight_ = weight == nullptr ? c10::optional<at::Tensor>(c10::nullopt) : *weight;

    CATCH(res = torch::histogram_outf(*input, *bins, weight_, density, **hist, **bin_edges););
    *r_bin_edges = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSTensor_histogram_out_i(const Tensor input, const int64_t bins, const double* range, const int length, const Tensor weight, const bool density, Tensor* hist, Tensor* bin_edges, Tensor* r_bin_edges)
{
    std::tuple<at::Tensor, at::Tensor> res;

    auto range_ = range == nullptr ? c10::optional<at::ArrayRef<double>>() : c10::optional<at::ArrayRef<double>>(at::ArrayRef<double>(range, length));
    c10::optional<at::Tensor> weight_ = weight == nullptr ? c10::optional<at::Tensor>(c10::nullopt) : *weight;

    CATCH(res = torch::histogram_outf(*input, bins, range_, weight_, density, **hist, **bin_edges););
    *r_bin_edges = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

bool THSTensor_has_names(Tensor tensor)
{
    CATCH(
        return tensor->has_names();
    );
    return false;
}

void THSTensor_names(Tensor tensor, const char** (*allocator)(size_t length))
{
    if (!tensor->has_names()) return;

    auto names = tensor->names();

    const char** result2 = allocator(names.size());

    for (size_t i = 0; i < names.size(); i++)
    {
        result2[i] = make_sharable_string(names[i].symbol().toUnqualString());
    }
}

Tensor THSTensor_rename(Tensor tensor, const char** names, int64_t nLength)
{
    CATCH(
        if (names != nullptr && nLength > 0) {
            std::vector<at::Dimname> nvec;
            for (int i = 0; i < nLength; ++i) {
                if (names[i] != nullptr && strcmp(names[i], "*") != 0) {
                    nvec.push_back(at::Dimname::fromSymbol(at::Symbol::dimname(names[i])));
                }
                else {
                    nvec.push_back(at::Dimname::wildcard());
                }
            }

            return ResultTensor(tensor->rename(nvec));
        }
        else {
            return ResultTensor(tensor->rename(c10::nullopt));
        }
    );

    return nullptr;
}

Tensor THSTensor_rename_(Tensor tensor, const char** names, int64_t nLength)
{
    CATCH(
        if (names != nullptr && nLength > 0) {
            std::vector<at::Dimname> nvec;
            for (int i = 0; i < nLength; ++i) {
                if (names[i] != nullptr && strcmp(names[i], "*") != 0) {
                    nvec.push_back(at::Dimname::fromSymbol(at::Symbol::dimname(names[i])));
                }
                else {
                    nvec.push_back(at::Dimname::wildcard());
                }
            }

            return ResultTensor(tensor->rename_(nvec));
        }
        else {
            return ResultTensor(tensor->rename_(c10::nullopt));
        }
    );

    return nullptr;
}

Tensor THSTensor_refine_names(Tensor tensor, const char** names, int64_t nLength)
{
    CATCH(
        std::vector<at::Dimname> nvec;
        for (int i = 0; i < nLength; ++i) {
            if (names[i] != nullptr && strcmp(names[i], "*") != 0) {
                nvec.push_back(at::Dimname::fromSymbol(at::Symbol::dimname(names[i])));
            }
            else {
                nvec.push_back(at::Dimname::wildcard());
            }
        }

        return ResultTensor(tensor->refine_names(nvec));
    );

    return nullptr;
}

Tensor THSTensor_align_to(Tensor tensor, const char** names, int64_t nLength)
{
    CATCH(
        std::vector<at::Dimname> nvec;
        int64_t ellipsis = -1;
        for (int i = 0; i < nLength; ++i) {
            if (strcmp(names[i], "...") != 0) {
                nvec.push_back(at::Dimname::fromSymbol(at::Symbol::dimname(names[i])));
            }
            else {
                ellipsis = i;
            }
        }

        return (ellipsis > -1) ? ResultTensor(tensor->align_to(nvec, ellipsis)) : ResultTensor(tensor->align_to(nvec));
    );

    return nullptr;
}

Tensor THSTensor_flatten_names(Tensor tensor, const char** names, int64_t nLength)
{
    CATCH(
        std::vector<at::Dimname> nvec;
        for (int i = 0; i < nLength - 1; ++i)
            nvec.push_back(at::Dimname::fromSymbol(at::Symbol::dimname(names[i])));

        at::Dimname out_dim = at::Dimname::fromSymbol(at::Symbol::dimname(names[nLength - 1]));

        return ResultTensor(tensor->flatten(nvec, out_dim));
    );

    return nullptr;
}

Tensor THSTensor_unflatten_names(Tensor tensor, const char** names, const int64_t* sizes, int64_t nLength)
{
    CATCH(
        std::vector<at::Dimname> nvec;

        at::Dimname dim = at::Dimname::fromSymbol(at::Symbol::dimname(names[0]));

        for (int i = 1; i < nLength; ++i)
            nvec.push_back(at::Dimname::fromSymbol(at::Symbol::dimname(names[i])));

        return ResultTensor(tensor->unflatten(dim, c10::IntArrayRef(sizes, nLength - 1), nvec));
    );

    return nullptr;
}
