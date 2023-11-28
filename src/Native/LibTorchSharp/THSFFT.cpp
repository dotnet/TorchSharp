// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSTensor_fft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm)
{
    auto nArg = (n == -1 ? c10::optional<c10::SymInt>() : c10::optional<c10::SymInt>(n));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    CATCH_TENSOR(torch::fft::fft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_ifft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm)
{
    auto nArg = (n == -1 ? c10::optional<c10::SymInt>() : c10::optional<c10::SymInt>(n));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    CATCH_TENSOR(torch::fft::ifft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_fft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, 2));
    auto dArg = (dim == nullptr) ? c10::IntArrayRef({-2, -1}) : c10::IntArrayRef(dim, 2);

    CATCH_TENSOR(torch::fft::fft2(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_ifft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, 2));
    auto dArg = (dim == nullptr) ? c10::IntArrayRef({ -2, -1 }) : c10::IntArrayRef(dim, 2);

    CATCH_TENSOR(torch::fft::ifft2(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_fftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, s_length));
    auto dArg = (dim == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(dim, dim_length));

    CATCH_TENSOR(torch::fft::fftn(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_ifftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, s_length));
    auto dArg = (dim == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(dim, dim_length));

    CATCH_TENSOR(torch::fft::ifftn(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_hfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm)
{
    auto nArg = (n == -1 ? c10::optional<c10::SymInt>() : c10::optional<c10::SymInt>(n));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    CATCH_TENSOR(torch::fft::hfft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_hfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm)
{
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, 2));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto dArg = (dim == nullptr) ? c10::IntArrayRef({ -2, -1 }) : c10::IntArrayRef(dim, 2);
    CATCH_TENSOR(torch::fft::hfft2(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_hfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, s_length));
    c10::IntArrayRef dArg = (dim == nullptr) ? c10::IntArrayRef({-2, -1}) : c10::IntArrayRef(dim, dim_length);

    CATCH_TENSOR(torch::fft::hfftn(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_ihfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm)
{
    auto nArg = (n == -1 ? c10::optional<c10::SymInt>() : c10::optional<c10::SymInt>(n));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    CATCH_TENSOR(torch::fft::ihfft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_ihfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm)
{
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, 2));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto dArg = (dim == nullptr) ? c10::IntArrayRef({ -2, -1 }) : c10::IntArrayRef(dim, 2);
    CATCH_TENSOR(torch::fft::ihfft2(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_ihfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, s_length));
    c10::IntArrayRef dArg = (dim == nullptr) ? c10::IntArrayRef({ -2, -1 }) : c10::IntArrayRef(dim, dim_length);

    CATCH_TENSOR(torch::fft::ihfftn(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_rfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm)
{
    auto nArg = (n == -1 ? c10::optional<c10::SymInt>() : c10::optional<c10::SymInt>(n));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    CATCH_TENSOR(torch::fft::rfft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_irfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm)
{
    auto nArg = (n == -1 ? c10::optional<c10::SymInt>() : c10::optional<c10::SymInt>(n));
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    CATCH_TENSOR(torch::fft::irfft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_rfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, 2));
    auto dArg = (dim == nullptr) ? c10::IntArrayRef({ -2, -1 }) : c10::IntArrayRef(dim, 2);

    CATCH_TENSOR(torch::fft::rfft2(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_irfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, 2));
    auto dArg = (dim == nullptr) ? c10::IntArrayRef({ -2, -1 }) : c10::IntArrayRef(dim, 2);

    CATCH_TENSOR(torch::fft::irfft2(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_rfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, s_length));
    auto dArg = (dim == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(dim, dim_length));

    CATCH_TENSOR(torch::fft::rfftn(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_irfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm)
{
    auto normArg = (norm == 0) ? "backward" : (norm == 1) ? "forward" : "ortho";
    auto sArg = (s == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(s, s_length));
    auto dArg = (dim == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(dim, dim_length));

    CATCH_TENSOR(torch::fft::irfftn(*tensor, sArg, dArg, normArg));
}

Tensor THSTensor_fftfreq(const int64_t n, const double d, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(d == 0.0 ? torch::fft::fftfreq(n, options) : torch::fft::fftfreq(n, d, options));
}

Tensor THSTensor_rfftfreq(const int64_t n, const double d, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(d == 0.0 ? torch::fft::rfftfreq(n, options) : torch::fft::rfftfreq(n, d, options));
}

Tensor THSTensor_fftshift(const Tensor tensor, const int64_t* dim, const int dim_length)
{
    auto dArg = (dim == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(dim, dim_length));
    CATCH_TENSOR(torch::fft::fftshift(*tensor, dArg));
}

Tensor THSTensor_ifftshift(const Tensor tensor, const int64_t* dim, const int dim_length)
{
    auto dArg = (dim == nullptr) ? c10::nullopt : c10::optional<c10::IntArrayRef>(c10::IntArrayRef(dim, dim_length));
    CATCH_TENSOR(torch::fft::ifftshift(*tensor, dArg));
}

Tensor THSTensor_stft(const Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor window, bool normalized, int64_t onesided, bool return_complex)
{
    auto _hop_length = hop_length == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(hop_length);
    auto _win_length = win_length == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(win_length);
    auto _window = window == nullptr ? c10::optional<at::Tensor>() : *window;
    auto _onesided = (onesided == -1) ? c10::optional<bool>() : c10::optional<bool>((bool)onesided);
    CATCH_TENSOR(x->stft(n_fft, _hop_length, _win_length, _window, normalized, _onesided, return_complex));
}

Tensor THSTensor_istft(const Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor window, bool center, bool normalized, int64_t onesided, int64_t length, bool return_complex)
{
    auto _hop_length = hop_length == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(hop_length);
    auto _win_length = win_length == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(win_length);
    auto _window = window == nullptr ? c10::optional<at::Tensor>() : *window;
    auto _length = length == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(length);
    auto _onesided = (onesided == -1) ? c10::optional<bool>() : c10::optional<bool>((bool)onesided);
    CATCH_TENSOR(x->istft(n_fft, _hop_length, _win_length, _window, center, normalized, _onesided, _length, return_complex));
}
