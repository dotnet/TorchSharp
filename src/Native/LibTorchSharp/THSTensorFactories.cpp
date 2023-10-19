// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSTensor_arange(const Scalar start, const Scalar end, const Scalar step, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::arange(*start, *end, *step, options));
}

Tensor THSTensor_arange_out(const Scalar start, const Scalar end, const Scalar step, const Tensor out)
{
    CATCH_TENSOR(torch::arange_out(*out, *start, *end, *step));
}


Tensor THSTensor_empty(
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

    CATCH_TENSOR(torch::empty(at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_empty_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::empty_like(*input, options));
}

Tensor THSTensor_empty_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::empty_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_empty_strided(
    const int64_t* sizes,
    const int sz_length,
    const int64_t* strides,
    const int str_length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::empty_strided(at::ArrayRef<int64_t>(sizes, sz_length), at::ArrayRef<int64_t>(strides, str_length), options));
}

Tensor THSTensor_as_strided(
    const Tensor input,
    const int64_t* sizes,
    const int sz_length,
    const int64_t* strides,
    const int str_length,
    const int64_t storage_offset)
{
    CATCH_TENSOR(input->as_strided(at::ArrayRef<int64_t>(sizes, sz_length), at::ArrayRef<int64_t>(strides, str_length), storage_offset));
}

Tensor THSTensor_eye(const int64_t n, const int64_t m, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::eye(n, m, options));
}

Tensor THSTensor_eye_out(const int64_t n, const int64_t m, const Tensor out)
{
    CATCH_TENSOR(torch::eye_out(*out, n, m));
}

Tensor THSTensor_full(
    const int64_t* sizes,
    const int length,
    const Scalar value,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::full(at::ArrayRef<int64_t>(sizes, length), *value, options));
}

Tensor THSTensor_full_like(
    const Tensor input,
    const Scalar value,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::full_like(*input, *value, options));
}

Tensor THSTensor_full_out(const int64_t* sizes, const int length, const Scalar value, const Tensor out)
{
    CATCH_TENSOR(torch::full_out(*out, at::ArrayRef<int64_t>(sizes, length), *value));
}

Tensor THSTensor_linspace(const double start, const double end, const int64_t steps, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::linspace(start, end, steps, options));
}

Tensor THSTensor_logspace(const double start, const double end, const int64_t steps, double base, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::logspace(start, end, steps, base, options));
}


Tensor THSTensor_from_file(
    const char* filename,
    const int8_t shared,
    const int64_t size,
    const int8_t scalar_type,
    const int device_type,
    const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    c10::optional<bool> sh = shared == -1 ? c10::optional<bool>() : (shared == 1);
    c10::optional<int64_t> sz = size == -1 ? c10::optional<int64_t>() : size;
    CATCH_TENSOR(torch::from_file(filename, sh, sz, options));
}

Tensor THSTensor_new(
    void* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    int8_t scalar_type,         // The element type in the data array
    int8_t dtype,               // The element type of the constructed tensor
    const int device_type,
    const int device_index,
    const bool requires_grad)
{
    bool move = device_type != 0;
    bool convert = scalar_type != dtype;

    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .requires_grad(requires_grad && !move);

    CATCH(
        auto res = torch::from_blob(data, at::ArrayRef<int64_t>(sizes, szlength), deleter, options);
        if (move) // Not CPU
        {
            auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
            res = convert
                ? res.to(device, at::ScalarType(dtype), false, false).set_requires_grad(requires_grad)
                : res.to(device, false, false).set_requires_grad(requires_grad);
        }
        else if (convert)
        {
            res = res.to(at::ScalarType(dtype), false, false);
        }
        return ResultTensor(res);
    );
    return nullptr;
}

Tensor THSTensor_frombuffer(
    void* data,
    void (*deleter)(void*),
    const int64_t count,
    const ptrdiff_t offset,
    int8_t scalar_type,         // The element type in the data array
    int8_t dtype,               // The element type of the constructed tensor
    const int device_type,
    const int device_index,
    const bool requires_grad)
{
    bool move = device_type != 0;
    bool convert = scalar_type != dtype;

    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .requires_grad(requires_grad && !move);

    auto dataPtr = reinterpret_cast<char*>(data);
    CATCH(
        auto res = torch::from_blob(dataPtr + offset, at::ArrayRef<int64_t>(count), deleter, options);
        if (move) // Not CPU
        {
            auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
            res = convert
                ? res.to(device, at::ScalarType(dtype), false, false).set_requires_grad(requires_grad)
                : res.to(device, false, false).set_requires_grad(requires_grad);
        }
        else if (convert)
        {
            res = res.to(at::ScalarType(dtype), false, false);
        }
        return ResultTensor(res);
    );
    return nullptr;
}

Tensor THSTensor_newInt64(
    int64_t* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const int device_type,
    const int device_index,
    const bool requires_grad)
{
    bool move = device_type != 0;

    auto options = at::TensorOptions()
        .dtype(at::ScalarType(at::kLong))
        .requires_grad(requires_grad && !move);

    auto dataPtr = reinterpret_cast<char*>(data);
    CATCH(
        auto res = torch::from_blob(data, at::ArrayRef<int64_t>(sizes, szlength), deleter, options);
        if (move) // Not CPU
        {
            auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
            res = res.to(device, false, false).set_requires_grad(requires_grad);
        }
        return ResultTensor(res);
    );
    return nullptr;
}

// The data is passed in as float and copied into the array of Half in the C++ code
// since .NET doesn't know about 'float16' values.
Tensor THSTensor_newFloat16(
    float* rawArray,
    c10::Half* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const int device_type,
    const int device_index,
    const bool requires_grad)
{
    bool move = device_type != 0;
    CATCH_RETURN_Tensor(
        int64_t sz = 1;
        for (int k = 0; k < szlength; k++)
            sz *= sizes[k];
        for (int64_t i = 0; i < sz; i++)
            dataArray[i] = (c10::Half)rawArray[i];

        auto options = at::TensorOptions()
            .dtype(at::ScalarType(at::kLong))
            .requires_grad(requires_grad && !move);
        auto r = torch::from_blob(dataArray, at::ArrayRef<int64_t>(sizes, szlength), deleter, options);
        if (move) // Not CPU
        {
            auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
            r = r.to(device, false, false).set_requires_grad(requires_grad);
        }
        res = ResultTensor(r);
    )
}

// The data is passed in as float and copied into the array of Half in the C++ code
// since .NET doesn't know about 'float16' values.
Tensor THSTensor_newBFloat16(
    float* rawArray,
    c10::BFloat16* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const int device_type,
    const int device_index,
    const bool requires_grad)
{
    bool move = device_type != 0;
    CATCH_RETURN_Tensor(
        int64_t sz = 1;
        for (int k = 0; k < szlength; k++)
            sz *= sizes[k];
        for (int64_t i = 0; i < sz; i++)
            dataArray[i] = (c10::BFloat16)rawArray[i];

        auto options = at::TensorOptions()
            .dtype(at::ScalarType(at::kLong))
            .requires_grad(requires_grad && !move);
        auto r = torch::from_blob(dataArray, at::ArrayRef<int64_t>(sizes, szlength), deleter, options);
        if (move) // Not CPU
        {
            auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
            r = r.to(device, false, false).set_requires_grad(requires_grad);
        }
        res = ResultTensor(r);
        )
}

Tensor THSTensor_newInt8Scalar(int8_t data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Char))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newByteScalar(char data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Byte))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newBoolScalar(bool  data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Bool))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newInt16Scalar(short data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Short))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newInt32Scalar(int data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Int))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newInt64Scalar(int64_t data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Long))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newFloat64Scalar(double data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Double))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newFloat32Scalar(float data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Float))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newComplexFloat32Scalar(float real, float imaginary, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::ComplexFloat))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    c10::complex<float> data(real, imaginary);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newComplexFloat64Scalar(double real, double imaginary, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::ComplexDouble))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    c10::complex<double> data(real, imaginary);
    CATCH_TENSOR(torch::tensor(data, options));
}


Tensor THSTensor_newFloat16Scalar(float data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Half))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor((c10::Half)data, options));
}

Tensor THSTensor_newBFloat16Scalar(float data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::BFloat16))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor((c10::BFloat16)data, options));
}

Tensor THSTensor_ones(
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

    CATCH_TENSOR(torch::ones(at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_ones_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::ones_like(*input, options));
}


Tensor THSTensor_ones_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::ones_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_zeros(
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

    CATCH_TENSOR(torch::zeros(at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_zeros_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::zeros_like(*input, options));
}

Tensor THSTensor_zeros_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::zeros_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

// Spectral windows construction


Tensor THSTensor_bartlett_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::bartlett_window(len, periodic, options));
}

Tensor THSTensor_blackman_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::blackman_window(len, periodic, options));
}

Tensor THSTensor_hamming_window(const int64_t len, bool periodic, double alpha, double beta, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::hamming_window(len, periodic, alpha, beta, options));
}

Tensor THSTensor_hann_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::hann_window(len, periodic, options));
}

Tensor THSTensor_kaiser_window(const int64_t len, bool periodic, double beta, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::kaiser_window(len, periodic, beta, options));
}

double THSInit_calculate_gain(int64_t nonlinearity, double param)
{
    CATCH_RETURN(double, 0.0, torch::nn::init::calculate_gain(get_nl_type(nonlinearity), param))
}

torch::nn::init::FanModeType get_fan_mode(int64_t mode)
{
    return mode == 0 ? torch::nn::init::FanModeType(torch::kFanIn) : torch::nn::init::FanModeType(torch::kFanOut);
}

void THSInit_constant_(Tensor tensor, Scalar value)
{
    CATCH(torch::nn::init::constant_(*tensor, *value);)
}

void THSInit_dirac_(Tensor tensor)
{
    CATCH(torch::nn::init::dirac_(*tensor);)
}

void THSInit_eye_(Tensor tensor)
{
    CATCH(torch::nn::init::eye_(*tensor);)
}

void THSInit_ones_(Tensor tensor)
{
    CATCH(torch::nn::init::ones_(*tensor);)
}

void THSInit_orthogonal_(Tensor tensor, double gain)
{
    CATCH(torch::nn::init::orthogonal_(*tensor, gain);)
}

void THSInit_sparse_(Tensor tensor, double sparsity, double std)
{
    CATCH(torch::nn::init::sparse_(*tensor, sparsity, std);)
}

void THSInit_zeros_(Tensor tensor)
{
    CATCH(torch::nn::init::zeros_(*tensor);)
}