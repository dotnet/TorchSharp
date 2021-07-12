// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTorch.h"

#include "torch/torch.h"
#include "torch/cuda.h"

void THSTorch_manual_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

Generator THSGenerator_manual_seed(const int64_t seed)
{
    torch::manual_seed(seed);
    return THSGenerator_default_generator();
}

void THSCuda_manual_seed(const int64_t seed)
{
    CATCH(torch::cuda::manual_seed(seed);)
}

void THSCuda_manual_seed_all(const int64_t seed)
{
    CATCH(torch::cuda::manual_seed_all(seed);)
}


void THSGenerator_gen_manual_seed(const Generator generator, const int64_t seed)
{
    generator->set_current_seed(seed);
}

Generator THSGenerator_default_generator()
{
    auto gen = at::globalContext().defaultGenerator(at::DeviceType::CPU);
    return new at::Generator(gen.getIntrusivePtr());
}

int64_t THSGenerator_initial_seed(const Generator gen)
{
    return gen->current_seed();
}

Tensor THSGenerator_get_rng_state(const Generator gen)
{
    CATCH_TENSOR(gen->get_state());
}

void  THSGenerator_set_rng_state(const Generator gen, const Tensor tensor)
{
    gen->set_state(*tensor);
}


Generator THSGenerator_new(uint64_t seed, int64_t device, int64_t index)
{
    // TODO: Support creation of GPU RNGs. 'device' and 'index' are in the
    //       function signature in preparation thereof.
    return new at::Generator(at::detail::createCPUGenerator(seed));
}

void THSGenerator_dispose(const Generator generator)
{
    delete generator;
}

int THSTorchCuda_is_available()
{
    return torch::cuda::is_available();
}

int THSTorchCuda_cudnn_is_available()
{
    return torch::cuda::cudnn_is_available();
}

int THSTorchCuda_device_count()
{
    return (int)torch::cuda::device_count();
}

EXPORT_API(void) THSTorchCuda_synchronize(const int64_t device_index)
{
    CATCH(torch::cuda::synchronize(device_index);)
}


const char * THSTorch_get_and_reset_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

Scalar THSTorch_int8_to_scalar(int8_t value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_uint8_to_scalar(uint8_t value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int16_to_scalar(short value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int32_to_scalar(int value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int64_to_scalar(long value)
{
    return new torch::Scalar(int64_t(value));
}

Scalar THSTorch_float32_to_scalar(float value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_float64_to_scalar(double value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_float16_to_scalar(float value)
{
    return new torch::Scalar((c10::Half)value);
}

Scalar THSTorch_bfloat16_to_scalar(float value)
{
    return new torch::Scalar((c10::BFloat16)value);
}

Scalar THSTorch_bool_to_scalar(bool value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_complex32_to_scalar(float real, float imaginary)
{
    return new torch::Scalar(c10::complex<float>(real, imaginary));
}

Scalar THSTorch_complex64_to_scalar(double real, double imaginary)
{
    return new torch::Scalar(c10::complex<double>(real, imaginary));
}

int8_t THSTorch_scalar_to_int8(Scalar value)
{
    return value->toChar();
}

uint8_t THSTorch_scalar_to_uint8(Scalar value)
{
    return value->toByte();
}

int16_t THSTorch_scalar_to_int16(Scalar value)
{
    return value->toShort();
}

int32_t THSTorch_scalar_to_int32(Scalar value)
{
    return value->toInt();
}

int64_t THSTorch_scalar_to_int64(Scalar value)
{
    return value->toLong();
}

float THSTorch_scalar_to_float32(Scalar value)
{
    return value->toFloat();
}

double THSTorch_scalar_to_float64(Scalar value)
{
    return value->toDouble();
}

void THSTorch_scalar_to_complex32(Scalar value, float* (*allocator)(size_t length))
{
    auto result = value->toComplexFloat();
    auto space = allocator(2);
    space[0] = result.real();
    space[1] = result.imag();
}

void THSTorch_scalar_to_complex64(Scalar value, double* (*allocator)(size_t length))
{
    auto result = value->toComplexDouble();
    auto space = allocator(2);
    space[0] = result.real();
    space[1] = result.imag();
}

bool THSTorch_scalar_to_bool(Scalar value)
{
    return value->toBool();
}

int8_t THSTorch_scalar_type(Scalar value)
{
    return (int8_t)value->type();
}

void THSTorch_dispose_scalar(Scalar scalar)
{
    delete scalar;
}

Tensor THSTorch_lstsq(const Tensor input, const Tensor A, Tensor* qr)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::lstsq(*input, *A);)
    *qr = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}