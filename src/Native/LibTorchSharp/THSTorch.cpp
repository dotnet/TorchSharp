// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSTorch.h"

#include "torch/torch.h"
#include "torch/cuda.h"

void THSTorch_manual_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

Generator THSGenerator_manual_seed(const int64_t seed)
{
    auto gen = at::globalContext().defaultGenerator(at::DeviceType::CPU);
    gen.set_current_seed(seed);
    return new at::Generator(gen.getIntrusivePtr());
}

void THSCuda_manual_seed(const int64_t seed)
{
    CATCH(torch::cuda::manual_seed(seed);)
}

void THSCuda_manual_seed_all(const int64_t seed)
{
    CATCH(torch::cuda::manual_seed_all(seed);)
}

bool THSBackend_cublas_get_allow_tf32()
{
    auto result = false;
    CATCH(result = at::globalContext().allowTF32CuBLAS(););
    return result;
}

void THSBackend_cublas_set_allow_tf32(const bool flag)
{
    CATCH(at::globalContext().setAllowTF32CuBLAS(flag););
}

bool THSBackend_cudnn_get_allow_tf32()
{
    auto result = false;
    CATCH(result = at::globalContext().allowTF32CuDNN(););
    return result;
}

void THSBackend_cudnn_set_allow_tf32(const bool flag)
{
    CATCH(at::globalContext().setAllowTF32CuDNN(flag););
}

bool THSBackend_cuda_get_allow_fp16_reduced_precision_reduction()
{
    auto result = false;
    CATCH(result = at::globalContext().allowFP16ReductionCuBLAS(););
    return result;
}

void THSBackend_cuda_set_allow_fp16_reduced_precision_reduction(const bool flag)
{
    CATCH(at::globalContext().setAllowFP16ReductionCuBLAS(flag););
}

bool THSBackend_cuda_get_enable_flash_sdp()
{
    auto result = false;
    CATCH(result = at::globalContext().userEnabledFlashSDP(););
    return result;
}

void THSBackend_cuda_set_enable_flash_sdp(const bool flag)
{
    CATCH(at::globalContext().setSDPUseFlash(flag););
}

bool THSBackend_cuda_get_enable_math_sdp()
{
    auto result = false;
    CATCH(result = at::globalContext().userEnabledMathSDP(););
    return result;
}

void THSBackend_cuda_set_enable_math_sdp(const bool flag)
{
    CATCH(at::globalContext().setSDPUseMath(flag););
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

void THSTorchCuda_synchronize(const int64_t device_index)
{
    CATCH(torch::cuda::synchronize(device_index);)
}


const char * THSTorch_get_and_reset_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

int THSTorch_get_num_threads()
{
    CATCH_RETURN_RES(int, -1, res = torch::get_num_threads());
}

void THSTorch_set_num_threads(const int threads)
{
    torch::set_num_threads(threads);
}

int THSTorch_get_num_interop_threads()
{
    CATCH_RETURN_RES(int, -1, res = torch::get_num_interop_threads());
}

void THSTorch_set_num_interop_threads(const int threads)
{
    torch::set_num_interop_threads(threads);
}

int THSTorch_can_cast(const int type1, const int type2)
{
    CATCH_RETURN_RES(int, -1, res = (int)torch::can_cast((c10::ScalarType)type1, (c10::ScalarType)type2));
}

int THSTorch_promote_types(const int type1, const int type2)
{
    CATCH_RETURN_RES(int, -1, res = (int)torch::promote_types((c10::ScalarType)type1, (c10::ScalarType)type2));
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

void THSTorch_scalar_to_float16(Scalar value, unsigned short *res)
{
    *res = value->toHalf().x;
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

double THSSpecial_erf_scalar(const double x)
{
    return erf(x);
}

double THSSpecial_erfc_scalar(const double x)
{
    return erfc(x);
}

bool THSTorch_is_torch_function_mode_enabled()
{
    return at::impl::torch_function_mode_enabled(); //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L911
}

bool THSTorch_is_autocast_cache_enabled()
{
    return at::autocast::is_autocast_cache_enabled();
}

bool THSTorch_is_autocast_cpu_enabled()
{
    return at::autocast::is_cpu_enabled();  //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L523
}

bool THSTorch_is_autocast_gpu_enabled()
{
    return at::autocast::is_enabled(); //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/amp/autocast_mode.py#L363
}
bool THSTorch_is_autocast_xpu_enabled()
{
    return at::autocast::is_xpu_enabled();
}
bool THSTorch_is_autocast_hpu_enabled()
{
    return at::autocast::is_hpu_enabled();
}

#if (TORCH_VERSION_MAJOR ==2 && TORCH_VERSION_MINOR > 0)
bool THSTorch_is_autocast_ipu_enabled()
{
    return at::autocast::is_ipu_enabled();
}

bool THSTorch_is_autocast_xla_enabled()
{
    return at::autocast::is_xla_enabled();
}

#endif

int8_t THSTorch_get_autocast_cpu_dtype()
{
    return (int8_t)at::autocast::get_autocast_cpu_dtype();
}

int8_t THSTorch_get_autocast_gpu_dtype()
{
    //TODO: Implement AUTOCAST AMP AND GRADSCALER

    //INFO: Enter/Exit function of autocast_mode not need to do in C/C++ only in C# with Disposable C# Can handle all of that function (if exists)
    //https://github.com/pytorch/pytorch/blob/main/torch/amp/autocast_mode.py


    //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L629
    //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/aten/src/ATen/autocast_mode.h#L20
    return (int8_t)at::autocast::get_autocast_gpu_dtype();
}

int8_t THSTorch_get_autocast_xpu_dtype()
{
    return (int8_t)at::autocast::get_autocast_xpu_dtype();
}


int THSTorch_autocast_increment_nesting()
{
    return at::autocast::increment_nesting();
}

int THSTorch_autocast_decremental_nesting()
{
    return at::autocast::decrement_nesting();
}

void THSTorch_set_autocast_enabled(bool enabled)
{
    at::autocast::set_enabled(enabled);
}

void THSTorch_set_autocast_cache_enabled(bool enabled)
{
    at::autocast::set_autocast_cache_enabled(enabled);
}

void THSTorch_set_autocast_cpu_dtype(int8_t dtype)
{
    at::autocast::set_autocast_cpu_dtype((c10::ScalarType)dtype);
}

void THSTorch_set_autocast_gpu_dtype(int8_t dtype)
{
    at::autocast::set_autocast_gpu_dtype((c10::ScalarType)dtype);
}

void THSTorch_set_autocast_xpu_dtype(int8_t dtype)
{
    at::autocast::set_autocast_xpu_dtype((c10::ScalarType)dtype);
}

void THSTorch_clear_autocast_cache()
{
    at::autocast::clear_cache();
}

/*bool THSTorch_jit_is_scripting()
{
    
}*/