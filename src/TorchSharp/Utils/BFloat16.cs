using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace System
{
    [StructLayout(LayoutKind.Sequential,Pack=2)]
    public struct BFloat16
    {
        [MarshalAs(UnmanagedType.I2)]
        private short x;
        public struct from_bits_t{};
    }

    /*
     * 
struct alignas(2) BFloat16 {
  uint16_t x;

  // HIP wants __host__ __device__ tag, CUDA does not
#if defined(USE_ROCM)
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits) {}
  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) && !defined(USE_ROCM)
  inline C10_HOST_DEVICE BFloat16(const __nv_bfloat16& value);
  explicit inline C10_HOST_DEVICE operator __nv_bfloat16() const;
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
  inline C10_HOST_DEVICE BFloat16(const sycl::ext::oneapi::bfloat16& value);
  explicit inline C10_HOST_DEVICE operator sycl::ext::oneapi::bfloat16() const;
#endif
};
     */
}
