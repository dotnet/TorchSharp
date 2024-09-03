using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static bool is_autocast_cache_enabled()
        {
            return THSAmp_is_autocast_cache_enabled();
        }

        public static bool is_autocast_enabled(Device device)
        {
            if(device.type == DeviceType.CPU)
                return THSAmp_is_autocast_cpu_enabled();
            if(device.type == DeviceType.CUDA)
                return THSAmp_is_autocast_gpu_enabled();
            return THSAmp_is_autocast_cache_enabled();
        }
        public static bool is_autocast_cpu_enabled()
        {
            return THSAmp_is_autocast_cpu_enabled();
        }
        public static bool is_autocast_gpu_enabled()
        {
            return THSAmp_is_autocast_gpu_enabled();
        }
        public static bool is_autocast_xpu_enabled()
        {
            return THSAmp_is_autocast_xpu_enabled();
        }
        public static bool is_autocast_hpu_enabled()
        {
            return THSAmp_is_autocast_hpu_enabled();
        }

        public static ScalarType get_autocast_dtype(Device device)
        {
            if (device.type == DeviceType.CPU)
                return get_autocast_cpu_dtype();
            if (device.type == DeviceType.CUDA)
                return get_autocast_gpu_dtype();
            return ScalarType.Float32;
        }
        public static ScalarType get_autocast_cpu_dtype()
        {
            return (ScalarType)THSAmp_get_autocast_cpu_dtype();
        }
        public static ScalarType get_autocast_gpu_dtype()
        {
            return (ScalarType)THSAmp_get_autocast_gpu_dtype();
        }
        public static ScalarType get_autocast_xpu_dtype()
        {
            return (ScalarType)THSAmp_get_autocast_xpu_dtype();
        }

        public static int autocast_increment_nesting()
        {
            return THSAmp_autocast_increment_nesting();
        }

        public static int autocast_decrement_nesting()
        {
            return THSAmp_autocast_decrement_nesting();
        }

        public static void set_autocast_enabled(bool enabled)
        {
            THSAmp_set_autocast_enabled(enabled);
        }
        public static void set_autocast_cache_enabled(bool enabled)
        {
            THSAmp_set_autocast_cache_enabled(enabled);
        }

        public static void set_autocast_cpu_dtype(ScalarType dtype)
        {
            THSAmp_set_autocast_cpu_dtype((sbyte)dtype);
        }
        public static void set_autocast_gpu_dtype(ScalarType dtype)
        {
            THSAmp_set_autocast_gpu_dtype((sbyte)dtype);
        }
        public static void set_autocast_xpu_dtype(ScalarType dtype)
        {
            THSAmp_set_autocast_xpu_dtype((sbyte)dtype);
        }

        public static void clear_autocast_cache()
        {
            THSAmp_clear_autocast_cache();
        }
    }
}