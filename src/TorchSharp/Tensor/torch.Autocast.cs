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

        public static bool is_autocast_enabled(DeviceType device)
        {
            return THSAmp_is_autocast_enabled((int)device);
            //return THSAmp_is_autocast_cache_enabled();
        }
        public static ScalarType get_autocast_dtype(DeviceType device)
        {
            return (ScalarType)THSAmp_get_autocast_dtype((int)device);
            /*if (device.type == DeviceType.CPU)
                return get_autocast_cpu_dtype();
            if (device.type == DeviceType.CUDA)
                return get_autocast_gpu_dtype();
            return ScalarType.Float32;*/
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
        public static void set_autocast_cache_enabled(DeviceType device, ScalarType dtype)
        {
            THSAmp_set_autocast_dtype((int)device, (sbyte)dtype);
        }

        public static void clear_autocast_cache()
        {
            THSAmp_clear_autocast_cache();
        }
    }
}