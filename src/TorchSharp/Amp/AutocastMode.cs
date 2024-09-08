using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.PInvoke;
using TorchSharp.Utils;

namespace TorchSharp.Amp
{
    /*public static class Autocast
    {
        public static torch.Tensor AutoCast(this torch.Tensor input)
        {
            return AutocastMode.GetInstance().CastTensor(input);
        }
    }*/
    //TODO: Should make Singleton and IDisposable on ENTER
    public sealed class AutocastMode : IDisposable
    {
        public bool _enabled=false;
        public bool IsEnter = false;
        public bool IsDisposed = false;
        private bool prev_cache_enabled, prev;
        private torch.ScalarType prev_fastdtype;
        //internal bool Prev;
        private bool _cache_enabled=false;
        internal torch.ScalarType fast_dtype = torch.ScalarType.Float32;
        internal torch.ScalarType? dtype = torch.ScalarType.Float32;
        public DeviceType device = DeviceType.CUDA;
        private static AutocastMode instance;
        public static AutocastMode GetInstance(bool enabled=false)
        {
            return instance ??= new AutocastMode(torch.cuda_is_available() ? torch.CUDA : torch.CPU, enabled:enabled,cache_enabled:true);
        }

        public torch.ScalarType GetFastType()
        {
            return torch.get_autocast_dtype(device);
        }
        private AutocastMode(torch.Device dev, torch.ScalarType? dtype = null, bool enabled=true, bool? cache_enabled = null)
        {
            /*dtype_by_methods[nameof(torch.matmul), DeviceType.CUDA] = torch.ScalarType.Float16;
            dtype_by_methods[nameof(torch.matmul), DeviceType.CUDA] = torch.ScalarType.Float16;*/
            //https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
            if (dtype == null)
                dtype = torch.get_autocast_dtype(dev.type);
            this.device = dev.type;
            if (!torch.is_autocast_available(device))
                throw new Exception($"User specified an unsupported autocast device_type {device}");
            fast_dtype = torch.get_autocast_dtype(device);
            //TODO: is_autocast_available();
            //IntPtr ptr = IntPtr.Zero;

            _cache_enabled = torch.is_autocast_cache_enabled();
            if (enabled && !torch.cuda_is_available() && dev.type == DeviceType.CUDA) //Is not available for doing multicast
                enabled = false;
            if (this.dtype.HasValue)
                fast_dtype = dtype.Value;
            if (cache_enabled.HasValue)
                _cache_enabled = cache_enabled.Value;

            if (dev.type == DeviceType.CPU) {
                if (fast_dtype != torch.ScalarType.Float16 || fast_dtype != torch.ScalarType.BFloat16) {
                    Debug.WriteLine($"In CPU autocast, but the target d type is not suported. Disabling autocast. CPU autocast only supports dtype of {torch.ScalarType.Float16} or {torch.ScalarType.BFloat16}");
                    enabled = false;
                }
            } else if (dev.type == DeviceType.CUDA) {

                if (enabled && fast_dtype == torch.ScalarType.BFloat16 && !torch.cuda.is_bf16_supported())
                    throw new Exception("Current CUDA Device does not support bfloat16. Please switch dtype to float16.");
            }
            this._enabled = enabled;
        }
        private torch.ScalarType GetType(IntPtr handle)
        {
            return (torch.ScalarType)NativeMethods.THSTensor_type(handle);
        }

        public static IntPtr AutoCast(IntPtr handle)
        {
            return ToIf(handle, GetInstance().GetFastType());
        }
        public static IntPtr AutoCast(IntPtr handle, torch.ScalarType dtype)
        {
            return ToIf(handle, dtype);
        }


        public static torch.Tensor AutoCast(torch.Tensor tensor)
        {
            return new torch.Tensor(AutoCast(tensor.Handle));
            //return tensor.to(AutocastMode.GetInstance().GetFastType());
        }
        public static IntPtr To(IntPtr ptr, torch.ScalarType type)
        {
            Debug.WriteLine($"{nameof(AutocastMode)} Tensor converting from: {(torch.ScalarType)NativeMethods.THSTensor_type(ptr)} to: {type}");
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }
        public static IntPtr ToIf(IntPtr ptr, torch.ScalarType type)
        {
            if (!GetInstance()._enabled)
                return ptr;
            /*if (!NativeMethods.THSAmp_is_autocast_enabled(NativeMethods.THSTensor_device_type(ptr)))
                return ptr;*/
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }
        public static IntPtr ToIf(IntPtr ptr, torch.ScalarType type, DeviceType device_type)
        {
            bool is_elegible = (torch.ScalarType)NativeMethods.THSTensor_type(ptr) != torch.ScalarType.Float64 && (DeviceType)NativeMethods.THSTensor_device_type(ptr) == device_type;
            
            if (!NativeMethods.THSAmp_is_autocast_enabled(NativeMethods.THSTensor_device_type(ptr)))
                return ptr;
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }

        public static bool IsAutocastEnabled(DeviceType device = DeviceType.CUDA)
        {
            return torch.is_autocast_enabled(!torch.cuda_is_available() ? DeviceType.CPU : device);
        }

        public IDisposable Enter()
        {
            prev_cache_enabled = torch.is_autocast_cache_enabled();
            prev = torch.is_autocast_enabled(device);
            prev_fastdtype = torch.get_autocast_dtype(device);
            torch.set_autocast_enabled(device, _enabled);
            torch.set_autocast_dtype(device, fast_dtype);
            torch.autocast_increment_nesting();
            torch.set_autocast_cache_enabled(_cache_enabled);
            return this;
        }

        private void Dispose(bool disposing)
        {
            this._enabled = false;
            if (torch.autocast_decrement_nesting() == 0)
                torch.clear_autocast_cache();
            torch.set_autocast_enabled(device, prev);
            torch.set_autocast_dtype(device, prev_fastdtype);
            torch.set_autocast_cache_enabled(prev_cache_enabled);
        }
        
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
