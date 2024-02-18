using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.Amp
{
    public class AutocastMode : IDisposable
    {
        private bool Enabled, Prev;
        //private torch.ScalarType Dtype = torch.ScalarType.Float32;
        private torch.ScalarType fast_dtype = torch.ScalarType.Float32;
        private torch.Device Device = new torch.Device(DeviceType.CUDA);
        public AutocastMode(torch.Device dev, torch.ScalarType? dtype = null, bool enabled=true, bool? cache_enabled = null)
        {
            fast_dtype = dtype.Value;
            if (dev.type == DeviceType.CUDA)
                fast_dtype = torch.get_autocast_gpu_dtype();
            if (dev.type == DeviceType.CPU)
                fast_dtype = torch.get_autocast_cpu_dtype();

            bool _cache_enabled = torch.is_autocast_cache_enabled();
            if (!torch.cuda.is_available() && dev.type == DeviceType.CUDA) //Is not available for doing multicast
                Enabled = false;
            if (dtype.HasValue)
                fast_dtype = dtype.Value;
            if(cache_enabled.HasValue)
                _cache_enabled=cache_enabled.Value;

            if (dev.type == DeviceType.CUDA) {
                if (enabled && fast_dtype == torch.ScalarType.BFloat16 && !torch.cuda.is_bf16_supported())
                    throw new Exception("Current CUDA Device does not support bfloat16. Please switch dtype to float16.");
            }
            this.Enabled = enabled;

            this.Prev = torch.is_autocast_cpu_enabled();
            if (dev.type == DeviceType.CUDA) {
                this.Prev = torch.is_autocast_gpu_enabled();
            }
            throw new NotImplementedException();
        }
        public void Dispose()
        {
            if (Device.type == DeviceType.CUDA) {
                if(torch.autocast_decrement_nesting() == 0)
                    torch.clear_autocast_cache();
                torch.set_autocast_gpu_dtype(this.fast_dtype);
                torch.set_autocast_enabled(this.Prev);
            }
            throw new NotImplementedException();
        }
    }
}
