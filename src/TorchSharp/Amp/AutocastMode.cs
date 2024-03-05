using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchSharp.Amp
{
    public static class Autocast
    {
        public static torch.Tensor AutoCast(this torch.Tensor input)
        {
            return AutocastMode.GetInstance().CastTensor(input);
        }
    }
    //TODO: Should make Singleton and IDisposable on ENTER
    public sealed class AutocastMode : IDisposable
    {
        private bool Enabled, Prev;
        //private torch.ScalarType Dtype = torch.ScalarType.Float32;
        private torch.ScalarType fast_dtype = torch.ScalarType.Float32;
        private torch.Device Device = new torch.Device(DeviceType.CUDA);
        private static AutocastMode instance;
        /*public static AutocastMode GetInstance(torch.Device dev, torch.ScalarType? dtype = null, bool enabled = true, bool? cache_enabled = null)
        {
            if(instance ==null)
                instance = new AutocastMode(dev, dtype, enabled, cache_enabled);
            return instance;
        }*/
        public static AutocastMode GetInstance()
        {
            return instance ?? (instance = new AutocastMode(torch.CUDA, cache_enabled:true));
        }

        private AutocastMode(torch.Device dev, torch.ScalarType? dtype = null, bool enabled=true, bool? cache_enabled = null)
        {
            //var la = torch.tensor(9);
            fast_dtype = dtype ?? torch.ScalarType.Float32;
            if (dev.type == DeviceType.CUDA)
                fast_dtype = torch.get_autocast_gpu_dtype();
            if (dev.type == DeviceType.CPU)
                fast_dtype = torch.get_autocast_cpu_dtype();
            IntPtr ptr = IntPtr.Zero;
            
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

            torch.set_autocast_cache_enabled(_cache_enabled);
            torch.set_autocast_enabled(this.Enabled);
            //throw new NotImplementedException();
        }

        /*internal void Cast(torch.Tensor tensor)
        {
            tensor.to(fast_dtype, tensor.device);
        }*/

        internal torch.Tensor CastTensor(torch.Tensor tensor)
        {
            if (!Enabled)
                return tensor;
            return tensor.to(fast_dtype, tensor.device);
        }
        /*public IDisposable Enter()
        {

            return this;
        }*/
        public void Dispose()
        {
            this.Enabled = false;
            if (Device.type == DeviceType.CUDA) {
                if(torch.autocast_decrement_nesting() == 0)
                    torch.clear_autocast_cache();
                torch.set_autocast_gpu_dtype(this.fast_dtype);
                //torch.set_autocast_enabled(this.Prev);
                torch.set_autocast_enabled(false);
                torch.set_autocast_cache_enabled(false);
            }

            if (Device.type == DeviceType.CPU) {
                if (torch.autocast_decrement_nesting() == 0)
                    torch.clear_autocast_cache();
                //torch.set_autocast_enabled(this.Prev);
                torch.set_autocast_cpu_dtype(this.fast_dtype);
                torch.set_autocast_enabled(false);
                torch.set_autocast_cache_enabled(false);
            }
            //throw new NotImplementedException();
        }
    }
}
