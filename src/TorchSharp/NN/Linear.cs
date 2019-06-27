using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Linear : ProvidedModule
    {
        public Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_linearModule(long input_size, long output_size, bool with_bias);

        public Linear(long inputSize, long outputSize, bool hasBias = false) : base()
        {
            handle = new HType(THSNN_linearModule(inputSize, outputSize, hasBias), true);
        }

        [DllImport("LibTorchSharp")]
        extern static bool THSNN_linear_with_bias(Module.HType module);

        public bool WithBias
        {
            get { return THSNN_linear_with_bias(handle); }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_linear_get_bias(Module.HType module);

        [DllImport("LibTorchSharp")]
        extern static void THSNN_linear_set_bias(Module.HType module, IntPtr tensor);

        public TorchTensor? Bias
        {
            get
            {
                var bias = THSNN_linear_get_bias(handle);
                return bias == IntPtr.Zero ? (TorchTensor?)null : new TorchTensor(bias);
            }
            set { THSNN_linear_set_bias(handle, value?.Handle ?? throw new ArgumentNullException("bias")); }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_linear_get_weight(Module.HType module);

        [DllImport("LibTorchSharp")]
        extern static void THSNN_linear_set_weight(Module.HType module, IntPtr tensor);

        public TorchTensor Weight
        {
            get
            {
                return new TorchTensor(THSNN_linear_get_weight(handle));
            }
            set { THSNN_linear_set_weight(handle, value.Handle); }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_linearModuleApply(Module.HType module, IntPtr tensor);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_linearModuleApply(handle, tensor.Handle));
        }
    }
}
