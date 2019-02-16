using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_linearModule_Forward(Module.HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_linearModule_Forward(handle, tensor.Handle));
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_linearModule_ZeroGrad(HType module);

        public override void ZeroGrad()
        {
            NN_linearModule_ZeroGrad(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_linearModule_GetParameters(HType module, AllocatePinnedArray allocator);

        public override IEnumerable<ITorchTensor<float>> Parameters()
        {
            TensorPointerWrapper[] ros;

            using (var pa = new PinnedArray<TensorPointerWrapper>())
            {
                NN_linearModule_GetParameters(handle, pa.CreateArray);
                ros = pa.Array;
            }
            return ros.Select(x => new FloatTensor(new FloatTensor.HType(x.ptr, true)));
        }
    }
}
