using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_linearModule_Forward(Module.HType module, FloatTensor.HType tensor);

        public override FloatTensor Forward(FloatTensor tensor)
        {
            return new FloatTensor(NN_linearModule_Forward(handle, tensor.handle));
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_linearModule_ZeroGrad(Module.HType module);

        public override void ZeroGrad()
        {
            NN_linearModule_ZeroGrad(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_linearModule_GetParameters(Module.HType module, AllocatePinnedArray allocator);

        public override IEnumerable<FloatTensor> Parameters()
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
