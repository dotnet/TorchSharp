using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType Forward_linear(Module.HType module, FloatTensor.HType tensor);

        public override FloatTensor Forward(FloatTensor tensor)
        {
            return new FloatTensor(Forward_linear(handle, tensor.handle));
        }

        [DllImport("LibTorchSharp")]
        extern static void Zero_grad_linear(Module.HType module);

        public override void ZeroGrad()
        {
            Zero_grad_linear(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void Param_linear(Module.HType module, AllocatePinnedArray allocator);

        public override IEnumerable<FloatTensor> Parameters()
        {
            Tensor[] ros;

            using (var pa = new PinnedArray<Tensor>())
            {
                Param_linear(handle, pa.CreateArray);
                ros = pa.Array;
            }
            return ros.Select(x => new FloatTensor(new FloatTensor.HType(x.ptr, true)));
        }
    }
}
