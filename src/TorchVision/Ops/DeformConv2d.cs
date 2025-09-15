using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchVision.Modules;
using static TorchSharp.torch;

namespace TorchVision
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            public static Modules.DeformConv2d DeformConv2d()
            {
                return new DeformConv2d();
            }
        }
    }

    namespace Modules
    {
        public class DeformConv2d : torch.nn.Module<Tensor, Tensor, Tensor, Tensor>
        {
            protected internal DeformConv2d() : base(nameof(DeformConv2d))
            {

            }
            public override Tensor forward(Tensor input, Tensor offset, Tensor mask)
            {
                throw new NotImplementedException();
            }
        }
    }
}
