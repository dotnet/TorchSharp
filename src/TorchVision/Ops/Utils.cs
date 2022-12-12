using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.
            /// </summary>
            public static Tensor _upcast(Tensor t)
            {
                if (t.is_floating_point())
                    return t.dtype == torch.float32 || t.dtype == torch.float64 ? t : t.@float();
                else
                    return t.dtype == torch.int32 || t.dtype == torch.int64 ? t : t.@int();
            }
        }
    }
}
