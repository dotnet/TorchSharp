// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a log softmax module.
        /// </summary>
        public sealed class LogSoftmax : ParameterLessModule<Tensor, Tensor>
        {
            internal LogSoftmax(long dim) : base(nameof(LogSoftmax))
            {
                this.dim = dim;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.log_softmax(tensor, dim);
            }

            public long dim { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public static LogSoftmax LogSoftmax(long dim)
            {
                return new LogSoftmax(dim);
            }

            public static partial class functional
            {
                public static Tensor log_softmax(Tensor x, long dim)
                {
                    return torch.special.log_softmax(x, dim);
                }
            }
        }
    }
}
