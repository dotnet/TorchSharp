// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public sealed class Permute : nn.Module<Tensor, Tensor>
        {
            public Permute(IEnumerable<long> dims) : base(nameof(Permute))
            {
                this.dims = dims.ToArray();
            }

            public override Tensor forward(Tensor input)
            {
                return torch.permute(input, dims);
            }

            private long[] dims;
        }

        public static partial class ops
        {
            /// <summary>
            /// This module returns a view of the tensor input with its dimensions permuted.
            /// </summary>
            /// <param name="dims">The desired ordering of dimensions</param>
            public static Permute Permute(params long[] dims)
            {
                return new Permute(dims);
            }
        }
    }
}
