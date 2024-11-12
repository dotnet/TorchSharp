// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/ops/misc.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Linq;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using TorchSharp.Modules;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public sealed class Permute : ParameterLessModule<Tensor, Tensor>
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
