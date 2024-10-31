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
        /// This class is used to represent a flattening of the input tensors.
        /// </summary>
        public sealed class Flatten : ParameterLessModule<Tensor, Tensor>
        {
            internal Flatten(long start_dim = 1, long end_dim = -1) : base(nameof(Flatten))
            {
                this.start_dim = start_dim;
                this.end_dim = end_dim;
            }

            public override Tensor forward(Tensor input)
            {
                return input.flatten(start_dim, end_dim);
            }

            public long start_dim { get; set; }
            public long end_dim { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Flattens a contiguous range of dims into a tensor. For use with Sequential.
            /// </summary>
            /// <param name="start_dim">First dim to flatten (default = 1).</param>
            /// <param name="end_dim">Last dim to flatten (default = -1).</param>
            /// <returns></returns>
            public static Flatten Flatten(long start_dim = 1, long end_dim = -1)
            {
                return new Flatten(start_dim, end_dim);
            }
        }
    }
}
