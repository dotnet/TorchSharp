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
        /// This class is used to represent a Tanh module.
        /// </summary>
        public sealed class Tanh : ParameterLessModule<Tensor, Tensor>
        {
            internal Tanh(bool inplace) : base(nameof(Tanh))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.tanh(tensor, inplace);
            }

            public override string GetName()
            {
                return typeof(Tanh).Name;
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Tanh activation
            /// </summary>
            /// <returns></returns>
            public static Tanh Tanh()
            {
                return new Tanh(false);
            }

            /// <summary>
            /// Tanh activation
            /// </summary>
            /// <returns></returns>
            public static Tanh Tanh(bool inplace = false)
            {
                return new Tanh(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Tanh activation
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor tanh(Tensor x, bool inplace = false)
                {
                    return inplace ? x.tanh_().alias() : x.tanh();
                }
            }
        }
    }
}
