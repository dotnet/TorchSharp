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
        /// This class is used to represent a GLU (gated linear unit) module.
        /// </summary>
        public sealed class GLU : ParameterLessModule<Tensor, Tensor>
        {
            internal GLU(long dim) : base(nameof(GLU))
            {
                this.dim = dim;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.glu(tensor, dim);
            }

            public long dim {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Gated Linear Unit
            /// </summary>
            /// <param name="dim">the dimension on which to split the input. Default: -1</param>
            /// <returns></returns>
            public static GLU GLU(long dim = -1)
            {
                return new GLU(dim);
            }

            public static partial class functional
            {
                /// <summary>
                /// The gated linear unit function.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="dim">the dimension on which to split the input. Default: -1</param>
                /// <returns></returns>
                public static Tensor glu(Tensor input, long dim = -1)
                {
                    return input.glu(dim);
                }
            }
        }
    }
}
