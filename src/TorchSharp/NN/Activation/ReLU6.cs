// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;
    using TorchSharp.PInvoke;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a ReLU6 module.
        /// </summary>
        public sealed class ReLU6 : ParameterLessModule<Tensor, Tensor>
        {
            internal ReLU6(bool inplace) : base(nameof(ReLU6))
            {
                this.inplace = inplace;
            }


            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.relu6(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Rectified Linear Unit
            ///
            /// This ReLU version caps positive values at 6.
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static ReLU6 ReLU6(bool inplace = false)
            {
                return new ReLU6(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Rectified Linear Unit
                ///
                /// This ReLU version caps positive values at 6.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor relu6(Tensor x, bool inplace = false)
                {
                    return inplace ? x.relu6_().alias() : x.relu6();
                }
            }
        }
    }
}
