// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Hardsigmoid module.
        /// </summary>
        public class Hardsigmoid : torch.nn.Module
        {
            internal Hardsigmoid(bool inplace = false) : base(nameof(Hardsigmoid))
            {
                this.inplace = inplace;
            }
            private bool inplace;

            public override Tensor forward(Tensor tensor)
            {
                return inplace ? tensor.hardsigmoid_() : tensor.hardsigmoid();
            }

            public override string GetName()
            {
                return typeof(Hardsigmoid).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Hardsigmoid
            /// </summary>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            static public Hardsigmoid Hardsigmoid(bool inplace = false)
            {
                return new Hardsigmoid(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Hardsigmoid
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <returns></returns>
                static public Tensor Hardsigmoid(Tensor x, bool inplace = false)
                {
                    using (var m = nn.Hardsigmoid(inplace)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
