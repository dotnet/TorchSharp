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
        /// This class is used to represent a Hardtanh module.
        /// </summary>
        public sealed class Hardtanh : ParameterLessModule<Tensor, Tensor>
        {
            internal Hardtanh(double min_val = -1.0, double max_val = 1.0, bool inplace = false) : base(nameof(Hardtanh))
            {
                this.min_val = min_val;
                this.max_val = max_val;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.hardtanh(tensor, min_val, max_val, inplace);
            }

            public override string GetName()
            {
                return typeof(Hardtanh).Name;
            }

            public double min_val { get; set; }
            public double max_val { get; set; }
            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Hardtanh
            /// </summary>
            /// <param name="min_val">Minimum value of the linear region range.</param>
            /// <param name="max_val">Maximum value of the linear region range.</param>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            public static Hardtanh Hardtanh(double min_val = -1.0, double max_val = 1.0, bool inplace = false)
            {
                return new Hardtanh(min_val, max_val, inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Hardtanh
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="min_val">Minimum value of the linear region range.</param>
                /// <param name="max_val">Maximum value of the linear region range.</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <returns></returns>
                public static Tensor hardtanh(Tensor x, double min_val = -1.0, double max_val = 1.0, bool inplace = false)
                {
                    return inplace ? x.hardtanh_(min_val, max_val).alias() : x.hardtanh(min_val, max_val);
                }

                /// <summary>
                /// Hardshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="min_val">Minimum value of the linear region range.</param>
                /// <param name="max_val">Maximum value of the linear region range.</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <remarks>Only here for backward comaptibility.</remarks>
                [Obsolete("Not using the PyTorch naming convention.",false)]
                public static Tensor Hardtanh(Tensor x, double min_val = -1.0, double max_val = 1.0, bool inplace = false) => hardtanh(x, min_val, max_val, inplace);
            }
        }
    }
}
