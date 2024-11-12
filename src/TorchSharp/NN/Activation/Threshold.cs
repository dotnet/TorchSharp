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
        /// This class is used to represent a Threshold module.
        /// </summary>
        public sealed class Threshold : ParameterLessModule<Tensor, Tensor>
        {
            internal Threshold(double threshold, double value, bool inplace) : base(nameof(Threshold))
            {
                this.inplace = inplace;
                this.threshold = threshold;
                this.value = value;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.threshold(tensor, threshold, value, inplace);
            }

            public double threshold {get; set;}

            public double value {get; set;}

            public bool inplace {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Threshold
            /// </summary>
            /// <param name="threshold">The value to threshold at</param>
            /// <param name="value">The value to replace with</param>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            public static Threshold Threshold(double threshold, double value, bool inplace = false)
            {
                return new Threshold(threshold, value, inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Thresholds each element of the input Tensor.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="threshold">The value to threshold at</param>
                /// <param name="value">The value to replace with</param>
                /// <param name="inplace">Do the operation in-place</param>
                public static Tensor threshold(Tensor x, double threshold, double value, bool inplace = false)
                {
                    return inplace ? x.threshold_(threshold, value).alias() : x.threshold(threshold, value);
                }

                /// <summary>
                /// Thresholds each element of the input Tensor.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="threshold">The value to threshold at</param>
                /// <param name="value">The value to replace with</param>
                /// <param name="inplace">Do the operation in-place</param>
                [Obsolete("Not using the PyTorch naming convention.",false)]
                public static Tensor Threshold(Tensor x, double threshold, double value, bool inplace = false) => nn.functional.threshold(x, threshold, value, inplace);
            }
        }
    }
}
