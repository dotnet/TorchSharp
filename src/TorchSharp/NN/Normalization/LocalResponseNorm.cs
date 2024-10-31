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
        /// This class is used to represent a LocalResponseNorm module.
        /// </summary>
        public sealed class LocalResponseNorm : ParameterLessModule<Tensor, Tensor>
        {
            internal LocalResponseNorm(long size, double alpha = 0.0001, double beta = 0.75, double k = 1.0) : base(nameof(LocalResponseNorm))
            {
                this.size = size;
                this.alpha = alpha;
                this.beta = beta;
                this.k = k;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.local_response_norm(input, this.size, this.alpha, this.beta, this.k);
            }

            public long size { get; set; }
            public double alpha { get; set; }
            public double beta { get; set; }
            public double k { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies local response normalization over an input signal composed of several input planes, where channels occupy the second dimension. Applies normalization across channels.
            /// </summary>
            public static LocalResponseNorm LocalResponseNorm(long size, double alpha = 0.0001, double beta = 0.75, double k = 1.0)
            {
                return new LocalResponseNorm(size, alpha, beta, k);
            }

            public static partial class functional
            {

                /// <summary>
                /// Applies local response normalization over an input signal.
                /// The input signal is composed of several input planes, where channels occupy the second dimension.
                /// Applies normalization across channels.
                /// </summary>
                public static Tensor local_response_norm(Tensor input, long size, double alpha = 0.0001, double beta = 0.75, double k = 1.0)
                {
                    if (input.Dimensions < 3) throw new ArgumentException($"Invalid number of dimensions for LocalResponseNorm argument: {input.Dimensions}");
                    var res = THSNN_local_response_norm(input.Handle, size, alpha, beta, k);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
            }
        }
    }
}
