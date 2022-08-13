// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Gamma distribution parameterized by a single shape parameter.
        /// </summary>
        public class Chi2 : Gamma
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="df">Shape parameter of the distribution</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Chi2(Tensor df, torch.Generator generator = null) : base(df * 0.5, torch.tensor(0.5), generator)
            {
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Chi2))
                    throw new ArgumentException("expand(): 'instance' must be a Chi2 distribution");

                if (instance == null) {
                    instance = new Chi2(concentration, generator);
                }
                return base.expand(batch_shape, instance);
            }

            public Tensor df => concentration * 2;
        }
    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Gamma distribution parameterized by a single shape parameter.
            /// </summary>
            /// <param name="df">Shape parameter of the distribution</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Chi2 Chi2(Tensor df, torch.Generator generator = null)
            {
                return new Chi2(df, generator);
            }
        }
    }
}
