// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Text;

using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {

        public class Chi2 : Gamma
        {
            public Chi2(Tensor df, torch.Generator generator = null) : base(df * 0.5, torch.tensor(0.5), generator)
            {
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Gamma))
                    throw new ArgumentException("expand(): 'instance' must be a Chi2 distribution");

                if (instance == null) {
                    instance = new Chi2(concentration);
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
