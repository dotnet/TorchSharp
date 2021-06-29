// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;
using TorchSharp.NN;

namespace TorchSharp.TorchVision
{
    internal class Solarize : ITransform
    {
        internal Solarize(double threshold)
        {
            this.threshold = threshold;
        }

        public TorchTensor forward(TorchTensor input)
        {
            using (var inverted = Transforms.Invert().forward(input))
                return input.where(input < threshold, inverted);
        }

        protected double threshold;
    }

    public static partial class Transforms
    {
        static public ITransform Solarize(double threshold)
        {
            return new Solarize(threshold);
        }
    }
}
