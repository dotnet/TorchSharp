// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;
using TorchSharp.NN;

namespace TorchSharp.TorchVision
{
    internal class Invert : ITransform
    {
        internal Invert()
        {
        }

        public TorchTensor forward(TorchTensor input)
        {
            if (input.IsIntegral()) {
                return -input + 255;
            }
            else {
                return -input + 1.0.ToScalar();
            }
        }
    }

    public static partial class Transforms
    {
        static public ITransform Invert()
        {
            return new Invert();
        }
    }
}
