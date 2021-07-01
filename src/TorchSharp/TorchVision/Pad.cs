// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;
using static TorchSharp.nn;

namespace TorchSharp.TorchVision
{
    internal class Pad : ITransform
    {
        internal Pad(long[] pad, PaddingModes mode = PaddingModes.Constant, double value = 0)
        {
            this.pad = pad;
            this.mode = mode;
            this.value = value;
        }

        public TorchTensor forward(TorchTensor input)
        {
            return TorchSharp.nn.functional.Pad(input, pad, mode, value);
        }

        private long[] pad;
        private PaddingModes mode;
        private double value;
    }

    public static partial class Transforms
    {
        static public ITransform Pad(long[] pad, PaddingModes mode = PaddingModes.Constant, double value = 0)
        {
            return new Pad(pad, mode, value);
        }
    }
}
