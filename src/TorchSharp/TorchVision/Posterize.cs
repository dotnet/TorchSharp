// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;
using static TorchSharp.nn;

namespace TorchSharp.TorchVision
{
    internal class Posterize : ITransform
    {
        internal Posterize(int bits)
        {
            this.bits = bits;
        }

        public TorchTensor forward(TorchTensor input)
        {
            if (input.Type != ScalarType.Byte) throw new ArgumentException("Only torch.byte image tensors are supported");
            var mask = -(1 << (8-bits));
            return input & ByteTensor.from((byte)mask);
        }

        protected int bits;
    }

    public static partial class Transforms
    {
        /// <summary>
        /// Posterize the image by reducing the number of bits for each color channel. 
        /// </summary>
        /// <param name="bits">Number of high-order bits to keep for each channel.</param>
        /// <returns></returns>
        /// <remarks>The tensor must be an integer tensor.</remarks>
        static public ITransform Posterize(int bits)
        {
            return new Posterize(bits);
        }
    }
}
