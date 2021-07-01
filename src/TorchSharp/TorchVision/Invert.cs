// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


namespace TorchSharp.torchvision
{
    internal class Invert : ITransform
    {
        internal Invert()
        {
        }

        public Tensor forward(Tensor input)
        {
            if (input.IsIntegral()) {
                return -input + 255;
            }
            else {
                return -input + 1.0.ToScalar();
            }
        }
    }

    public static partial class transforms
    {
        /// <summary>
        /// Invert the image colors.
        /// </summary>
        /// <remarks>The code assumes that integer color values lie in the range [0,255], and floating point colors in [0,1[.</remarks>
        static public ITransform Invert()
        {
            return new Invert();
        }
    }
}
