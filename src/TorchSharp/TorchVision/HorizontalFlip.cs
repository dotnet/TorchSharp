// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;


namespace TorchSharp.TorchVision
{
    internal class HorizontalFlip : ITransform
    {
        internal HorizontalFlip()
        {
        }

        public TorchTensor forward(TorchTensor input)
        {
            return input.fliplr();
        }
    }

    public static partial class Transforms
    {
        /// <summary>
        /// Flip the image horizontally.
        /// </summary>
        /// <returns></returns>
        static public ITransform HorizontalFlip()
        {
            return new HorizontalFlip();
        }
    }
}
