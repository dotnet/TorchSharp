// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


namespace TorchSharp.torchvision
{
    internal class HorizontalFlip : ITransform
    {
        internal HorizontalFlip()
        {
        }

        public Tensor forward(Tensor input)
        {
            return input.flip(-1);
        }
    }

    public static partial class transforms
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
