// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;


namespace TorchSharp.torchvision
{
    internal class Crop : ITransform
    {
        internal Crop(int top, int left, int height, int width)
        {
            this.top = top;
            this.left = left;
            this.height = height;
            this.width = width;
        }

        public TorchTensor forward(TorchTensor input)
        {
            return input.crop(top, left, height, width);
        }

        private int top, left, height, width;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Crop the image.
        /// </summary>
        /// <returns></returns>
        /// <remarks>The image will not be cropped outside its boundaries.</remarks>
        static public ITransform Crop(int top, int left, int height, int width)
        {
            return new Crop(top, left, height, width);
        }
    }
}
