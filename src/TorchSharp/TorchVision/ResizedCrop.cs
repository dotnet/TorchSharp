// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    internal class ResizedCrop : ITransform
    {
        internal ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth)
        {
            cropper = transforms.Crop(top, left, height, width);
            resizer = transforms.Resize(newHeight, newWidth);
        }

        public Tensor forward(Tensor input)
        {
            return resizer.forward(cropper.forward(input));
        }

        private ITransform cropper;
        private ITransform resizer;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Crop the given image and resize it to desired size.
        /// </summary>
        /// <param name="top">Vertical component of the top left corner of the crop box.</param>
        /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
        /// <param name="height">Height of the crop box.</param>
        /// <param name="width">Width of the crop box.</param>
        /// <param name="newHeight">New height.</param>
        /// <param name="newWidth">New width.</param>

        static public ITransform ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth)
        {
            return new ResizedCrop(top, left, height, width, newHeight, newWidth);
        }

        /// <summary>
        /// Crop the given image and resize it to desired (square) size.
        /// </summary>
        /// <param name="top">Vertical component of the top left corner of the crop box.</param>
        /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
        /// <param name="size">Height and width of the crop box.</param>
        /// <param name="newSize">New height and width.</param>
        static public ITransform ResizedCrop(int top, int left, int size, int newSize)
        {
            return new ResizedCrop(top, left, size, size, newSize, -1);
        }
    }
}
