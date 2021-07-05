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
        internal ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth, InterpolateMode mode)
        {
            cropper = transforms.Crop(top, left, height, width);
            resizer = transforms.Resize(newHeight, newWidth, mode);
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
        static public ITransform ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth, InterpolateMode mode = InterpolateMode.Nearest)
        {
            return new ResizedCrop(top, left, height, width, newHeight, newWidth, mode);
        }

        static public ITransform ResizedCrop(int top, int left, int height, int width, int newSize, InterpolateMode mode = InterpolateMode.Nearest)
        {
            return new ResizedCrop(top, left, height, width, newSize, -1, mode);
        }
    }
}
