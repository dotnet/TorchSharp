// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;
using static TorchSharp.nn;

namespace TorchSharp.TorchVision
{
    internal class ResizedCrop : ITransform
    {
        internal ResizedCrop(int top, int left, int height, int width, int newHeight, int newWidth, InterpolateMode mode)
        {
            cropper = Transforms.Crop(top, left, height, width);
            resizer = Transforms.Resize(newHeight, newWidth, mode);
        }

        public TorchTensor forward(TorchTensor input)
        {
            return resizer.forward(cropper.forward(input));
        }

        private ITransform cropper;
        private ITransform resizer;
    }

    public static partial class Transforms
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
