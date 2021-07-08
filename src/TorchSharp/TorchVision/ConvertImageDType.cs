// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.TensorExtensionMethods;


namespace TorchSharp.torchvision
{
    internal class ConvertImageDType : ITransform
    {
        internal ConvertImageDType(ScalarType dtype)
        {
            this.dtype = dtype;
        }

        public Tensor forward(Tensor image)
        {
            return transforms.functional.convert_image_dtype(image, dtype);
        }

        private ScalarType dtype;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Crop the image.
        /// </summary>
        /// <returns></returns>
        /// <remarks>The image will not be cropped outside its boundaries.</remarks>
        static public ITransform ConvertImageDType(ScalarType dtype)
        {
            return new ConvertImageDType(dtype);
        }
    }
}
