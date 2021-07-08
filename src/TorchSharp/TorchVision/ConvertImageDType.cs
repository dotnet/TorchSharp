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
            this.output_max = MaxValue(dtype);
        }

        public Tensor forward(Tensor image)
        {
            if (image.dtype == this.dtype)
                return image;

            if (torch.is_floating_point(image)) {

                if (torch.is_floating_point(this.dtype)) {
                    return image.to_type(dtype);
                }

                if ((image.dtype == torch.float32 && (dtype == torch.int32 || dtype == torch.int64)) ||
                    (image.dtype == torch.float64 && dtype == torch.int64))        {
                    throw new ArgumentException($"The cast from {image.dtype} to {dtype} cannot be performed safely.");
                }

                var eps = 1e-3;
                var result = image.mul(output_max + 1.0 - eps);
                return result.to_type(dtype);

            } else {
                // Integer to floating point.

                var input_max = MaxValue(image.dtype);

                if (torch.is_floating_point(this.dtype)) {
                    return image.to_type(dtype) / input_max;
                }

                if (input_max > output_max) {
                    var factor = (input_max + 1) / (output_max + 1);
                    image = torch.div(image, factor);
                    return image.to_type(dtype);
                }
                else {
                    var factor = (output_max + 1) / (input_max + 1);
                    image = image.to_type(dtype);
                    return image * factor;
                }
            }
        }

        private long MaxValue(ScalarType dtype)
        {
            switch (dtype) {
            case ScalarType.Byte:
                return byte.MaxValue;
            case ScalarType.Int8:
                return sbyte.MaxValue;
            case ScalarType.Int16:
                return short.MaxValue;
            case ScalarType.Int32:
                return int.MaxValue;
            case ScalarType.Int64:
                return long.MaxValue;
            }

            return 0L;
        }

        private ScalarType dtype;
        private long output_max;
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
