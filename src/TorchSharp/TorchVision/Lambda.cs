// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;


namespace TorchSharp.torchvision
{
    internal class Lambda : ITransform
    {
        internal Lambda(Func<Tensor, Tensor> lambda)
        {
            this.lambda = lambda;
        }

        public Tensor forward(Tensor img)
        {
            return lambda(img);
        }

        private Func<Tensor,Tensor> lambda;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Crop the image.
        /// </summary>
        /// <returns></returns>
        /// <remarks>The image will not be cropped outside its boundaries.</remarks>
        static public ITransform Lambda(Func<Tensor, Tensor> lambda)
        {
            return new Lambda(lambda);
        }
    }
}
