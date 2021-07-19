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
        /// Apply a user-defined function as a transform. 
        /// </summary>
        /// <param name="lambda">Lambda/function to be used for transform.</param>
        /// <returns></returns>
        static public ITransform Lambda(Func<Tensor, Tensor> lambda)
        {
            return new Lambda(lambda);
        }
    }
}
