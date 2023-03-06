// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Lambda : ITransform
        {
            internal Lambda(Func<Tensor, Tensor> lambda)
            {
                this.lambda = lambda;
            }

            public Tensor call(Tensor img)
            {
                return lambda(img);
            }

            private Func<Tensor, Tensor> lambda;
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
}