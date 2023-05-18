// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class functional
            {
                /// <summary>
                /// Takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes) that have zeros
                /// everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be 1.
                /// </summary>
                /// <param name="x">Category values of any shape</param>
                /// <param name="num_classes">Total number of classes.
                /// If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor</param>
                /// <returns></returns>
                public static Tensor one_hot(Tensor x, long num_classes = -1)
                {
                    if (x.dtype != ScalarType.Int64) throw new ArgumentException("OneHot input tensor must have elements of type Int64");
                    var res = THSNN_one_hot(x.Handle, num_classes);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
