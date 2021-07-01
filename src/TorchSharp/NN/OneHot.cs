// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    public static partial class nn
    {
        public static partial class functional
        {
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_one_hot(IntPtr self, long num_classes);

            static public TorchTensor OneHot(TorchTensor x, long num_classes = -1)
            {
                if (x.Type != ScalarType.Int64) throw new ArgumentException("OneHot input tensor must have elements of type Int64");
                var res = THSNN_one_hot(x.Handle, num_classes);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }
    }
}
