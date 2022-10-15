// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class datasets
        {
            public abstract class VisionDataset<T, TDataInput, TDataTarget, TInput, TTarget> : torch.utils.data.Dataset<T>
            {
                protected readonly string root;
                protected readonly Func<TDataInput, TDataTarget, (TInput, TTarget)> transforms;

                public VisionDataset(
                    string root,
                    Func<TDataInput, TDataTarget, (TInput, TTarget)> transforms)
                {
                    this.root = root;
                    this.transforms = transforms;
                }
            }
        }
    }
}
