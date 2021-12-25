// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class data
            {
                public abstract class Dataset : IDisposable
                {
                    public virtual void Dispose()
                    {
                    }

                    /// <summary>
                    /// Size of dataset
                    /// </summary>
                    public abstract long Count { get; }

                    /// <summary>
                    /// Get tensor via index
                    /// </summary>
                    /// <param name="index">Index for tensor</param>
                    /// <returns>Tensor for index</returns>
                    public abstract Dictionary<string, Tensor> GetTensor(long index);
                }
            }
        }
    }
}