// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

namespace TorchSharp
{
    /// <summary>
    /// Interface to implement progress bar.
    /// </summary>
    public interface IProgressBar : IDisposable
    {
        /// <summary>
        /// The current position of the progress bar
        /// </summary>
        public long Value { set; get; }

        /// <summary>
        /// The maximum position of the progress bar
        /// </summary>
        public long? Maximum { set; get; }
    }
}