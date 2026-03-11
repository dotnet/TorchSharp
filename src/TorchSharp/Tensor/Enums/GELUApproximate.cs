// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    /// <summary>
    /// Specifies the approximation method for the GELU activation function.
    /// </summary>
    public enum GELUApproximate
    {
        /// <summary>
        /// Exact GELU computation.
        /// </summary>
        none,
        /// <summary>
        /// Tanh-based approximation.
        /// </summary>
        tanh
    }
}
