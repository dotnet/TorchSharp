// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a functional module (e.g., ReLU).
    /// </summary>
    public abstract class ProvidedModule : Module
    {
        internal ProvidedModule() : base(IntPtr.Zero)
        {
        }

        internal ProvidedModule(IntPtr handle) : base(handle)
        {
        }
    }
}
