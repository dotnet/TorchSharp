// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        internal static extern bool THSTorchCuda_is_available();

        [DllImport("LibTorchSharp")]
        internal static extern bool THSTorchCuda_cudnn_is_available();

        [DllImport("LibTorchSharp")]
        internal static extern int THSTorchCuda_device_count();
    }
}
