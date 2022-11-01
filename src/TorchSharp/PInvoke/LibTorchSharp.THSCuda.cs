// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_manual_seed(long seed);

        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_manual_seed_all(long seed);

        [DllImport("LibTorchSharp")]
        internal static extern void THSCuda_synchronize(long device_index);
    }
}
