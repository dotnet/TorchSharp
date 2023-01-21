// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Diagnostics.Contracts;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#parallelism
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.get_num_threads
        [Pure, Obsolete("not implemented", true)]
        public static int get_num_threads() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.set_num_threads
        [Obsolete("not implemented", true)]
        public static void set_num_threads(int num) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.get_num_interop_threads
        [Pure, Obsolete("not implemented", true)]
        public static int get_num_interop_threads() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads
        [Obsolete("not implemented", true)]
        public static void set_num_interop_threads(int num) => throw new NotImplementedException();
    }
}