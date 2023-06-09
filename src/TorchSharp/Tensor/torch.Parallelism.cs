// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Diagnostics.Contracts;

using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#parallelism
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.get_num_threads
        /// <summary>
        /// Returns the number of threads used for parallelizing CPU operations
        /// </summary>
        public static int get_num_threads()
        {
            var res = THSTorch_get_num_threads();
            if (res == -1) CheckForErrors();
            return res;
        }

        // https://pytorch.org/docs/stable/generated/torch.set_num_threads
        /// <summary>
        /// Sets the number of threads used for parallelizing CPU operations
        /// </summary>
        /// <param name="num">The number of threads to use.</param>
        public static void set_num_threads(int num)
        {
            THSTorch_set_num_threads(num);
            CheckForErrors();
        }

        // https://pytorch.org/docs/stable/generated/torch.get_num_interop_threads
        /// <summary>
        /// Returns the number of threads used for inter-op parallelism on CPU (e.g. in JIT interpreter)
        /// </summary>
        public static int get_num_interop_threads()
        {
            var res = THSTorch_get_num_interop_threads();
            if (res == -1) CheckForErrors();
            return res;
        }

        // https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads
        /// <summary>
        /// Sets the number of threads used for inter-op parallelism on CPU (e.g. in JIT interpreter)
        /// </summary>
        /// <param name="num">The number of threads to use.</param>
        public static void set_num_interop_threads(int num)
        {
            THSTorch_set_num_interop_threads(num);
            CheckForErrors();
        }
    }
}