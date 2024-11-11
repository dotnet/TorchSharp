// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Diagnostics.Contracts;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#utilities
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.compiled_with_cxx11_abi
        [Pure, Obsolete("not implemented", true)]
        public static bool compiled_with_cxx11_abi() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.result_type
        public static ScalarType result_type(Tensor tensor1, Tensor tensor2)
        {
            var res = THSTensor_result_type(tensor1.Handle, tensor2.Handle);
            if (res == -1) CheckForErrors();
            return (ScalarType)res;
        }

        // https://pytorch.org/docs/stable/generated/torch.can_cast
        public static bool can_cast(ScalarType from, ScalarType to)
        {
            var res = THSTorch_can_cast((int)from, (int)to);
            if (res == -1) CheckForErrors();
            return res != 0;
        }

        // https://pytorch.org/docs/stable/generated/torch.promote_types
        public static ScalarType promote_types(ScalarType type1, ScalarType type2)
        {
            var res = THSTorch_promote_types((int)type1, (int)type2);
            if (res == -1) CheckForErrors();
            return (ScalarType)res;
        }

        // https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms
        [Obsolete("not implemented", true)]
        public static IDisposable use_deterministic_algorithms(bool mode = true, bool warn_only = false) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled
        [Pure, Obsolete("not implemented", true)]
        public static bool are_deterministic_algorithms_enabled() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.is_deterministic_algorithms_warn_only_enabled
        [Pure, Obsolete("not implemented", true)]
        public static bool is_deterministic_algorithms_warn_only_enabled() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode
        [Obsolete("not implemented", true)]
        public static IDisposable set_deterministic_debug_mode(DebugMode mode) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.get_deterministic_debug_mode
        [Pure, Obsolete("not implemented", true)]
        public static DebugMode get_deterministic_debug_mode() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision
        [Obsolete("not implemented", true)]
        public static IDisposable set_float32_matmul_precision(Float32MatmulPrecision precision) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision
        [Pure, Obsolete("not implemented", true)]
        public static Float32MatmulPrecision get_float32_matmul_precision() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.set_warn_always
        [Obsolete("not implemented", true)]
        public static IDisposable set_warn_always(bool mode = true) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.is_warn_always_enabled
        [Pure, Obsolete("not implemented", true)]
        public static bool is_warn_always_enabled() => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch._assert
        [Obsolete("not implemented", true)]
        public static void _assert(bool condition, string message) => throw new NotImplementedException();

        [Obsolete("not implemented", true)]
        public static void _assert(Func<bool> condition, string message) => throw new NotImplementedException();
    }
}