// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Diagnostics.Contracts;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#locally-disabling-gradient-computation
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.no_grad
        /// <summary>
        /// Context-manager that disables gradient calculation.
        /// </summary>
        /// <returns></returns>
        public static IDisposable no_grad() => new AutoGradMode(false);

        // https://pytorch.org/docs/stable/generated/torch.enable_grad
        /// <summary>
        /// Context-manager that enables gradient calculation.
        /// </summary>
        /// <returns></returns>
        public static IDisposable enable_grad(bool enabled = true) => new AutoGradMode(enabled);

        // https://pytorch.org/docs/stable/generated/torch.set_grad_enabled
        /// <summary>
        /// Context-manager that sets gradient calculation to on or off.
        /// </summary>
        /// <returns></returns>
        public static IDisposable set_grad_enabled(bool mode = true) => new AutoGradMode(mode);

        // https://pytorch.org/docs/stable/generated/torch.is_grad_enabled
        /// <summary>
        /// Returns true if grad mode is currently enabled.
        /// </summary>
        /// <returns></returns>
        [Pure]public static bool is_grad_enabled() => AutoGradMode.IsEnabled;

        // https://pytorch.org/docs/stable/generated/torch.inference_mode
        /// <summary>
        /// Context-manager that enables inference mode.
        /// </summary>
        /// <returns></returns>
        public static IDisposable inference_mode(bool mode = true) => new InferenceMode(mode);

        // https://pytorch.org/docs/stable/generated/torch.is_inference_mode_enabled
        /// <summary>
        /// Returns true if inference mode mode is currently enabled.
        /// </summary>
        /// <returns></returns>
        [Pure]public static bool is_inference_mode_enabled() => InferenceMode.IsEnabled;
    }
}