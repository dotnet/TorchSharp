// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    using static torch;

    internal class AutoGradMode : IDisposable
    {
        private bool _isPrevGrad;

        [DllImport("LibTorchSharp")]
        private static extern bool THSAutograd_isGradEnabled();

        [DllImport("LibTorchSharp")]
        private static extern void THSAutograd_setGrad(bool enabled);

        public AutoGradMode(bool enabled)
        {
            _isPrevGrad = THSAutograd_isGradEnabled();
            THSAutograd_setGrad(enabled);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            if (disposing)
            {
                THSAutograd_setGrad(_isPrevGrad);
            }
        }

        public static bool IsAutogradEnabled()
        {
            return THSAutograd_isGradEnabled();
        }
    }

    public static partial class torch
    {
        /// <summary>
        /// Context-manager that disables gradient calculation.
        /// </summary>
        /// <returns></returns>
        public static IDisposable no_grad() => new AutoGradMode(false);

        /// <summary>
        /// Context-manager that enables gradient calculation.
        /// </summary>
        /// <returns></returns>
        public static IDisposable enable_grad() => new AutoGradMode(true);

        /// <summary>
        /// Context-manager that sets gradient calculation to on or off.
        /// </summary>
        /// <returns></returns>
        public static IDisposable set_grad_enabled(bool mode) => new AutoGradMode(mode);

        /// <summary>
        /// Returns true if grad mode is currently enabled.
        /// </summary>
        /// <returns></returns>
        public static bool is_grad_enabled() => AutoGradMode.IsAutogradEnabled();

        public static partial class autograd
        {
            // Repeating some of what's already in the 'torch' module

            /// <summary>
            /// Context-manager that disables gradient calculation.
            /// </summary>
            /// <returns></returns>
            public static IDisposable no_grad() => new AutoGradMode(false);

            /// <summary>
            /// Context-manager that enables gradient calculation.
            /// </summary>
            /// <returns></returns>
            public static IDisposable enable_grad() => new AutoGradMode(true);

            /// <summary>
            /// Context-manager that sets gradient calculation to on or off.
            /// </summary>
            /// <returns></returns>
            public static IDisposable set_grad_enabled(bool mode) => new AutoGradMode(mode);

            /// <summary>
            /// Returns true if grad mode is currently enabled.
            /// </summary>
            /// <returns></returns>
            public static bool is_grad_enabled() => AutoGradMode.IsAutogradEnabled();


            [DllImport("LibTorchSharp")]
            private static extern void THSAutograd_grad(
             IntPtr outputs, long oLength,
             IntPtr inputs, long iLength,
             IntPtr grad_outs, long gLength,
             bool retain_graph, bool create_graph, bool allow_unused,
             AllocatePinnedArray allocator);

            public static IList<Tensor> grad(IList<Tensor> outputs, IList<Tensor> inputs, IList<Tensor> grad_outputs = null, bool retain_graph = false, bool create_graph = false, bool allow_unused = false)
            {
                IntPtr[] result;

                using (var outs = new PinnedArray<IntPtr>())
                using (var ins = new PinnedArray<IntPtr>())
                using (var grads = new PinnedArray<IntPtr>())
                using (var results = new PinnedArray<IntPtr>()) {

                    IntPtr outsRef = outs.CreateArray(outputs.Select(p => p.Handle).ToArray());
                    IntPtr insRef = ins.CreateArray(inputs.Select(p => p.Handle).ToArray());
                    IntPtr gradsRef = grad_outputs == null ? IntPtr.Zero : grads.CreateArray(grad_outputs.Select(p => p.Handle).ToArray());
                    long gradsLength = grad_outputs == null ? 0 : grads.Array.Length;

                    THSAutograd_grad(outsRef, outs.Array.Length, insRef, ins.Array.Length, gradsRef, gradsLength, retain_graph, create_graph, allow_unused, results.CreateArray);
                    torch.CheckForErrors();
                    result = results.Array;
                }

                return result.Select(x => new Tensor(x)).ToList();


            }
        }
    }
}
