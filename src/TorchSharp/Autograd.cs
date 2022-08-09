// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    /// <summary>
    /// Helper class, relying on IDisposable to implement block-based scoping of autograd settings.
    /// </summary>
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
             [MarshalAs(UnmanagedType.U1)] bool retain_graph,
             [MarshalAs(UnmanagedType.U1)] bool create_graph,
             [MarshalAs(UnmanagedType.U1)] bool allow_unused,
             AllocatePinnedArray allocator);

            /// <summary>
            /// Computes and returns the sum of gradients of outputs with respect to the inputs.
            /// </summary>
            /// <param name="outputs">Outputs of the differentiated function.</param>
            /// <param name="inputs">Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad)..</param>
            /// <param name="grad_outputs">
            /// The “vector” in the Jacobian-vector product. Usually gradients w.r.t. each output.
            /// Null values can be specified for scalar Tensors or ones that don’t require grad.
            /// If a null value would be acceptable for all grad_tensors, then this argument is optional.
            /// </param>
            /// <param name="retain_graph">
            /// If false, the graph used to compute the grad will be freed.
            /// Note that in nearly all cases setting this option to true is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.
            /// </param>
            /// <param name="create_graph">
            ///  If true, graph of the derivative will be constructed, allowing to compute higher order derivative products.
            ///  </param>
            /// <param name="allow_unused">
            /// If false, specifying inputs that were not used when computing outputs (and therefore their grad is always zero) is an error.
            /// </param>
            /// <returns></returns>
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

            [DllImport("LibTorchSharp")]
            private static extern void THSAutograd_backward(
                 IntPtr tensors, long tLength,
                 IntPtr grad_tensors, long gtLength,
                 [MarshalAs(UnmanagedType.U1)] bool retain_graph,
                 [MarshalAs(UnmanagedType.U1)] bool create_graph,
                 IntPtr inputs, long iLength);

            /// <summary>
            /// Computes the sum of gradients of given tensors with respect to graph leaves.
            /// </summary>
            /// <param name="tensors">Tensors of which the derivative will be computed.</param>
            /// <param name="grad_tensors">
            /// The “vector” in the Jacobian-vector product, usually gradients w.r.t. each element of corresponding tensors.
            /// Null values can be specified for scalar Tensors or ones that don’t require grad.
            /// If a null value would be acceptable for all grad_tensors, then this argument is optional.
            /// </param>
            /// <param name="retain_graph">If false, the graph used to compute the grad will be freed.
            /// Note that in nearly all cases setting this option to true is not needed and often can be worked around in a much more efficient way.
            /// Defaults to the value of create_graph.</param>
            /// <param name="create_graph">If true, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to false.</param>
            /// <param name="inputs">
            /// Inputs w.r.t. which the gradient be will accumulated into .grad. All other Tensors will be ignored.
            /// If not provided, the gradient is accumulated into all the leaf Tensors that were used to compute the attr::tensors.
            /// </param>
            /// <remarks>
            /// The graph is differentiated using the chain rule. If any of tensors are non-scalar (i.e. their data has more than one element) and require gradient,
            /// then the Jacobian-vector product would be computed, in this case the function additionally requires specifying grad_tensors.
            /// 
            /// It should be a sequence of matching length, that contains the “vector” in the Jacobian-vector product, usually the gradient of the differentiated
            /// function w.r.t. corresponding tensors (null is an acceptable value for all tensors that don’t need gradient tensors).
            /// 
            /// This function accumulates gradients in the leaves - you might need to zero the .grad properties or set them to null before calling it.
            /// </remarks>
            public static void backward(IList<Tensor> tensors, IList<Tensor> grad_tensors = null, bool? retain_graph = null, bool create_graph = false, IList<Tensor> inputs = null)
            {
                bool rt = retain_graph.HasValue ? retain_graph.Value : create_graph;

                using (var ts = new PinnedArray<IntPtr>())
                using (var gts = new PinnedArray<IntPtr>())
                using (var ins = new PinnedArray<IntPtr>()) {

                    IntPtr tensRef = ts.CreateArray(tensors.Select(p => p.Handle).ToArray());
                    IntPtr gradsRef = grad_tensors == null ? IntPtr.Zero : gts.CreateArray(grad_tensors.Select(p => p.Handle).ToArray());
                    IntPtr insRef = inputs == null ? IntPtr.Zero : ins.CreateArray(inputs.Select(p => p.Handle).ToArray());
                    long insLength = inputs == null ? 0 : ins.Array.Length;
                    long gradsLength = grad_tensors == null ? 0 : gts.Array.Length;

                    THSAutograd_backward(tensRef, ts.Array.Length, gradsRef, gradsLength, rt, create_graph, insRef, insLength);
                    torch.CheckForErrors();
                }
            }

            /// <summary>
            /// Computes the sum of gradients of given tensors with respect to graph leaves.
            /// </summary>
            /// <param name="tensor">Tensor of which the derivative will be computed.</param>
            /// <param name="grad_tensors">
            /// The “vector” in the Jacobian-vector product, usually gradients w.r.t. each element of corresponding tensors.
            /// Null values can be specified for scalar Tensors or ones that don’t require grad.
            /// If a null value would be acceptable for all grad_tensors, then this argument is optional.
            /// </param>
            /// <param name="retain_graph">If false, the graph used to compute the grad will be freed.
            /// Note that in nearly all cases setting this option to true is not needed and often can be worked around in a much more efficient way.
            /// Defaults to the value of create_graph.</param>
            /// <param name="create_graph">If true, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to false.</param>
            /// <param name="inputs">
            /// Inputs w.r.t. which the gradient be will accumulated into .grad. All other Tensors will be ignored.
            /// If not provided, the gradient is accumulated into all the leaf Tensors that were used to compute the attr::tensors.
            /// </param>
            /// <remarks>
            /// The graph is differentiated using the chain rule. If any of tensors are non-scalar (i.e. their data has more than one element) and require gradient,
            /// then the Jacobian-vector product would be computed, in this case the function additionally requires specifying grad_tensors.
            /// 
            /// It should be a sequence of matching length, that contains the “vector” in the Jacobian-vector product, usually the gradient of the differentiated
            /// function w.r.t. corresponding tensors (null is an acceptable value for all tensors that don’t need gradient tensors).
            /// 
            /// This function accumulates gradients in the leaves - you might need to zero the .grad properties or set them to null before calling it.
            /// </remarks>
            public static void backward(Tensor tensor, IList<Tensor> grad_tensors = null, bool? retain_graph = null, bool create_graph = false, IList<Tensor> inputs = null)
            {
                backward(new[] { tensor }, grad_tensors, retain_graph, create_graph, inputs);
            }

            /// <summary>
            /// Computes the sum of gradients of given tensors with respect to graph leaves.
            /// </summary>
            /// <param name="tensor">Tensor of which the derivative will be computed.</param>
            /// <param name="grad_tensor">
            /// The “vector” in the Jacobian-vector product, usually gradients w.r.t. each element of corresponding tensors.
            /// Null values can be specified for scalar Tensors or ones that don’t require grad.
            /// If a null value would be acceptable for all grad_tensors, then this argument is optional.
            /// </param>
            /// <param name="retain_graph">If false, the graph used to compute the grad will be freed.
            /// Note that in nearly all cases setting this option to true is not needed and often can be worked around in a much more efficient way.
            /// Defaults to the value of create_graph.</param>
            /// <param name="create_graph">If true, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to false.</param>
            /// <param name="inputs">
            /// Inputs w.r.t. which the gradient be will accumulated into .grad. All other Tensors will be ignored.
            /// If not provided, the gradient is accumulated into all the leaf Tensors that were used to compute the attr::tensors.
            /// </param>
            /// <remarks>
            /// The graph is differentiated using the chain rule. If any of tensors are non-scalar (i.e. their data has more than one element) and require gradient,
            /// then the Jacobian-vector product would be computed, in this case the function additionally requires specifying grad_tensors.
            /// 
            /// It should be a sequence of matching length, that contains the “vector” in the Jacobian-vector product, usually the gradient of the differentiated
            /// function w.r.t. corresponding tensors (null is an acceptable value for all tensors that don’t need gradient tensors).
            /// 
            /// This function accumulates gradients in the leaves - you might need to zero the .grad properties or set them to null before calling it.
            /// </remarks>
            public static void backward(Tensor tensor, Tensor grad_tensor, bool? retain_graph = null, bool create_graph = false, IList<Tensor> inputs = null)
            {
                backward(new[] { tensor }, new[] { grad_tensor }, retain_graph, create_graph, inputs);
            }

        }
    }
}
