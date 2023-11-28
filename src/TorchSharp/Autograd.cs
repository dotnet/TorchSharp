// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    /// <summary>
    /// Helper class, relying on IDisposable to implement block-based scoping of autograd settings.
    /// </summary>
    internal class AutoGradMode : IDisposable
    {
        private readonly bool _isPrevGrad;

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

        public static bool IsEnabled { get => THSAutograd_isGradEnabled(); }
    }

    /// <summary>
    /// Helper class, relying on IDisposable to implement block-based scoping of autograd settings.
    /// </summary>
    internal class InferenceMode : IDisposable
    {
        private IntPtr _guard;

        public InferenceMode(bool mode)
        {
            _guard = THSAutograd_getInferenceModeGuard(mode);
        }

        /// <summary>
        /// Finalize the inference mode. Releases the guard.
        /// </summary>
        ~InferenceMode() => Dispose(false);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            if (_guard != IntPtr.Zero) {
                THSAutograd_deleteInferenceModeGuard(_guard);
                _guard = IntPtr.Zero;
            }
        }

        public static bool IsEnabled { get => THSAutograd_isInferenceModeEnabled(); }
    }

    /// <summary>
    /// Helper class, relying on IDisposable to implement block-based scoping of anomaly settings.
    /// </summary>
    public class AnomalyMode : IDisposable
    {
        private readonly bool _isPrevGrad;
        private readonly bool _shouldCheckNaN;

        public AnomalyMode(bool enabled, bool check_nan = true)
        {
            _isPrevGrad = THSAutograd_isAnomalyEnabled();
            _shouldCheckNaN = THSAutograd_shouldCheckNaN();
            THSAutograd_setAnomaly(enabled, check_nan);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing) {
                THSAutograd_setAnomaly(_isPrevGrad, _shouldCheckNaN);
            }
        }

        public static bool IsEnabled { get => THSAutograd_isAnomalyEnabled(); }

        public static bool ShouldCheckNaN { get => THSAutograd_shouldCheckNaN(); }
    }

    public static partial class torch
    {
        public static partial class autograd
        {
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
                using var outs = new PinnedArray<IntPtr>();
                using var ins = new PinnedArray<IntPtr>();
                using var grads = new PinnedArray<IntPtr>();
                using var results = new PinnedArray<IntPtr>();

                IntPtr outsRef = outs.CreateArray(outputs.Select(p => p.Handle).ToArray());
                IntPtr insRef = ins.CreateArray(inputs.Select(p => p.Handle).ToArray());
                IntPtr gradsRef = grad_outputs == null ? IntPtr.Zero : grads.CreateArray(grad_outputs.Select(p => p.Handle).ToArray());
                long gradsLength = grad_outputs == null ? 0 : grads.Array.Length;

                THSAutograd_grad(outsRef, outs.Array.Length, insRef, ins.Array.Length, gradsRef, gradsLength, retain_graph, create_graph, allow_unused, results.CreateArray);
                CheckForErrors();
                return results.Array.Select(x => new Tensor(x)).ToList();
            }

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
                bool rt = retain_graph ?? create_graph;

                using var ts = new PinnedArray<IntPtr>();
                using var gts = new PinnedArray<IntPtr>();
                using var ins = new PinnedArray<IntPtr>();
                IntPtr tensRef = ts.CreateArray(tensors.Select(p => p.Handle).ToArray());
                IntPtr gradsRef = grad_tensors == null ? IntPtr.Zero : gts.CreateArray(grad_tensors.Select(p => p.Handle).ToArray());
                IntPtr insRef = inputs == null ? IntPtr.Zero : ins.CreateArray(inputs.Select(p => p.Handle).ToArray());
                long insLength = inputs == null ? 0 : ins.Array.Length;
                long gradsLength = grad_tensors == null ? 0 : gts.Array.Length;

                THSAutograd_backward(tensRef, ts.Array.Length, gradsRef, gradsLength, rt, create_graph, insRef, insLength);
                CheckForErrors();
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

            /// <summary>
            /// Context-manager that enable anomaly detection for the autograd engine.
            /// </summary>
            /// <param name="check_nan">Flag whether to raise an error when the backward generate “nan”</param>
            /// <returns></returns>
            public static AnomalyMode detect_anomaly(bool check_nan = true)
            {
                return new AnomalyMode(true, check_nan);
            }

            /// <summary>
            /// Context-manager that enable anomaly detection for the autograd engine.
            /// </summary>
            /// <param name="mode">Flag whether to enable anomaly detection (true), or disable (false)</param>
            /// <param name="check_nan">Flag whether to raise an error when the backward generate “nan”</param>
            /// <returns></returns>
            public static AnomalyMode set_detect_anomaly(bool mode, bool check_nan = true)
            {
                return new AnomalyMode(mode, check_nan);
            }
        }
    }
}
