// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    public static partial class torch
    {
        public static partial class optim
        {

            /// <summary>
            /// Implements the RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSProp RMSProp(IEnumerable<Parameter> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                return new RMSProp(parameters, lr, alpha, eps, weight_decay, momentum, centered);
            }

            /// <summary>
            /// Implements the RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSProp RMSProp(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                return new RMSProp(parameters.Select(np => np.parameter), lr, alpha, eps, weight_decay, momentum, centered);
            }

            /// <summary>
            /// Implements the RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSProp RMSProp(IEnumerable<RMSProp.ParamGroup> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                return new RMSProp(parameters, lr, alpha, eps, weight_decay, momentum, centered);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        public class RMSProp : OptimizerHelper, IMomentum
        {

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="alpha">Smoothing constant.</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="centered">if ``True``, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public RMSProp(IEnumerable<Parameter> parameters, double lr = 1e-3, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0.0, bool centered = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, alpha, eps, weight_decay, momentum, centered)
            {
            }

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="alpha">Smoothing constant.</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="centered">if ``True``, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public RMSProp(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0.0, bool centered = false)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (alpha < 0) throw new ArgumentException($"Invalid alpha: {alpha}");
                if (momentum < 0.0) throw new ArgumentException($"Invalid momentum value: {momentum}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    eps = eps,
                    alpha = alpha,
                    momentum = momentum,
                    centered = centered,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            /// <summary>
            /// Performs a single optimization step (parameter update).
            /// </summary>
            /// <param name="closure">A closure that reevaluates the model and returns the loss. Optional for most optimizers.</param>
            /// <returns></returns>
            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var momentum = options.momentum.Value;
                    var alpha = options.alpha.Value;
                    var weight_decay = options.weight_decay.Value;
                    var centered = options.centered.Value;
                    var eps = options.eps.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = _state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        state.square_avg.mul_(alpha).addcmul_(grad, grad, value: 1 - alpha);

                        Tensor avg = null;

                        if (centered) {
                            var grad_avg = state.grad_avg;
                            grad_avg.mul_(alpha).add_(grad, alpha: 1 - alpha);
                            avg = state.square_avg.addcmul(grad_avg, grad_avg, value: -1).sqrt_().add_(eps);
                        } else {
                            avg = state.square_avg.sqrt().add_(eps);
                        }

                        if (momentum > 0) {
                            var buf = state.momentum_buffer;
                            buf.mul_(momentum).addcdiv_(grad, avg);
                            param.add_(buf, alpha: -lr);
                        } else {
                            param.addcdiv_(grad, avg, -lr);
                        }
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var kvp in _state) {
                    kvp.Value.Dispose();
                }
                _state.Clear();
            }

            private class State : IDisposable
            {
                public long step;
                public Tensor square_avg;
                public Tensor momentum_buffer;
                public Tensor grad_avg;

                public void Dispose()
                {
                    momentum_buffer.Dispose();
                    square_avg.Dispose();
                    grad_avg.Dispose();
                }
            }



            /// <summary>
            /// Add a param group to the Optimizer s param_groups.
            /// </summary>
            /// <param name="param_group"></param>
            /// <remarks>This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the Optimizer as training progresses.</remarks>
            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.momentum.HasValue) opt.momentum = def.momentum;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.alpha.HasValue) opt.alpha = def.alpha;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.centered.HasValue) opt.centered = def.centered;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.square_avg = torch.zeros_like(p);
                    state.grad_avg = torch.zeros_like(p);
                    state.momentum_buffer = torch.zeros_like(p);
                }
            }
            public class Options : Modules.OptimizerOptions
            {
                public double? momentum;
                public double? alpha;
                public double? eps;
                public double? weight_decay;
                public bool? centered;
            }

            public class ParamGroup : ParamGroup<Options>, IMomentum
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-3, double eps = 1e-8, double alpha = 0.99, double weight_decay = 0, double momentum = 0.0, bool centered = false)
                    : base(parameters, new RMSProp.Options { LearningRate = lr, eps = eps, alpha = alpha, weight_decay = weight_decay, momentum = momentum, centered = centered })
                {
                }

                public double Momentum { get => Options.momentum.Value; set => Options.momentum = value; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();

            public double Momentum { get => (_defaults as Options).momentum.Value; set => (_defaults as Options).momentum = value; }
        }
    }
}
