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
            /// Implements the NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public static NAdam NAdam(IEnumerable<Parameter> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                return new NAdam(named_parameters, lr, beta1, beta2, eps, weight_decay, momentum_decay);
            }

            /// <summary>
            /// Implements the NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public static NAdam NAdam(IEnumerable<(string name, Parameter parameter)> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                return new NAdam(named_parameters.Select(np => np.parameter), lr, beta1, beta2, eps, weight_decay, momentum_decay);
            }

            /// <summary>
            /// Implements the NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public static NAdam NAdam(IEnumerable<NAdam.ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                return new NAdam(parameters, lr, beta1, beta2, eps, weight_decay, momentum_decay);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        public class NAdam : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public NAdam(IEnumerable<Parameter> parameters, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay, momentum_decay)
            {
            }

            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            /// <returns></returns>
            public NAdam(IEnumerable<ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (beta1 < 0.0 || beta1 > 1.0) throw new ArgumentException($"Invalid beta1 value: {beta1}");
                if (beta2 < 0.0 || beta2 > 1.0) throw new ArgumentException($"Invalid beta2 value: {beta2}");
                if (eps < 0.0) throw new ArgumentException($"Invalid eps value: {eps}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");
                if (momentum_decay < 0.0) throw new ArgumentException($"Invalid momentum_decay value: {momentum_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    eps = eps,
                    weight_decay = weight_decay,
                    momentum_decay = momentum_decay
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
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var eps = options.eps.Value;
                    var weight_decay = options.weight_decay.Value;
                    var momentum_decay = options.momentum_decay.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        var state = _state[param.handle];

                        state.step += 1;

                        var exp_avg = state.exp_avg;
                        var exp_avg_sq = state.exp_avg_sq;

                        var bias_correction2 = 1 - Math.Pow(beta2, state.step);

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        var mu = beta1 * (1.0 - 0.5 * Math.Pow(0.96, state.step * momentum_decay));
                        var mu_next = beta1 * (1.0 - 0.5 * Math.Pow(0.96, (state.step + 1) * momentum_decay));

                        var mu_product = state.mu_product * mu;
                        var mu_product_next = mu_product * mu * mu_next;

                        exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value: 1 - beta2);

                        var denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps);

                        param.addcdiv_(grad, denom, value: -lr * (1 - mu) / (1 - mu_product));
                        param.addcdiv_(exp_avg, denom, value: -lr * mu_next / (1 - mu_product_next));

                        state.mu_product = mu_product;
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var kvp in _state) {
                    kvp.Value.Dispose();
                }
            }

            private class State : IDisposable
            {
                public long step;
                public double mu_product;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;

                public void Dispose()
                {
                    exp_avg.Dispose();
                    exp_avg_sq.Dispose();
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
                if (!opt.beta1.HasValue) opt.beta1 = def.beta1;
                if (!opt.beta2.HasValue) opt.beta2 = def.beta2;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.momentum_decay.HasValue) opt.momentum_decay = def.momentum_decay;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? eps;
                public double? weight_decay;
                public double? momentum_decay;
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1.0, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
                    : base(parameters, new NAdam.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay, momentum_decay = momentum_decay })
                {
                }

                public (double, double) Betas {
                    get => (Options.beta1.Value, Options.beta2.Value);
                    set { Options.beta1 = value.Item1; Options.beta2 = value.Item2; }
                }
            }

            public (double, double) Betas {
                get => ((_defaults as Options).beta1.Value, (_defaults as Options).beta2.Value);
                set { (_defaults as Options).beta1 = value.Item1; (_defaults as Options).beta2 = value.Item2; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }
    }
}
