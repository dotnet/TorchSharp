// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    public static partial class torch
    {
        public static partial class optim
        {

            /// <summary>
            /// Implements the Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// http://jmlr.org/papers/v12/duchi11a.html
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public static Adagrad Adagrad(IEnumerable<Parameter> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                return new Adagrad(parameters, lr, lr_decay, weight_decay, initial_accumulator_value, eps);
            }

            /// <summary>
            /// Implements the Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// http://jmlr.org/papers/v12/duchi11a.html
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public static Adagrad Adagrad(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                return new Adagrad(parameters.Select(np => np.parameter), lr, lr_decay, weight_decay, initial_accumulator_value, eps);
            }

            /// <summary>
            /// Implements the Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// http://jmlr.org/papers/v12/duchi11a.html
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public static Adagrad Adagrad(IEnumerable<Adagrad.ParamGroup> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                return new Adagrad(parameters, lr, lr_decay, weight_decay, initial_accumulator_value, eps);
            }
        }
    }

    namespace Modules
    {
        public class Adagrad : OptimizerHelper
        {
            /// <summary>
            /// Implements Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public Adagrad(IEnumerable<Parameter> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
                : this(new ParamGroup[] { new() { Parameters = parameters } }, lr, lr_decay, weight_decay, initial_accumulator_value, eps)
            {
            }
            /// <summary>
            /// Implements Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public Adagrad(IEnumerable<ParamGroup> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (lr_decay < 0.0) throw new ArgumentException($"Invalid lr_decay value: {lr_decay}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");
                if (initial_accumulator_value < 0) throw new ArgumentException($"Invalid initial_accumulator_value: {initial_accumulator_value}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    lr_decay = lr_decay,
                    eps = eps,
                    initial_accumulator_value = initial_accumulator_value,
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
                    var lr_decay = options.lr_decay.Value;
                    var weight_decay = options.weight_decay.Value;
                    var eps = options.eps.Value;
                    var initial_accumulator_value = options.initial_accumulator_value.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = _state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        var clr = lr / (1 + (state.step - 1) * lr_decay);

                        if (grad.is_sparse)
                            throw new NotImplementedException("Adagrad optimization over sparse parameters");
                        if (torch.is_complex(grad))
                            throw new NotImplementedException("Adagrad optimization over complex parameters");

                        state.sum.addcmul_(grad, grad, value: 1);
                        var std = state.sum.sqrt().add_(eps);
                        param.addcdiv_(grad, std, value: -clr);
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
                public Tensor sum;

                public void Dispose()
                {
                    sum.Dispose();
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
                if (!opt.lr_decay.HasValue) opt.lr_decay = def.lr_decay;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.initial_accumulator_value.HasValue) opt.initial_accumulator_value = def.initial_accumulator_value;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    var init_value = torch.is_complex(p.dtype)
                        ? (Scalar)new System.Numerics.Complex((param_group.Options as Options).initial_accumulator_value.Value, (param_group.Options as Options).initial_accumulator_value.Value)
                        : (Scalar)(param_group.Options as Options).initial_accumulator_value.Value;
                    state.sum = torch.full_like(p, init_value);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? lr_decay;
                public double? initial_accumulator_value;
                public double? eps;
                public double? weight_decay;
            }

            public class ParamGroup : ParamGroup<Options>
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 0.01, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
                    : base(parameters, new Adagrad.Options { LearningRate = lr, lr_decay = lr_decay, initial_accumulator_value = initial_accumulator_value, weight_decay = weight_decay, eps = eps })
                {
                }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }
    }
}
