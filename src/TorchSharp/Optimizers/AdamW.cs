// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.IO;
    using Modules;

    public static partial class torch
    {
        public static partial class optim
        {

            /// <summary>
            /// Implements the AdamW algorithm.
            ///
            /// The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            ///  https://arxiv.org/abs/1711.05101
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static AdamW AdamW(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new AdamW(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
            }

            /// <summary>
            /// Implements the AdamW algorithm.
            ///
            /// The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            ///  https://arxiv.org/abs/1711.05101
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static AdamW AdamW(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new AdamW(parameters.Select(np => np.parameter), lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
            }

            /// <summary>
            /// Implements the AdamW algorithm.
            ///
            /// The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            ///  https://arxiv.org/abs/1711.05101
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static AdamW AdamW(IEnumerable<AdamW.ParamGroup> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new AdamW(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        public class AdamW : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements AdamW algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="beta1">First coefficient used for computing running averages of gradient and its square</param>
            /// <param name="beta2">Second coefficient used for computing running averages of gradient and its square</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing</param>
            /// <returns></returns>
            public AdamW(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize)
            {
            }

            /// <summary>
            /// Implements AdamW algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="beta1">First coefficient used for computing running averages of gradient and its square</param>
            /// <param name="beta2">Second coefficient used for computing running averages of gradient and its square</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing</param>
            /// <returns></returns>
            public AdamW(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (beta1 < 0 || beta1 >= 1.0) throw new ArgumentException($"Invalid beta1: {beta1}");
                if (beta2 < 0 || beta2 >= 1.0) throw new ArgumentException($"Invalid beta2: {beta2}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    maximize = maximize,
                    eps = eps,
                    amsgrad = amsgrad,
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
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var weight_decay = options.weight_decay.Value;
                    var amsgrad = options.amsgrad.Value;
                    var maximize = options.maximize.Value;
                    var eps = options.eps.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = (State)_state[param.handle];

                        var grad = (maximize) ? -param.grad() : param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        param.mul_(1 - lr * weight_decay);

                        var bias_correction1 = 1 - Math.Pow(beta1, state.step);
                        var bias_correction2 = 1 - Math.Pow(beta2, state.step);

                        state.exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);
                        state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value: 1 - beta2);

                        Tensor denom = null;
                        if (amsgrad) {
                            var t0 = state.max_exp_avg_sq;
                            state.max_exp_avg_sq = torch.maximum(t0, state.exp_avg_sq).DetachFromDisposeScope();
                            t0.Dispose();
                            denom = (state.max_exp_avg_sq.sqrt() / Math.Sqrt(bias_correction2)).add_(eps);
                        } else {
                            denom = (state.exp_avg_sq.sqrt() / Math.Sqrt(bias_correction2)).add_(eps);
                        }

                        var step_size = lr / bias_correction1;
                        param.addcdiv_(state.exp_avg, denom, value: -step_size);
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var kvp in _state) {
                    ((State)kvp.Value).Dispose();
                }
            }

            public class State : OptimizerState, IDisposable
            {
                public long step;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
                public Tensor max_exp_avg_sq;

                public void Dispose()
                {
                    exp_avg.Dispose();
                    exp_avg_sq.Dispose();
                    if (max_exp_avg_sq is not null) {
                        max_exp_avg_sq.Dispose();
                    }
                }

                /// <summary>
                /// Move all the state to the indicated device.
                /// </summary>
                /// <param name="device">The device to move all state to.</param>
                public override void to(Device device)
                {
                    exp_avg.to(device);
                    exp_avg_sq.to(device);
                    if (max_exp_avg_sq is not null) {
                        max_exp_avg_sq.to(device);
                    }
                }

                /// <summary>
                /// Load the optimizer parameter state from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    step = reader.ReadInt64();
                    exp_avg.Load(reader);
                    exp_avg_sq.Load(reader);
                    var hasMax = reader.ReadBoolean();
                    if (hasMax) {
                        if (max_exp_avg_sq is null) {
                            max_exp_avg_sq = torch.zeros_like(exp_avg_sq);
                        }
                        max_exp_avg_sq.Load(reader);
                    } else {
                        if (max_exp_avg_sq is not null)
                            max_exp_avg_sq.Dispose();
                        max_exp_avg_sq = null;
                    }
                }

                /// <summary>
                /// Save the optimizer parameter state to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    writer.Write(step);
                    exp_avg.Save(writer);
                    exp_avg_sq.Save(writer);
                    if (max_exp_avg_sq is not null) {
                        writer.Write(true);
                        max_exp_avg_sq.Save(writer);
                    } else {
                        writer.Write(false);
                    }
                }

                /// <summary>
                /// Load optimizer parameter state from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer state record.</param>
                public override void LoadStateDict(OptimizerState source)
                {
                    var st_state = source as State;
                    exp_avg.Dispose();
                    exp_avg_sq.Dispose();
                    if (max_exp_avg_sq is not null) {
                        max_exp_avg_sq.Dispose();
                    }

                    step = st_state.step;
                    exp_avg = st_state.exp_avg;
                    exp_avg_sq = st_state.exp_avg_sq;
                    max_exp_avg_sq = st_state.max_exp_avg_sq;
                }

                public override bool ApproximatelyEquals(OptimizerState other)
                {
                    var rhs = other as State;
                    return (rhs is not null) && step == rhs.step &&
                        exp_avg.allclose(rhs.exp_avg) &&
                        exp_avg_sq.allclose(rhs.exp_avg_sq) &&
                        (max_exp_avg_sq is null || max_exp_avg_sq.allclose(rhs.max_exp_avg_sq));
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
                if (!opt.amsgrad.HasValue) opt.amsgrad = def.amsgrad;
                if (!opt.maximize.HasValue) opt.maximize = def.maximize;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p).DetachFromDisposeScope();
                    state.exp_avg_sq = torch.zeros_like(p).DetachFromDisposeScope();
                    if (opt.amsgrad.Value) {
                        state.max_exp_avg_sq = torch.zeros_like(p).DetachFromDisposeScope();
                    }
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? weight_decay;
                public double? eps;
                public bool? amsgrad;
                public bool? maximize;

                /// <summary>
                /// Load optimizer options (param-group hyperparameters) from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer options record.</param>
                public override void LoadStateDict(OptimizerOptions source)
                {
                    base.LoadStateDict(source);
                    var opts = source as Options;
                    beta1 = opts.beta1;
                    beta2 = opts.beta2;
                    eps = opts.eps;
                    weight_decay = opts.weight_decay;
                    amsgrad = opts.amsgrad;
                    maximize = opts.maximize;
                }

                /// <summary>
                /// Load the optimizer options (param-group hyperparameters) from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    base.LoadStateDict(reader);
                    beta1 = reader.ReadDouble();
                    beta2 = reader.ReadDouble();
                    eps = reader.ReadDouble();
                    weight_decay = reader.ReadDouble();
                    amsgrad = reader.ReadBoolean();
                    maximize = reader.ReadBoolean();
                }

                /// <summary>
                /// Save the optimizer options (param-group hyperparameters) to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    base.SaveStateDict(writer);
                    writer.Write(beta1.Value);
                    writer.Write(beta2.Value);
                    writer.Write(eps.Value);
                    writer.Write(weight_decay.Value);
                    writer.Write(amsgrad.Value);
                    writer.Write(maximize.Value);
                }
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
                    : base(parameters, new AdamW.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad, maximize = maximize })
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
        }
    }
}
