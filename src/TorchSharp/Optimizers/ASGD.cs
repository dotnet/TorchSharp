// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
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
            /// Implements the Averaged Stochastic Gradient Descent.
            ///
            /// It has been proposed in Acceleration of stochastic approximation by averaging.
            /// https://dl.acm.org/citation.cfm?id=131098
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public static ASGD ASGD(IEnumerable<Parameter> parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0, bool maximize = false)
            {
                return new Modules.ASGD(parameters, lr, lambd, alpha, t0, weight_decay, maximize);
            }

            /// <summary>
            /// Implements the Averaged Stochastic Gradient Descent.
            ///
            /// It has been proposed in Acceleration of stochastic approximation by averaging.
            /// https://dl.acm.org/citation.cfm?id=131098
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public static ASGD ASGD(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0, bool maximize = false)
            {
                return new Modules.ASGD(parameters.Select(np => np.parameter), lr, lambd, alpha, t0, weight_decay, maximize);
            }

            /// <summary>
            /// Implements the Averaged Stochastic Gradient Descent.
            ///
            /// It has been proposed in Acceleration of stochastic approximation by averaging.
            /// https://dl.acm.org/citation.cfm?id=131098
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public static ASGD ASGD(IEnumerable<ParamGroup<ASGD.Options>> parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0, bool maximize = false)
            {
                return new Modules.ASGD(parameters, lr, lambd, alpha, t0, weight_decay, maximize);
            }
        }
    }

    namespace Modules
    {
        public class ASGD : OptimizerHelper
        {
            /// <summary>
            /// Implements ASGD algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public ASGD(IEnumerable<Parameter> parameters, double lr = 0.01, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0, bool maximize = false)
                : this(new ParamGroup[] { new() { Parameters = parameters } }, lr, lambd, alpha, t0, weight_decay, maximize)
            {
            }

            /// <summary>
            /// Implements ASGD algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            /// <returns></returns>
            public ASGD(IEnumerable<ParamGroup<Options>> parameters, double lr = 0.01, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0, bool maximize = false)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    maximize = maximize,
                    lambd = lambd,
                    alpha = alpha,
                    t0 = t0,
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
                    var maximize = options.maximize.Value;
                    var lambd = options.lambd.Value;
                    var alpha = options.alpha.Value;
                    var weight_decay = options.weight_decay.Value;
                    var t0 = options.t0.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (grad.is_sparse) throw new ArgumentException("ASGD does not support sparse gradients");

                        if (maximize) grad = -grad;

                        var state = (State)_state[param.handle];

                        state.step += 1;

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        param.mul_(1 - lambd * state.eta);
                        param.add_(grad, alpha: -state.eta);

                        if (state.mu != 1) {
                            state.ax.add_(param.sub(state.ax).mul(state.mu));
                        } else {
                            state.ax.copy_(param);
                        }

                        state.eta = lr / Math.Pow((1 + lambd * lr * state.step), alpha);
                        state.mu = 1 / Math.Max(1, state.step - t0);
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var kvp in _state) {
                    ((State)kvp.Item2).Dispose();
                }
                _state.Clear();
            }

            public class State : OptimizerState, IDisposable
            {
                public long step;
                public double eta;
                public double mu;
                public Tensor ax;

                public void Dispose()
                {
                    ax.Dispose();
                }

                /// <summary>
                /// Move all the state to the indicated device.
                /// </summary>
                /// <param name="device">The device to move all state to.</param>
                public override void to(Device device)
                {
                    ax.to(device);
                }

                /// <summary>
                /// Load the optimizer parameter state from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    step = reader.ReadInt64();
                    eta = reader.ReadDouble();
                    mu = reader.ReadDouble();
                    ax.Load(reader);
                }

                /// <summary>
                /// Load optimizer parameter state from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer state record.</param>
                public override void LoadStateDict(OptimizerState source)
                {
                    var st_state = source as State;
                    step = st_state.step;
                    eta = st_state.eta;
                    mu = st_state.mu;
                    ax.Dispose();
                    ax = st_state.ax;
                }

                /// <summary>
                /// Save the optimizer parameter state to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    writer.Write(step);
                    writer.Write(eta);
                    writer.Write(mu);
                    ax.Save(writer);
                }

                /// <summary>
                /// Useful for tests, allows comparison of one state with another.
                /// </summary>
                /// <param name="other">The other optimizer state</param>
                /// <returns></returns>
                public override bool ApproximatelyEquals(OptimizerState other)
                {
                    var rhs = other as State;
                    return (rhs is not null) && step == rhs.step && eta == rhs.eta && mu == rhs.mu && ax.allclose(rhs.ax);
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
                if (!opt.maximize.HasValue) opt.maximize = def.maximize;
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.lambd.HasValue) opt.lambd = def.lambd;
                if (!opt.alpha.HasValue) opt.alpha = def.alpha;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.t0.HasValue) opt.t0 = def.t0;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.eta = param_group.LearningRate;
                    state.mu = 1;
                    state.ax = torch.zeros_like(p).DetachFromDisposeScope();
                }
            }

            public class Options : OptimizerOptions
            {
                public bool? maximize;
                public double? lambd;
                public double? alpha;
                public double? weight_decay;
                public double? t0;

                /// <summary>
                /// Load optimizer options (param-group hyperparameters) from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer options record.</param>
                public override void LoadStateDict(OptimizerOptions source)
                {
                    base.LoadStateDict(source);
                    var opts = source as Options;
                    maximize = opts.maximize;
                    lambd = opts.lambd;
                    alpha = opts.alpha;
                    weight_decay = opts.weight_decay;
                    t0 = opts.t0;
                }

                /// <summary>
                /// Load the optimizer options (param-group hyperparameters) from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    base.LoadStateDict(reader);
                    maximize = reader.ReadBoolean();
                    lambd = reader.ReadDouble();
                    alpha = reader.ReadDouble();
                    weight_decay = reader.ReadDouble();
                    t0 = reader.ReadDouble();
                }

                /// <summary>
                /// Save the optimizer options (param-group hyperparameters) to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    base.SaveStateDict(writer);
                    writer.Write(maximize.Value);
                    writer.Write(lambd.Value);
                    writer.Write(alpha.Value);
                    writer.Write(weight_decay.Value);
                    writer.Write(t0.Value);
                }
            }

            public class ParamGroup : ParamGroup<Options>
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 0.01, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
                    : base(parameters, new ASGD.Options { LearningRate = lr, lambd = lambd, alpha = alpha, t0 = t0, weight_decay = weight_decay })
                {
                }
            }
        }
    }
}
