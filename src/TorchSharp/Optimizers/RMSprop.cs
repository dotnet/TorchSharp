// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;


namespace TorchSharp
{
    using System.Data;
    using System.IO;
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
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public static RMSProp RMSProp(IEnumerable<Parameter> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false, bool maximize = false)
            {
                return new RMSProp(parameters, lr, alpha, eps, weight_decay, momentum, centered, maximize);
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
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public static RMSProp RMSProp(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false, bool maximize = false)
            {
                return new RMSProp(parameters.Select(np => np.parameter), lr, alpha, eps, weight_decay, momentum, centered, maximize);
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
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public static RMSProp RMSProp(IEnumerable<RMSProp.ParamGroup> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false, bool maximize = false)
            {
                return new RMSProp(parameters, lr, alpha, eps, weight_decay, momentum, centered, maximize);
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
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public RMSProp(IEnumerable<Parameter> parameters, double lr = 1e-3, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0.0, bool centered = false, bool maximize = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, alpha, eps, weight_decay, momentum, centered, maximize)
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
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing.</param>
            public RMSProp(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0.0, bool centered = false, bool maximize = false)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (alpha < 0) throw new ArgumentException($"Invalid alpha: {alpha}");
                if (momentum < 0.0) throw new ArgumentException($"Invalid momentum value: {momentum}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    maximize = maximize,
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
                    var maximize = options.maximize.Value;
                    var momentum = options.momentum.Value;
                    var alpha = options.alpha.Value;
                    var weight_decay = options.weight_decay.Value;
                    var centered = options.centered.Value;
                    var eps = options.eps.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = (State)_state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (maximize) grad = -grad;

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
                    ((State)kvp.Item2).Dispose();
                }
                _state.Clear();
            }

            public class State : OptimizerState, IDisposable
            {
                public long step;
                public Tensor square_avg;
                public Tensor momentum_buffer;
                public Tensor grad_avg;

                public void Dispose()
                {
                    momentum_buffer?.Dispose();
                    square_avg.Dispose();
                    grad_avg?.Dispose();
                }

                /// <summary>
                /// Move all the state to the indicated device.
                /// </summary>
                /// <param name="device">The device to move all state to.</param>
                public override void to(Device device)
                {
                    square_avg.to(device);
                    momentum_buffer?.to(device);
                    grad_avg?.to(device);
                }

                /// <summary>
                /// Load the optimizer parameter state from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    step = reader.ReadInt64();
                    square_avg.Load(reader);
                    LoadConditionalStateTensor(reader, ref momentum_buffer);
                    LoadConditionalStateTensor(reader, ref grad_avg);
                }
                /// <summary>
                /// Save the optimizer parameter state to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    writer.Write(this.step);
                    square_avg.Save(writer);
                    SaveConditionalStateTensor(writer, momentum_buffer);
                    SaveConditionalStateTensor(writer, grad_avg);
                }

                /// <summary>
                /// Load optimizer parameter state from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer state record.</param>
                public override void LoadStateDict(OptimizerState source)
                {
                    var st_state = source as State;
                    square_avg.Dispose();
                    grad_avg.Dispose();
                    momentum_buffer.Dispose();

                    step = st_state.step;
                    square_avg = st_state.square_avg;
                    grad_avg = st_state.grad_avg;
                    momentum_buffer = st_state.momentum_buffer;
                }

                /// <summary>
                /// Useful for tests, allows comparison of one state with another.
                /// </summary>
                /// <param name="other">The other optimizer state</param>
                /// <returns></returns>
                public override bool ApproximatelyEquals(OptimizerState other)
                {
                    var rhs = other as State;
                    return (rhs is not null) && step == rhs.step &&
                        square_avg.allclose(rhs.square_avg) &&
                        grad_avg.allclose(rhs.grad_avg) &&
                        momentum_buffer.allclose(rhs.momentum_buffer);
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
                    state.square_avg = torch.zeros_like(p).DetachFromDisposeScope();
                    state.grad_avg = torch.zeros_like(p).DetachFromDisposeScope();
                    state.momentum_buffer = torch.zeros_like(p).DetachFromDisposeScope();
                }
            }
            public class Options : Modules.OptimizerOptions
            {
                public bool? maximize;
                public double? momentum;
                public double? alpha;
                public double? eps;
                public double? weight_decay;
                public bool? centered;

                /// <summary>
                /// Load optimizer options (param-group hyperparameters) from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer options record.</param>
                public override void LoadStateDict(OptimizerOptions source)
                {
                    base.LoadStateDict(source);
                    var opts = source as Options;
                    maximize = opts.maximize;
                    momentum = opts.momentum;
                    alpha = opts.alpha;
                    weight_decay = opts.weight_decay;
                    eps = opts.eps;
                    centered = opts.centered;
                }

                /// <summary>
                /// Load the optimizer options (param-group hyperparameters) from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    base.LoadStateDict(reader);
                    maximize = reader.ReadBoolean();
                    momentum = reader.ReadDouble();
                    alpha = reader.ReadDouble();
                    eps = reader.ReadDouble();
                    weight_decay = reader.ReadDouble();
                    centered = reader.ReadBoolean();
                }

                /// <summary>
                /// Save the optimizer options (param-group hyperparameters) to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    base.SaveStateDict(writer);
                    writer.Write(maximize.Value);
                    writer.Write(momentum.Value);
                    writer.Write(alpha.Value);
                    writer.Write(eps.Value);
                    writer.Write(weight_decay.Value);
                    writer.Write(centered.Value);
                }
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

            public double Momentum { get => (_defaults as Options).momentum.Value; set => (_defaults as Options).momentum = value; }
        }
    }
}
