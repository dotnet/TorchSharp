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
            /// Implements the stochastic gradient descent (optionally with momentum).
            ///
            /// The use of momentum is covered in:
            /// http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="learningRate">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static Modules.SGD SGD(IEnumerable<Parameter> parameters, double learningRate, double momentum = 0, double dampening = 0, double weight_decay = 0, bool nesterov = false, bool maximize = false)
            {
                return new Modules.SGD(parameters, learningRate, momentum, dampening, weight_decay, nesterov, maximize);
            }

            /// <summary>
            /// Implements the stochastic gradient descent (optionally with momentum).
            ///
            /// The use of momentum is covered in:
            /// http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="learningRate">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static Modules.SGD SGD(IEnumerable<(string name, Parameter parameter)> parameters, double learningRate, double momentum = 0, double dampening = 0, double weight_decay = 0, bool nesterov = false, bool maximize = false)
            {
                return new Modules.SGD(parameters.Select(np => np.parameter), learningRate, momentum, dampening, weight_decay, nesterov, maximize);
            }

            /// <summary>
            /// Implements the stochastic gradient descent (optionally with momentum).
            ///
            /// The use of momentum is covered in:
            /// http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="learningRate">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static Modules.SGD SGD(IEnumerable<SGD.ParamGroup> parameters, double learningRate, double momentum = 0.0, double dampening = 0, double weight_decay = 0, bool nesterov = false, bool maximize = false)
            {
                return new Modules.SGD(parameters, learningRate, momentum, dampening, weight_decay, nesterov, maximize);
            }
        }
    }
    namespace Modules
    {
        using static torch.optim;

        public class SGD : OptimizerHelper, IMomentum
        {
            /// <summary>
            /// Implements stochastic gradient descent (optionally with momentum).
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public SGD(IEnumerable<Parameter> parameters, double lr, double momentum = 0.0, double dampening = 0, double weight_decay = 0, bool nesterov = false, bool maximize = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, momentum, dampening, weight_decay, nesterov, maximize)
            {
            }

            /// <summary>
            /// Implements stochastic gradient descent (optionally with momentum).
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public SGD(IEnumerable<ParamGroup> parameters, double lr, double momentum = 0.0, double dampening = 0, double weight_decay = 0, bool nesterov = false, bool maximize = false)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (momentum < 0.0) throw new ArgumentException($"Invalid momentum value: {momentum}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");
                if (nesterov && (momentum <= 0 || dampening != 0)) throw new ArgumentException("Nesterov momentum requires a momentum and zero dampening");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    dampening = dampening,
                    maximize = maximize,
                    momentum = momentum,
                    nesterov = nesterov,
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

                    var options = group.Options;
                    var momentum = options.momentum.Value;
                    var dampening = options.dampening.Value;
                    var weight_decay = options.weight_decay.Value;
                    var nesterov = options.nesterov.Value;
                    var maximize = options.maximize.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = _state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        if (momentum != 0) {
                            var buf = state.momentum_buffer;

                            if (buf is null) {
                                buf = grad.clone().detach().DetatchFromDisposeScope();
                                state.momentum_buffer = buf;
                            } else {
                                buf.mul_(momentum).add_(grad, alpha: (1 - dampening));
                            }

                            if (nesterov) {
                                grad = grad.add(buf, alpha: momentum);
                            } else {
                                grad = buf;
                            }

                            state.momentum_buffer = buf;
                        }

                        var alpha = maximize ? lr : -lr;
                        param.add_(grad, alpha: alpha);

                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var kvp in _state) {
                    if (kvp.Value.momentum_buffer is not null) {
                        kvp.Value.momentum_buffer.Dispose();
                    }
                }
                _state.Clear();
            }

            private class State
            {
                public Tensor momentum_buffer;
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
                if (!opt.dampening.HasValue) opt.dampening = def.dampening;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.nesterov.HasValue) opt.nesterov = def.nesterov;
                if (!opt.maximize.HasValue) opt.maximize = def.maximize;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.momentum_buffer = null;
                }
            }

            public class Options : Modules.OptimizerOptions
            {
                public double? momentum;
                public double? dampening;
                public double? weight_decay;
                public bool? nesterov;
                public bool? maximize;
            }

            public class ParamGroup : ParamGroup<Options>, IMomentum
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-3, double momentum = 0.0, double dampening = 0, double weight_decay = 0, bool nesterov = false, bool maximize = false)
                    : base(parameters, new SGD.Options { LearningRate = lr, dampening = dampening, momentum = momentum, weight_decay = weight_decay, nesterov = nesterov, maximize = maximize })
                {
                }

                public double Momentum { get => Options.momentum.Value; set => Options.momentum = value; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();

            public double Momentum { get => (_defaults as Options).momentum.Value; set => (_defaults as Options).momentum = value; }
        }
    }
}
