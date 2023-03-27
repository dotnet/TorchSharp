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

                        var state = (State)_state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        if (momentum != 0) {
                            var buf = state.momentum_buffer;

                            if (buf is null) {
                                buf = grad.clone().detach().DetachFromDisposeScope();
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
                    var st = (State)kvp.Item2;
                    if (st.momentum_buffer is not null) {
                        st.momentum_buffer.Dispose();
                    }
                }
                _state.Clear();
            }

            public class State : OptimizerState, IDisposable
            {
                public Tensor momentum_buffer;

                public void Dispose()
                {
                    momentum_buffer?.Dispose();
                }

                /// <summary>
                /// Move all the state to the indicated device.
                /// </summary>
                /// <param name="device">The device to move all state to.</param>
                public override void to(Device device)
                {
                    if (momentum_buffer is not null)
                        momentum_buffer.to(device);
                }

                /// <summary>
                /// Load the optimizer parameter state from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    LoadConditionalStateTensor(reader, ref momentum_buffer);
                }

                /// <summary>
                /// Save the optimizer parameter state to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    SaveConditionalStateTensor(writer, momentum_buffer);
                }

                /// <summary>
                /// Load optimizer parameter state from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer state record.</param>
                public override void LoadStateDict(OptimizerState source)
                {
                    var st_state = source as State;
                    if (momentum_buffer is not null) {
                        momentum_buffer.Dispose();
                    }
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
                    return (rhs is not null) && (momentum_buffer is null || momentum_buffer.allclose(rhs.momentum_buffer));
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
                if (!opt.dampening.HasValue) opt.dampening = def.dampening;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.nesterov.HasValue) opt.nesterov = def.nesterov;
                if (!opt.maximize.HasValue) opt.maximize = def.maximize;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state.Add((p.Handle,state));
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

                /// <summary>
                /// Load the optimizer options (param-group hyperparameters) from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    base.LoadStateDict(reader);
                    momentum = reader.ReadDouble();
                    dampening = reader.ReadDouble();
                    weight_decay = reader.ReadDouble();
                    nesterov = reader.ReadBoolean();
                    maximize = reader.ReadBoolean();
                }

                /// <summary>
                /// Load optimizer options (param-group hyperparameters) from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer options record.</param>
                public override void LoadStateDict(OptimizerOptions source)
                {
                    base.LoadStateDict(source);
                    var opts = source as Options;
                    momentum = opts.momentum;
                    dampening = opts.dampening;
                    weight_decay = opts.weight_decay;
                    nesterov = opts.nesterov;
                    maximize = opts.maximize;
                }

                /// <summary>
                /// Save the optimizer options (param-group hyperparameters) to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    base.SaveStateDict(writer);
                    writer.Write(momentum.Value);
                    writer.Write(dampening.Value);
                    writer.Write(weight_decay.Value);
                    writer.Write(nesterov.Value);
                    writer.Write(maximize.Value);
                }
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

            public double Momentum { get => (_defaults as Options).momentum.Value; set => (_defaults as Options).momentum = value; }
        }
    }
}
