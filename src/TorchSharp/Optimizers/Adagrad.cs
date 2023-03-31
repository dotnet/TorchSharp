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

                        var state = (State)_state[param.handle];

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
                    ((State)kvp.Item2).Dispose();
                }
            }

            public class State : OptimizerState, IDisposable
            {
                public long step;
                public Tensor sum;

                public void Dispose()
                {
                    sum.Dispose();
                }

                /// <summary>
                /// Move all the state to the indicated device.
                /// </summary>
                /// <param name="device">The device to move all state to.</param>
                public override void to(Device device)
                {
                    sum.to(device);
                }

                /// <summary>
                /// Load the optimizer parameter state from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    step = reader.ReadInt64();
                    sum.Load(reader);
                }

                /// <summary>
                /// Load optimizer parameter state from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer state record.</param>
                public override void LoadStateDict(OptimizerState source)
                {
                    var st_state = source as State;
                    sum.Dispose();
                    step = st_state.step;
                    sum = st_state.sum;
                }

                /// <summary>
                /// Save the optimizer parameter state to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    writer.Write(step);
                    sum.Save(writer);
                }

                /// <summary>
                /// Useful for tests, allows comparison of one state with another.
                /// </summary>
                /// <param name="other">The other optimizer state</param>
                /// <returns></returns>
                public override bool ApproximatelyEquals(OptimizerState other)
                {
                    var rhs = other as State;
                    return (rhs is not null) && step == rhs.step && sum.allclose(rhs.sum);
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
                    state.sum = torch.full_like(p, init_value).DetachFromDisposeScope();
                }
            }

            public class Options : OptimizerOptions
            {
                public double? lr_decay;
                public double? initial_accumulator_value;
                public double? eps;
                public double? weight_decay;

                /// <summary>
                /// Load optimizer options (param-group hyperparameters) from another optimizer.
                /// </summary>
                /// <param name="source">An optimizer options record.</param>
                public override void LoadStateDict(OptimizerOptions source)
                {
                    base.LoadStateDict(source);
                    var opts = source as Options;
                    lr_decay = opts.lr_decay;
                    initial_accumulator_value = opts.initial_accumulator_value;
                    eps = opts.eps;
                    weight_decay = opts.weight_decay;
                }

                /// <summary>
                /// Load the optimizer options (param-group hyperparameters) from a stream.
                /// </summary>
                /// <param name="reader">A binary reader connected to a stream open for reading.</param>
                public override void LoadStateDict(BinaryReader reader)
                {
                    base.LoadStateDict(reader);
                    lr_decay = reader.ReadDouble();
                    initial_accumulator_value = reader.ReadDouble();
                    eps = reader.ReadDouble();
                    weight_decay = reader.ReadDouble();
                }

                /// <summary>
                /// Save the optimizer options (param-group hyperparameters) to a stream.
                /// </summary>
                /// <param name="writer">A binary writer connected to a stream open for writing.</param>
                public override void SaveStateDict(BinaryWriter writer)
                {
                    base.SaveStateDict(writer);
                    writer.Write(lr_decay.Value);
                    writer.Write(initial_accumulator_value.Value);
                    writer.Write(eps.Value);
                    writer.Write(weight_decay.Value);
                }
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
        }
    }
}
