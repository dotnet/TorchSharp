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
            /// Implements the the resilient backpropagation algorithm.
            ///
            /// A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
            /// http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public static Rprop Rprop(IEnumerable<Parameter> parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
            {
                return new Rprop(parameters, lr, etaminus, etaplus, min_step, max_step);
            }

            /// <summary>
            /// Implements the the resilient backpropagation algorithm.
            ///
            /// A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
            /// http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public static Rprop Rprop(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
            {
                return new Rprop(parameters.Select(np => np.parameter), lr, etaminus, etaplus, min_step, max_step);
            }

            /// <summary>
            /// Implements the the resilient backpropagation algorithm.
            ///
            /// A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
            /// http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public static Rprop Rprop(IEnumerable<Rprop.ParamGroup> parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
            {
                return new Rprop(parameters, lr, etaminus, etaplus, min_step, max_step);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        public class Rprop : OptimizerHelper
        {
            /// <summary>
            /// Implements Rprop algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public Rprop(IEnumerable<Parameter> parameters, double lr = 0.01, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
                : this(new ParamGroup[] { new() { Parameters = parameters } }, lr, etaminus, etaplus, min_step, max_step)
            {
            }

            /// <summary>
            /// Implements Rprop algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="etaminus">Multiplicative increase factor.</param>
            /// <param name="etaplus">Multiplicative decrease factor.</param>
            /// <param name="min_step">Minimum allowed step size.</param>
            /// <param name="max_step">Maximum allowed step size.</param>
            /// <returns></returns>
            public Rprop(IEnumerable<ParamGroup> parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    etaminus = etaminus,
                    etaplus = etaplus,
                    min_step = min_step,
                    max_step = max_step
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
                    var etaminus = options.etaminus.Value;
                    var etaplus = options.etaplus.Value;
                    var min_step = options.min_step.Value;
                    var max_step = options.max_step.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (grad.is_sparse) throw new ArgumentException("Rprop does not support sparse gradients");

                        var state = _state[param.handle];

                        state.step += 1;

                        grad = (max_step != 0)
                            ? grad.add(param, alpha: max_step)
                            : grad.alias();

                        var sign = grad.mul(state.prev).sign();
                        sign[sign.gt(0)] = (Tensor)etaplus;
                        sign[sign.lt(0)] = (Tensor)etaminus;
                        sign[sign.eq(0)] = (Tensor)1;

                        state.step_size.mul_(sign).clamp_(min_step, max_step);

                        grad = grad.clone();

                        grad.index_put_(0, sign.eq(etaminus));

                        param.addcmul_(grad.sign(), state.step_size, -1);

                        state.prev.copy_(grad);
                    }

                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.Dispose();
                }
            }

            private class State : IDisposable
            {
                public long step;
                public Tensor prev;
                public Tensor step_size;

                public void Dispose()
                {
                    prev.Dispose();
                    step_size.Dispose();
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
                if (!opt.etaminus.HasValue) opt.etaminus = def.etaminus;
                if (!opt.etaplus.HasValue) opt.etaplus = def.etaplus;
                if (!opt.min_step.HasValue) opt.min_step = def.min_step;
                if (!opt.max_step.HasValue) opt.max_step = def.max_step;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.prev = torch.zeros_like(p);
                    state.step_size = p.new_empty(p.shape).fill_(opt.LearningRate);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? etaminus;
                public double? etaplus;
                public double? min_step;
                public double? max_step;
            }

            public class ParamGroup : ParamGroup<Options>
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-2, double etaminus = 0.5, double etaplus = 1.2, double min_step = 1e-6, double max_step = 50)
                    : base(parameters, new Rprop.Options { LearningRate = lr, etaminus = etaminus, etaplus = etaplus, min_step = min_step, max_step = max_step })
                {
                }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();

        }
    }
}
