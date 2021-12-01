// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class optim
        {
            public static partial class lr_scheduler
            {
                public abstract class LRScheduler
                {
                    protected LRScheduler()
                    {

                    }

                    protected LRScheduler(ILearningRateController optimizer, int last_epoch = -1, bool verbose = false)
                    {
                        _optimizer = optimizer;
                        _last_epoch = last_epoch;
                        _verbose = verbose;
                        _base_lr = optimizer.InitialLearningRate;
                        _last_lr = optimizer.LearningRate;

                        if (last_epoch == -1) {
                            optimizer.InitialLearningRate = optimizer.LearningRate;
                        }
                    }

                    public virtual void step()
                    {
                        _step_count += 1;
                        _last_epoch += 1;

                        // NOTE: It is super-important to use the 'LearningRate' property once per step(), since
                        //       for most LR schedulers, it will modify the internal state.
                        var lr = LearningRate;

                        _optimizer.LearningRate = lr;
                        Print(_last_lr != lr);
                        _last_lr = lr;
                    }

                    public double get_learning_rate() => LearningRate;

                    private void Print(bool changed)
                    {
                        if (_verbose && changed)
                            Console.WriteLine($"Adjusting learning rate to {LearningRate}");
                    }

                    public virtual double LearningRate => _optimizer.LearningRate;

                    protected ILearningRateController _optimizer;
                    protected int _last_epoch = -1;
                    protected double _last_lr = 0;
                    protected bool _verbose = false;
                    protected int _step_count = 0;
                    protected double _base_lr;
                }

                public static partial class impl
                {
                    /// <summary>
                    /// Sets the learning rate of each parameter group to the initial lr times a given function.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class LambdaLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="lr_lambda">A function which computes a multiplicative factor given an integer parameter epoch.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public LambdaLR(ILearningRateController optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _lr_lambda = lr_lambda;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                return _base_lr * _lr_lambda(_last_epoch);
                            }
                        }

                        private Func<int, double> _lr_lambda;
                    }

                    /// <summary>
                    /// Multiply the learning rate of each parameter group by the factor given in the specified function.
                    /// When last_epoch = -1, sets initial lr as lr.
                    /// </summary>
                    public class MultiplicativeLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="lr_lambda">A function which computes a multiplicative factor given an integer parameter epoch.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public MultiplicativeLR(ILearningRateController optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _lr_lambda = lr_lambda;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                return (_last_epoch > 0)
                                    ? _optimizer.LearningRate * _lr_lambda(_last_epoch)
                                    : _optimizer.LearningRate;
                            }
                        }

                        private Func<int, double> _lr_lambda;
                    }

                    /// <summary>
                    /// Decays the learning rate of each parameter group by gamma every step_size epochs.
                    /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class StepLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="step_size">Period of learning rate decay.</param>
                        /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public StepLR(ILearningRateController optimizer, int step_size, double gamma = 0.1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _step_size = step_size;
                            _gamma = gamma;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                return (_last_epoch == 0) || (_last_epoch % _step_size != 0)
                                    ? _optimizer.LearningRate
                                    : _optimizer.LearningRate * _gamma;
                            }
                        }

                        private int _step_size;
                        private double _gamma;
                    }

                    /// <summary>
                    /// Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
                    /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class MultiStepLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="milestones">List of epoch indices. Must be increasing.</param>
                        /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public MultiStepLR(ILearningRateController optimizer, IList<int> milestones, double gamma = 0.1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _milestones = milestones;
                            _gamma = gamma;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                var idx = _milestones.IndexOf(_last_epoch);
                                return idx == -1
                                    ? _optimizer.LearningRate
                                    : _optimizer.LearningRate * Math.Pow(_gamma, _milestones[idx]);
                            }
                        }

                        private IList<int> _milestones;
                        private double _gamma;
                    }

                    /// <summary>
                    /// Decays the learning rate of each parameter group by gamma every epoch.
                    /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class ExponentialLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public ExponentialLR(ILearningRateController optimizer, double gamma = 0.1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _gamma = gamma;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                return (_last_epoch == 0)
                                    ? _optimizer.LearningRate
                                    : _optimizer.LearningRate * _gamma;
                            }
                        }

                        private double _gamma;
                    }

                    /// <summary>
                    /// Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.
                    /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class ConstantLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="factor">The number we multiply learning rate until the milestone.</param>
                        /// <param name="total_iters">The number of steps that the scheduler decays the learning rate.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public ConstantLR(ILearningRateController optimizer, double factor = 1.0 / 3, int total_iters = 5, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _factor = factor;
                            _total_iters = total_iters;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                if (_last_epoch == 0) {
                                    return _optimizer.LearningRate * _factor;
                                } else if (_last_epoch == _total_iters) {
                                    return _optimizer.LearningRate * (1.0 / _factor);
                                } else {
                                    return _optimizer.LearningRate;
                                }
                            }
                        }

                        private double _factor;
                        private int _total_iters;
                    }


                    /// <summary>
                    /// Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the
                    /// number of epoch reaches a pre-defined milestone: total_iters.
                    /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class LinearLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="start_factor">The number we multiply learning rate in the first epoch. The multiplication factor changes towards end_factor in the following epochs.</param>
                        /// <param name="end_factor">The number we multiply learning rate at the end of linear changing process.</param>
                        /// <param name="total_iters">The number of steps that the scheduler decays the learning rate.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public LinearLR(ILearningRateController optimizer, double start_factor = 1.0 / 3, double end_factor = 5, int total_iters = 5, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _start_factor = start_factor;
                            _end_factor = end_factor;
                            _total_iters = total_iters;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                if (_last_epoch == 0) {
                                    return _optimizer.LearningRate * _start_factor;
                                } else if (_last_epoch > _total_iters) {
                                    return _optimizer.LearningRate;
                                } else {
                                    return (_total_iters * _start_factor + (_last_epoch - 1) * (_end_factor - _start_factor));
                                }
                            }
                        }

                        private double _start_factor;
                        private double _end_factor;
                        private int _total_iters;
                    }


                    /// <summary>
                    /// Set the learning rate of each parameter group using a cosine annealing schedule.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class CosineAnnealingLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="T_max">Maximum number of iterations.</param>
                        /// <param name="eta_min">Minimum learning rate.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public CosineAnnealingLR(ILearningRateController optimizer, double T_max, double eta_min = 0, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _T_max = T_max;
                            _eta_min = eta_min;

                            step();
                        }

                        public override double LearningRate {
                            get {
                                if (_last_epoch == 0) {
                                    return _optimizer.LearningRate;
                                } else if ((_last_epoch - 1 - _T_max) % (2 * _T_max) == 0) {
                                    return _optimizer.LearningRate + (_base_lr - _eta_min) *
                                           (1 - Math.Cos(Math.PI / _T_max)) / 2;
                                } else {
                                    return (1 + Math.Cos(Math.PI * _last_epoch / _T_max)) /
                                           (1 + Math.Cos(Math.PI * (_last_epoch - 1) / _T_max)) *
                                           (_optimizer.LearningRate - _eta_min) + _eta_min;
                                }
                            }
                        }

                        private double _T_max;
                        private double _eta_min;
                    }

                    /// <summary>
                    /// Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
                    /// </summary>
                    public class OneCycleLR : LRScheduler
                    {
                        public enum AnnealStrategy
                        {
                            Cos,
                            Linear
                        }

                        /// <summary>
                        /// Constructor
                        /// </summary>
                        public OneCycleLR(ILearningRateController optimizer,
                            double max_lr,
                            int total_steps = -1,
                            int epochs = -1,
                            int steps_per_epoch = -1,
                            double pct_start = 0.3,
                            AnnealStrategy anneal_strategy = AnnealStrategy.Cos,
                            bool cycle_momentum = true,
                            double base_momentum = 0.85,
                            double max_momentum = 0.95,
                            double div_factor = 25,
                            double final_div_factor = 1e4,
                            bool three_phase = false,
                            int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");

                            _momentum = optimizer as IMomentum;
                            _betas = optimizer as IBetas;
                            if (_momentum == null && _betas == null) throw new ArgumentException($"optimizer must support momentum with `cycle_momentum` option enabled");

                            if (pct_start < 0 || pct_start > 1) throw new ArgumentException($"Expected float between 0 and 1 for pct_start, but got {pct_start}");

                            if (total_steps == -1 && epochs == -1 && steps_per_epoch == -1) {
                                throw new ArgumentException("You must define either total_steps OR (epochs AND steps_per_epoch)");
                            } else if (total_steps != -1) {
                                if (total_steps <= 0) throw new ArgumentException($"Expected positive integer total_steps, but got {total_steps}");
                                _total_steps = total_steps;
                            } else {
                                if (epochs <= 0) throw new ArgumentException($"Expected positive integer epochs, but got {epochs}");
                                if (steps_per_epoch <= 0) throw new ArgumentException($"Expected positive integer steps_per_epoch, but got {steps_per_epoch}");
                                _total_steps = epochs * steps_per_epoch;
                            }

                            _cycle_momentum = cycle_momentum;

                            var initial_lr = max_lr / div_factor;

                            _phase_values["initial_lr"] = initial_lr;
                            _phase_values["max_lr"] = max_lr;
                            _phase_values["min_lr"] = initial_lr / final_div_factor;

                            _annealing_func = (anneal_strategy == AnnealStrategy.Cos)
                                ? (start, end, pct) => end + (start - end) / 2.0 * (Math.Cos(Math.PI * pct) + 1)
                                : (start, end, pct) => (end - start) * pct + start;

                            _schedule_phases = three_phase
                                ? new PhaseDescriptor[] {
                                    new PhaseDescriptor { end_step = pct_start * _total_steps - 1, start_lr = "initial_lr", end_lr = "max_lr", start_momentum = "max_momentum", end_momentum = "base_momentum" },
                                    new PhaseDescriptor { end_step = 2 * pct_start * _total_steps - 2, start_lr = "max_lr", end_lr = "initial_lr", start_momentum = "base_momentum", end_momentum = "max_momentum" },
                                    new PhaseDescriptor { end_step = _total_steps - 1, start_lr = "initial_lr", end_lr = "min_lr", start_momentum = "max_momentum", end_momentum = "max_momentum" }
                                }
                                : new PhaseDescriptor[] {
                                    new PhaseDescriptor { end_step = pct_start * _total_steps - 1, start_lr = "initial_lr", end_lr = "max_lr", start_momentum = "max_momentum", end_momentum = "base_momentum" },
                                    new PhaseDescriptor { end_step = _total_steps - 1, start_lr = "max_lr", end_lr = "min_lr", start_momentum = "base_momentum", end_momentum = "max_momentum" },
                                };

                            if (last_epoch == -1) {

                                if (cycle_momentum) {
                                    if (_betas != null) {
                                        var (_, beta2) = _betas.Betas;
                                        _betas.Betas = (max_momentum, beta2);
                                    } else {
                                        _momentum.Momentum = max_momentum;
                                    }
                                    _phase_values["max_momentum"] = max_momentum;
                                    _phase_values["base_momentum"] = base_momentum;
                                }
                            }
                            step();
                        }

                        public override double LearningRate {
                            get {
                                var step_num = _last_epoch;
                                if (step_num > _total_steps) {
                                    throw new InvalidOperationException($"Tried to step {step_num + 1} times. The specified number of total steps is {_total_steps}");
                                }

                                double start_step = 0;
                                double computed_lr = 0;
                                double computed_momentum = 0;

                                for (var i = 0; i < _schedule_phases.Length; i++) {

                                    var phase = _schedule_phases[i];
                                    var end_step = phase.end_step;

                                    if (step_num <= end_step || i == _schedule_phases.Length - 1) {
                                        var pct = (step_num - start_step) / (end_step - start_step);
                                        computed_lr = _annealing_func(_phase_values[phase.start_lr], _phase_values[phase.end_lr], pct);
                                        if (_cycle_momentum) {
                                            computed_momentum = _annealing_func(_phase_values[phase.start_momentum], _phase_values[phase.end_momentum], pct);
                                        }
                                        break;
                                    }
                                    start_step = phase.end_step;
                                }

                                if (_cycle_momentum) {
                                    if (_betas != null) {
                                        var (_, beta2) = _betas.Betas;
                                        _betas.Betas = (computed_momentum, beta2);
                                    } else {
                                        _momentum.Momentum = computed_momentum;
                                    }
                                }
                                return computed_lr;
                            }
                        }

                        private Func<double, double, double, double> _annealing_func;

                        private PhaseDescriptor[] _schedule_phases;
                        private Dictionary<string, double> _phase_values = new Dictionary<string, double>();

                        private bool _cycle_momentum;

                        private IBetas _betas;
                        private IMomentum _momentum;

                        private int _total_steps;

                        private class PhaseDescriptor
                        {
                            public double end_step;
                            public string start_lr;
                            public string end_lr;
                            public string start_momentum;
                            public string end_momentum;
                        }
                    }

                    /// <summary>
                    /// Chains list of learning rate schedulers. It takes a list of chainable learning rate schedulers and applies step() to all of them.
                    /// </summary>
                    public class ChainedLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="schedulers">List of chained schedulers.</param>
                        /// <returns>A scheduler</returns>
                        public ChainedLR(IEnumerable<LRScheduler> schedulers)
                        {
                            _schedulers = schedulers;
                        }

                        public override void step()
                        {
                            foreach (var sched in _schedulers) {
                                sched.step();
                            }
                        }

                        private IEnumerable<LRScheduler> _schedulers;
                    }
                }

                /// <summary>
                /// Decays the learning rate of each parameter group by gamma every step_size epochs.
                /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                /// When last_epoch=-1, sets initial lr as lr.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="step_size">Period of learning rate decay.</param>
                /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler StepLR(ILearningRateController optimizer, int step_size, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.StepLR(optimizer, step_size, gamma, last_epoch, verbose);
                }

                /// <summary>
                /// Sets the learning rate of each parameter group to the initial lr times a given function.
                /// When last_epoch=-1, sets initial lr as lr.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="lr_lambda">A function which computes a multiplicative factor given an integer parameter epoch.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler LambdaLR(ILearningRateController optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.LambdaLR(optimizer, lr_lambda, last_epoch, verbose);

                }

                /// <summary>
                /// Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
                /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                /// When last_epoch=-1, sets initial lr as lr.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="milestones">List of epoch indices. Must be increasing.</param>
                /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler MultiStepLR(ILearningRateController optimizer, IList<int> milestones, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.MultiStepLR(optimizer, milestones, gamma, last_epoch, verbose);

                }

                /// <summary>
                /// Multiply the learning rate of each parameter group by the factor given in the specified function.
                /// When last_epoch = -1, sets initial lr as lr.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="lr_lambda">A function which computes a multiplicative factor given an integer parameter epoch.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler MultiplicativeLR(ILearningRateController optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.MultiplicativeLR(optimizer, lr_lambda, last_epoch, verbose);

                }

                /// <summary>
                /// Decays the learning rate of each parameter group by gamma every epoch.
                /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                /// When last_epoch=-1, sets initial lr as lr.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler ExponentialLR(ILearningRateController optimizer, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.ExponentialLR(optimizer, gamma, last_epoch, verbose);
                }

                /// <summary>
                /// Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.
                /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
                /// When last_epoch=-1, sets initial lr as lr.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="factor">The number we multiply learning rate until the milestone.</param>
                /// <param name="total_iters">The number of steps that the scheduler decays the learning rate.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler ConstantLR(ILearningRateController optimizer, double factor = 1.0 / 3, int total_iters = 5, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.ConstantLR(optimizer, factor, total_iters, last_epoch, verbose);
                }

                /// <summary>
                /// Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the
                /// number of epoch reaches a pre-defined milestone: total_iters.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="start_factor">The number we multiply learning rate in the first epoch. The multiplication factor changes towards end_factor in the following epochs.</param>
                /// <param name="end_factor">The number we multiply learning rate at the end of linear changing process.</param>
                /// <param name="total_iters">The number of steps that the scheduler decays the learning rate.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler LinearLR(ILearningRateController optimizer, double start_factor = 1.0 / 3, double end_factor = 5, int total_iters = 5, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.LinearLR(optimizer, start_factor, end_factor, total_iters, last_epoch, verbose);
                }

                /// <summary>
                /// Chains list of learning rate schedulers. It takes a list of chainable learning rate schedulers and applies step() to all of them.
                /// </summary>
                ///<param name="schedulers">List of chained schedulers.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler ChainedLR(IEnumerable<LRScheduler> schedulers)
                {
                    return new impl.ChainedLR(schedulers);
                }

                /// <summary>
                /// Sets the learning rate using a cosine annealing schedule.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="T_max">Maximum number of iterations.</param>
                /// <param name="eta_min">Minimum learning rate.</param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler CosineAnnealingLR(ILearningRateController optimizer, double T_max, double eta_min = 0, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch, verbose);
                }

                /// <summary>
                /// Sets the learning rate of each parameter group according to the
                /// 1cycle learning rate policy.The 1cycle policy anneals the learning
                /// rate from an initial learning rate to some maximum learning rate and then
                /// from that maximum learning rate to some minimum learning rate much lower
                /// than the initial learning rate.
                /// 
                /// This policy was initially described in the paper `Super-Convergence:
                /// Very Fast Training of Neural Networks Using Large Learning Rates`_.
                ///
                /// The 1cycle learning rate policy changes the learning rate after every batch.
                /// `step` should be called after a batch has been used for training.
                ///
                /// This scheduler is not chainable.                
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="max_lr">Upper learning rate boundaries in the cycle</param>
                /// <param name="total_steps">
                /// The total number of steps in the cycle.
                /// Note that if a value is not provided here, then it must be inferred by providing a value for epochs and steps_per_epoch.
                /// </param>
                /// <param name="epochs">
                /// The number of epochs to train for. This is used along
                /// with steps_per_epoch in order to infer the total number of steps in the cycle
                /// if a value for total_steps is not provided.
                /// </param>
                /// <param name="steps_per_epoch">
                /// The number of steps per epoch to train for. This is
                /// used along with epochs in order to infer the total number of steps in the
                /// cycle if a value for total_steps is not provided.
                /// </param>
                /// <param name="pct_start">The percentage of the cycle (in number of steps) spent increasing the learning rate.</param>
                /// <param name="anneal_strategy">Specifies the annealing strategy: "cos" for cosine annealing, "linear" for linear annealing.</param>
                /// <param name="cycle_momentum">If true, momentum is cycled inversely to learning rate between 'base_momentum' and 'max_momentum'.</param>
                /// <param name="base_momentum">
                /// Lower momentum boundaries in the cycle
                /// for each parameter group.Note that momentum is cycled inversely
                /// to learning rate; at the peak of a cycle, momentum is
                /// 'base_momentum' and learning rate is 'max_lr'.
                /// </param>
                /// <param name="max_momentum">
                /// Upper momentum boundaries in the cycle for each parameter group.
                /// Functionally, it defines the cycle amplitude(max_momentum - base_momentum).
                /// Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is 'max_momentum'
                /// and learning rate is 'base_lr'
                /// </param>
                /// <param name="div_factor">Determines the initial learning rate via initial_lr = max_lr/div_factor</param>
                /// <param name="final_div_factor">Determines the minimum learning rate via min_lr = initial_lr/final_div_factor</param>
                /// <param name="three_phase">
                /// If ``True``, use a third phase of the schedule to annihilate the
                /// learning rate according to 'final_div_factor' instead of modifying the second
                /// phase (the first two phases will be symmetrical about the step indicated by 'pct_start').
                /// </param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                /// <remarks>
                /// Note also that the total number of steps in the cycle can be determined in one
                /// of two ways (listed in order of precedence):
                ///
                /// #. A value for total_steps is explicitly provided.
                /// #. A number of epochs (epochs) and a number of steps per epoch
                /// (steps_per_epoch) are provided.
                /// In this case, the number of total steps is inferred by
                /// total_steps = epochs * steps_per_epoch
                ///
                /// You must either provide a value for total_steps or provide a value for both
                /// epochs and steps_per_epoch.
                ///
                /// The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
                /// claims that "unpublished work has shown even better results by using only two phases". To
                /// mimic the behaviour of the original paper instead, set ``three_phase= True``.
                /// </remarks>
                public static LRScheduler OneCycleLR(ILearningRateController optimizer,
                    double max_lr,
                    int total_steps = -1,
                    int epochs = -1,
                    int steps_per_epoch = -1,
                    double pct_start = 0.3,
                    impl.OneCycleLR.AnnealStrategy anneal_strategy = impl.OneCycleLR.AnnealStrategy.Cos,
                    bool cycle_momentum = true,
                    double base_momentum = 0.85,
                    double max_momentum = 0.95,
                    double div_factor = 25,
                    double final_div_factor = 1e4,
                    bool three_phase = false,
                    int last_epoch = -1, bool verbose = false)
                {
                    return new impl.OneCycleLR(optimizer, max_lr, total_steps, epochs, steps_per_epoch, pct_start, anneal_strategy, cycle_momentum, base_momentum, max_momentum, div_factor, final_div_factor, three_phase, last_epoch, verbose);
                }
            }
        }
    }
}
