// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Security;
using TorchSharp.Modules;

// All LR schedulers in this file are directly based on the Pytorch implementation at:
//
// https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
//

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

                    protected LRScheduler(Optimizer optimizer, int last_epoch = -1, bool verbose = false)
                    {
                        _optimizer = optimizer;
                        _last_epoch = last_epoch;
                        _verbose = verbose;
                        _base_lrs = optimizer.ParamGroups.Select(pg => pg.InitialLearningRate).ToList();
                        _last_lrs = optimizer.ParamGroups.Select(pg => pg.LearningRate).ToList();

                        if (last_epoch == -1) {
                            foreach (var pg in optimizer.ParamGroups) {
                                pg.InitialLearningRate = pg.LearningRate;
                            }
                        }
                    }

                    /// <summary>
                    /// Advance the learning rate schedule.                 
                    /// </summary>
                    /// <remarks>Typically, this is done once per epoch or once every N batches.</remarks>
                    public virtual void step()
                    {
                        step(null);
                    }

                    /// <summary>
                    /// Advance the learning rate schedule.                 
                    /// </summary>
                    /// <remarks>Typically, this is done once per epoch or once every N batches.</remarks>
                    public virtual void step(int? epoch)
                    {
                        _step_count += 1;
                        _last_epoch = (epoch.HasValue) ? epoch.Value : _last_epoch + 1;

                        // NOTE: It is super-important to use the 'get_lr()' method no more than once per step(), since
                        //       for many LR schedulers, it will modify the internal state of the scheduler,
                        //       as well as that of the controlled optimizer.
                        var lr = get_lr().ToList();
                        var pgs = _optimizer.ParamGroups.ToList();

                        for (int i = 0; i < _base_lrs.Count; i++) {

                            pgs[i].LearningRate = lr[i];
                            if (_verbose && _last_lrs[i] != lr[i])
                                Console.WriteLine($"Adjusting learning rate to {lr[i]}");
                            _last_lrs[i] = lr[i];
                        }
                    }

                    /// <summary>
                    /// Advance the learning rate scheduler, passing in the current value of some metric.
                    /// </summary>
                    /// <remarks>
                    /// The metric value is ignored by most LR schedulers.
                    /// </remarks>
                    public virtual void step(double current, int? epoch = null)
                    {
                        // Ignore the metric
                        step(epoch);
                    }

                    /// <summary>
                    /// Compute the current learning rate for the scheduler.
                    /// </summary>
                    protected virtual IEnumerable<double> get_lr() => _optimizer.ParamGroups.Select(pg => pg.LearningRate);

                    /// <summary>
                    /// Return last computed learning rate by current scheduler.
                    /// </summary>
                    public IEnumerable<double> get_last_lr() => _last_lrs;

                    internal Optimizer _optimizer;
                    internal int _last_epoch = -1;
                    internal bool _verbose = false;
                    protected int _step_count = 0;
                    internal IList<double> _last_lrs;
                    internal IList<double> _base_lrs;
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
                        public LambdaLR(Optimizer optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _lr_lambdas = Enumerable.Repeat(lr_lambda, optimizer.ParamGroups.Count()).ToList();

                            step();
                        }

                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="lr_lambdas">A list of functions, one for each paramater group, which computes a multiplicative factor given an integer parameter epoch.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public LambdaLR(Optimizer optimizer, IEnumerable<Func<int, double>> lr_lambdas, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _lr_lambdas = lr_lambdas.ToList();

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            var pgs = _optimizer.ParamGroups.ToList();
                            return Enumerable.Range(0, pgs.Count).Select(i => _base_lrs[i] * _lr_lambdas[i](_last_epoch));
                        }

                        private List<Func<int, double>> _lr_lambdas;
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
                        /// <param name="lr_lambdas">A list of functions, one for each paramater group, which computes a multiplicative factor given an integer parameter epoch.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public MultiplicativeLR(Optimizer optimizer, IEnumerable<Func<int, double>> lr_lambdas, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _lr_lambdas = lr_lambdas.ToList();

                            step();
                        }

                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="lr_lambda">A function which computes a multiplicative factor given an integer parameter epoch.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public MultiplicativeLR(Optimizer optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _lr_lambdas = Enumerable.Repeat(lr_lambda, optimizer.ParamGroups.Count()).ToList();

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            var pgs = _optimizer.ParamGroups.ToList();
                            return (_last_epoch > 0)
                                    ? Enumerable.Range(0, pgs.Count).Select(i => pgs[i].LearningRate * _lr_lambdas[i](_last_epoch))
                                    : Enumerable.Range(0, pgs.Count).Select(i => pgs[i].LearningRate);
                        }

                        private List<Func<int, double>> _lr_lambdas;
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
                        public StepLR(Optimizer optimizer, int step_size, double gamma = 0.1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _step_size = step_size;
                            _gamma = gamma;

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            return (_last_epoch == 0) || (_last_epoch % _step_size != 0)
                                    ? _optimizer.ParamGroups.Select(pg => pg.LearningRate)
                                    : _optimizer.ParamGroups.Select(pg => pg.LearningRate * _gamma);
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
                        public MultiStepLR(Optimizer optimizer, IList<int> milestones, double gamma = 0.1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _milestones = milestones;
                            _gamma = gamma;

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            var idx = _milestones.IndexOf(_last_epoch);
                            return idx == -1
                                ? _optimizer.ParamGroups.Select(pg => pg.LearningRate)
                                : _optimizer.ParamGroups.Select(pg => pg.LearningRate * _gamma);
                        }

                        private IList<int> _milestones;
                        private double _gamma;
                    }

                    /// <summary>
                    /// Decays the learning rate of each parameter group using a polynomial function in the given total_iters.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class PolynomialLR : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <param name="optimizer">Wrapped optimizer.</param>
                        /// <param name="total_iters">The number of steps that the scheduler decays the learning rate.</param>
                        /// <param name="power">The power of the polynomial.</param>
                        /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                        /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                        /// <returns>A scheduler</returns>
                        public PolynomialLR(Optimizer optimizer, int total_iters = 5, int power = 1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _power = power;
                            _total_iters = total_iters;
                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            if (_last_epoch == 0 || _last_epoch > _total_iters) {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate);
                            }
                            else {
                                var decay_factor = ((1.0 - _last_epoch / _total_iters) / (1.0 - (_last_epoch - 1) / _total_iters));
                                if (_power != 1) {
                                    decay_factor = Math.Pow(decay_factor, _power);
                                }
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate * decay_factor);
                            }
                        }

                        private double _total_iters;
                        private int _power;
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
                        public ExponentialLR(Optimizer optimizer, double gamma = 0.1, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _gamma = gamma;

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            return (_last_epoch == 0)
                                    ? _optimizer.ParamGroups.Select(pg => pg.LearningRate)
                                    : _optimizer.ParamGroups.Select(pg => pg.LearningRate * _gamma);
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
                        public ConstantLR(Optimizer optimizer, double factor = 1.0 / 3, int total_iters = 5, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _factor = factor;
                            _total_iters = total_iters;

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            if (_last_epoch == 0) {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate * _factor);
                            } else if (_last_epoch == _total_iters) {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate * (1.0 / _factor));
                            } else {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate);
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
                        public LinearLR(Optimizer optimizer, double start_factor = 1.0 / 3, double end_factor = 5, int total_iters = 5, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _start_factor = start_factor;
                            _end_factor = end_factor;
                            _total_iters = total_iters;

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            if (_last_epoch == 0) {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate * _start_factor);
                            } else if (_last_epoch > _total_iters) {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate);
                            } else {
                                var factor = (1 + (_end_factor - _start_factor)) / (_total_iters * _start_factor + (_last_epoch - 1) * (_end_factor - _start_factor));
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate * factor);
                            }
                        }

                        private double _start_factor;
                        private double _end_factor;
                        private int _total_iters;
                    }

                    /// <summary>
                    /// Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch.
                    /// </summary>
                    public class SequentialLR : LRScheduler
                    {
                        public SequentialLR(Optimizer optimizer, IEnumerable<LRScheduler> schedulers, IEnumerable<int> milestones, int last_epoch = -1) : base(optimizer, last_epoch + 1, false)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");

                            foreach (var scheduler in schedulers) {
                                if (scheduler._optimizer != optimizer)
                                    throw new ArgumentException($"The SequentialLR scheduler expects all its sub-schedulers to be bound to the same optimizer.");
                            }

                            if (milestones.Count() != schedulers.Count() - 1) {
                                throw new ArgumentException($"Received {schedulers.Count()} schedulers and {milestones.Count()} milestones. The SequentialLR scheduler expects the number of schedulers to be one more than the number of milestones. ");
                            }

                            _schedulers = schedulers.ToArray();
                            _milestones = milestones.ToArray();

                            foreach (var group in optimizer.ParamGroups) {
                                group.LearningRate = group.InitialLearningRate;
                            }
                            foreach (var scheduler in _schedulers) {
                                scheduler._last_epoch -= 1;
                            }
                            _schedulers[0].step();
                            _last_lrs = _schedulers[0]._last_lrs;
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {

                            var idx = _milestones.Length;
                            for (; idx > 0 ;) {
                                if (_milestones[idx-1] <= _last_epoch) break;
                                idx -= 1;
                            }

                            var scheduler = _schedulers[idx];
                            if (idx > 0 && _milestones[idx - 1] == _last_epoch)
                                scheduler.step(0);
                            else
                                scheduler.step();

                            return _optimizer.ParamGroups.Select(pg => pg.LearningRate);
                        }

                        private LRScheduler[] _schedulers;
                        private int[] _milestones;
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
                        public CosineAnnealingLR(Optimizer optimizer, double T_max, double eta_min = 0, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _T_max = T_max;
                            _eta_min = eta_min;

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            if (_last_epoch == 0) {
                                return _optimizer.ParamGroups.Select(pg => pg.LearningRate);
                            } else if ((_last_epoch - 1 - _T_max) % (2 * _T_max) == 0) {
                                var pgs = _optimizer.ParamGroups.ToList();
                                return Enumerable.Range(0, pgs.Count).Select(i => pgs[i].LearningRate + (_base_lrs[i] - _eta_min) * (1 - Math.Cos(Math.PI / _T_max)) / 2);
                            } else {
                                return _optimizer.ParamGroups.Select(pg => (1 + Math.Cos(Math.PI * _last_epoch / _T_max)) /
                                       (1 + Math.Cos(Math.PI * (_last_epoch - 1) / _T_max)) * (pg.LearningRate - _eta_min) + _eta_min);
                            }
                        }

                        private double _T_max;
                        private double _eta_min;
                    }

                    /// <summary>
                    /// Set the learning rate of each parameter group using a cosine annealing schedule.
                    /// When last_epoch=-1, sets initial lr as lr.
                    /// </summary>
                    public class ReduceLROnPlateau : LRScheduler
                    {
                        /// <summary>
                        /// Constructor
                        /// </summary>
                        /// <returns>A scheduler</returns>
                        public ReduceLROnPlateau(Optimizer optimizer, string mode = "min", double factor = 0.1, int patience = 10, double threshold = 1e-4, string threshold_mode = "rel", int cooldown = 0, IList<double> min_lr = null, double eps = 1e-8, bool verbose = false)
                            :base(optimizer, -1, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            if (factor >= 1.0) throw new ArgumentException("Factor should be < 1.0");

                            switch (mode) {
                            case "min":
                            case "max":
                                break;
                            default:
                                throw new ArgumentException($"mode {mode} is unknown!");
                            }
                            switch (threshold_mode) {
                            case "rel":
                            case "abs":
                                break;
                            default:
                                throw new ArgumentException($"threshold mode {threshold_mode} is unknown!");
                            }

                            this._optimizer = optimizer;
                            this.factor = factor;

                            var pgLength = optimizer.ParamGroups.Count();
                            if (min_lr == null || min_lr.Count == 0) {
                                this.min_lrs = Enumerable.Repeat(0.0, pgLength).ToList();
                            }
                            else if (min_lr.Count == 1) {
                                this.min_lrs = Enumerable.Repeat(min_lr[0], pgLength).ToList();
                            }
                            else {
                                this.min_lrs = min_lr.ToList();
                            }

                            this.patience = patience;
                            this.cooldown = cooldown;
                            this.cooldown_counter = 0;
                            this.mode = mode;
                            this.threshold = threshold;
                            this.threshold_mode = threshold_mode;
                            this.best = -1;
                            this.num_bad_epochs = -1;
                            this.eps = eps;

                            this.mode_worst = (mode == "min") ? double.PositiveInfinity : double.NegativeInfinity;

                            reset();
                        }

                        /// <summary>
                        /// Advance the LR scheduler one epoch.
                        /// Unlike other LR schedulers, you're supposed to pass in the most recent value of the metric
                        /// that is used to decided whether to modify the learning rates. 
                        /// </summary>
                        /// <param name="current">The current value of the metric that we're interested in imvproving.</param>
                        /// <param name="epoch">The current epoch.</param>
                        public override void step(double current, int? epoch = null)
                        {
                            if (epoch == null) {
                                epoch = _last_epoch + 1;
                            }
                            _last_epoch = epoch.Value;

                            if (is_better(current, best)) {
                                best = current;
                                num_bad_epochs = 0;
                            }
                            else {
                                num_bad_epochs += 1;
                            }

                            if (cooldown_counter > 0) {
                                cooldown_counter -= 1;
                                num_bad_epochs = 0;
                            }

                            if (num_bad_epochs > patience) {
                                reduce_lr(_last_epoch);
                                cooldown_counter = cooldown;
                                num_bad_epochs = 0;
                            }
                        }

                        public override void step()
                        {
                            throw new InvalidOperationException("step() should not be used with the ReduceLROnPlateau scheduler. Use step(double, int?), instead.");
                        }

                        public override void step(int? epoch)
                        {
                            throw new InvalidOperationException("step(int?) should not be used with the ReduceLROnPlateau scheduler. Use step(double, int?), instead.");
                        }

                        private bool is_better(double a, double best)
                        {
                            if (mode == "min" && threshold_mode == "rel") {
                                return a < best * (1 - threshold);
                            }
                            else if (mode == "min" && threshold_mode == "abs") {
                                return a < best - threshold;
                            }
                            else if (mode == "max" && threshold_mode == "rel") {
                                return a > best * (threshold + 1);
                            }
                            else {
                                return a > best + threshold;
                            }
                        }

                        private void reduce_lr(long epoch)
                        {
                            var pgs = _optimizer.ParamGroups.ToList();

                            for (var i = 0; i < pgs.Count; i++) {
                                var param_group = pgs[i];

                                var old_lr = param_group.LearningRate;
                                var new_lr = Math.Max(old_lr * factor, min_lrs[i]);
                                if (old_lr - new_lr > eps) {
                                    param_group.LearningRate = new_lr;
                                    if (_verbose) {
                                        Console.WriteLine($"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:g4}.");
                                    }
                                }
                            }
                        }

                        private void reset()
                        {
                            best = mode_worst;
                            cooldown_counter = 0;
                            num_bad_epochs = 0;
                        }

                        private int patience;
                        private int cooldown;
                        private int cooldown_counter;
                        private string mode;
                        private double threshold;
                        private string threshold_mode;
                        private double best;
                        private double mode_worst;
                        private int num_bad_epochs;
                        private double eps;

                        private double factor;
                        private List<double> min_lrs;
                    }

                    /// <summary>
                    /// Sets the learning rate of each parameter group according to cyclical learning rate policy(CLR).
                    ///
                    /// The policy cycles the learning rate between two boundaries with a constant frequency, as detailed in
                    /// the paper `Cyclical Learning Rates for Training Neural Networks`_.
                    /// 
                    /// The distance between the two boundaries can be scaled on a per-iteration or per-cycle basis.
                    /// </summary>
                    public class CyclicLR : LRScheduler
                    {
                        public enum Mode
                        {
                            Triangular,
                            Triangular2,
                            ExpRange
                        }
                        public enum ScaleMode
                        {
                            Cycle,
                            Iterations
                        }

                        /// <summary>
                        /// Constructor
                        /// </summary>
                        public CyclicLR(Optimizer optimizer,
                            double base_lr,
                            double max_lr,
                            int step_size_up = 2000,
                            int step_size_down = -1,
                            Mode mode = Mode.Triangular,
                            double gamma = 1.0,
                            Func<double, double> scale_fn = null,
                            ScaleMode scale_mode = ScaleMode.Cycle,
                            bool cycle_momentum = true,
                            double base_momentum = 0.8,
                            double max_momentum = 0.9,
                            int last_epoch = -1,
                            bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");

                            var pgCount = optimizer.ParamGroups.Count();

                            Initialize(
                                optimizer,
                                Enumerable.Repeat(base_lr, pgCount),
                                Enumerable.Repeat(max_lr, pgCount),
                                step_size_up,
                                step_size_down,
                                mode,
                                gamma,
                                scale_fn,
                                scale_mode,
                                cycle_momentum,
                                Enumerable.Repeat(base_momentum, pgCount),
                                Enumerable.Repeat(max_momentum, pgCount));
                        }

                        /// <summary>
                        /// Constructor
                        /// </summary>
                        public CyclicLR(Optimizer optimizer,
                            IEnumerable<double> base_lr,
                            IEnumerable<double> max_lr,
                            int step_size_up = 2000,
                            int step_size_down = -1,
                            Mode mode = Mode.Triangular,
                            double gamma = 1.0,
                            Func<double, double> scale_fn = null,
                            ScaleMode scale_mode = ScaleMode.Cycle,
                            bool cycle_momentum = true,
                            IEnumerable<double> base_momentum = null,
                            IEnumerable<double> max_momentum = null,
                            int last_epoch = -1,
                            bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");

                            Initialize(
                                optimizer,
                                base_lr,
                                max_lr,
                                step_size_up,
                                step_size_down,
                                mode,
                                gamma,
                                scale_fn,
                                scale_mode,
                                cycle_momentum,
                                base_momentum,
                                max_momentum);
                        }

                        private void Initialize(Optimizer optimizer, IEnumerable<double> base_lr, IEnumerable<double> max_lr, int step_size_up, int step_size_down, Mode mode, double gamma, Func<double, double> scale_fn, ScaleMode scale_mode, bool cycle_momentum, IEnumerable<double> base_momentum, IEnumerable<double> max_momentum)
                        {
                            double down = (step_size_down == -1) ? step_size_up : step_size_down;
                            _total_size = step_size_up + down;
                            _step_ratio = step_size_up / _total_size;

                            var pgs = optimizer.ParamGroups.ToList();

                            _max_lrs = max_lr.ToList();
                            _base_lrs = base_lr.ToList();

                            if (_last_epoch == -1) {
                                for (int i = 0; i < pgs.Count; i++) {
                                    pgs[i].LearningRate = _base_lrs[i];
                                }
                            }

                            _mode = mode;
                            _gamma = gamma;
                            _cycle_momentum = cycle_momentum;

                            if (cycle_momentum) {
                                var momentum = optimizer as IMomentum;
                                if (momentum == null && cycle_momentum) throw new ArgumentException($"optimizer must support momentum with `cycle_momentum` option enabled");

                                _base_momentum = (base_momentum is null) ? Enumerable.Repeat(0.8, pgs.Count).ToList() : base_momentum.ToList();
                                _max_momentum = (max_momentum is null) ? Enumerable.Repeat(0.9, pgs.Count).ToList() : max_momentum.ToList();

                                if (_last_epoch == -1) {
                                    for (int i = 0; i < pgs.Count; i++) {
                                        (pgs[i] as IMomentum).Momentum = _base_momentum[i];
                                    }
                                }
                            }


                            if (scale_fn == null) {
                                switch (mode) {
                                case Mode.Triangular:
                                    _scale_func = x => 1.0;
                                    _scale_mode = ScaleMode.Cycle;
                                    break;
                                case Mode.Triangular2:
                                    _scale_func = x => 1.0 / (Math.Pow(2, x - 1));
                                    _scale_mode = ScaleMode.Cycle;
                                    break;
                                case Mode.ExpRange:
                                    _scale_func = x => Math.Pow(this._gamma, x);
                                    _scale_mode = ScaleMode.Cycle;
                                    break;
                                }
                            } else {
                                _scale_func = scale_fn;
                                _scale_mode = scale_mode;
                            }

                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            var pgs = _optimizer.ParamGroups.ToList();

                            var cycle = Math.Floor(1.0 + _last_epoch / _total_size);
                            var x = 1.0 + _last_epoch / _total_size - cycle;

                            var scale_factor = (x <= _step_ratio) ? x / _step_ratio : (x - 1) / (_step_ratio - 1);

                            return Enumerable.Range(0, pgs.Count).Select(i => {

                                var base_height = (_max_lrs[i] - _base_lrs[i]) * scale_factor;

                                var computed_lr = (_scale_mode == ScaleMode.Cycle)
                                    ? _base_lrs[i] + base_height * _scale_func(cycle)
                                    : _base_lrs[i] + base_height * _scale_func(_last_epoch);

                                if (_cycle_momentum) {

                                    base_height = (_max_momentum[i] - _base_momentum[i]) * scale_factor;

                                    (pgs[i] as IMomentum).Momentum = (_scale_mode == ScaleMode.Cycle)
                                        ? _max_momentum[i] + base_height * _scale_func(cycle)
                                        : _max_momentum[i] + base_height * _scale_func(_last_epoch);

                                }

                                return computed_lr;
                            });
                        }

                        private double _total_size;
                        private double _step_ratio;

                        private Func<double, double> _scale_func;
                        private ScaleMode _scale_mode;

                        private bool _cycle_momentum;

                        private List<double> _max_lrs;

                        private List<double> _base_momentum;
                        private List<double> _max_momentum;

                        private Mode _mode;
                        private double _gamma;
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
                        public OneCycleLR(Optimizer optimizer,
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

                            var pgCount = _optimizer.ParamGroups.Count();

                            Initialize(optimizer,
                                Enumerable.Repeat(max_lr, pgCount),
                                total_steps,
                                epochs,
                                steps_per_epoch,
                                pct_start, anneal_strategy,
                                cycle_momentum,
                                Enumerable.Repeat(base_momentum, pgCount),
                                Enumerable.Repeat(max_momentum, pgCount),
                                div_factor,
                                final_div_factor,
                                three_phase,
                                last_epoch);
                        }

                        /// <summary>
                        /// Constructor
                        /// </summary>
                        public OneCycleLR(Optimizer optimizer,
                            IEnumerable<double> max_lr,
                            int total_steps = -1,
                            int epochs = -1,
                            int steps_per_epoch = -1,
                            double pct_start = 0.3,
                            AnnealStrategy anneal_strategy = AnnealStrategy.Cos,
                            bool cycle_momentum = true,
                            IEnumerable<double> base_momentum = null,
                            IEnumerable<double> max_momentum = null,
                            double div_factor = 25,
                            double final_div_factor = 1e4,
                            bool three_phase = false,
                            int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");

                            Initialize(optimizer,
                                max_lr,
                                total_steps,
                                epochs,
                                steps_per_epoch,
                                pct_start,
                                anneal_strategy,
                                cycle_momentum,
                                base_momentum,
                                max_momentum,
                                div_factor,
                                final_div_factor,
                                three_phase,
                                last_epoch);
                        }

                        private void Initialize(Optimizer optimizer, IEnumerable<double> max_lr, int total_steps, int epochs, int steps_per_epoch, double pct_start, AnnealStrategy anneal_strategy, bool cycle_momentum, IEnumerable<double> base_momentum, IEnumerable<double> max_momentum, double div_factor, double final_div_factor, bool three_phase, int last_epoch)
                        {
                            _cycle_momentum = cycle_momentum;

                            var pgs = optimizer.ParamGroups.ToList();

                            if (cycle_momentum) {
                                var _momentum = optimizer as IMomentum;
                                var _betas = optimizer as IBetas;
                                if (_momentum == null && _betas == null) throw new ArgumentException($"optimizer must support momentum with `cycle_momentum` option enabled");
                            }

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

                            var mlr = max_lr.ToList();

                            _betas = pgs.Select(pg => pg as IBetas).ToList();
                            _momentum = pgs.Select(pg => pg as IMomentum).ToList();

                            var initial_lrs = max_lr.Select(mlr => mlr / div_factor).ToList();

                            _phase_values["initial_lr"] = initial_lrs;
                            _phase_values["max_lr"] = mlr;
                            _phase_values["min_lr"] = initial_lrs.Select(ilr => ilr / final_div_factor).ToList();

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

                                    var _base_momentum = (base_momentum is null) ? Enumerable.Repeat(0.85, pgs.Count).ToList() : base_momentum.ToList();
                                    var _max_momentum = (max_momentum is null) ? Enumerable.Repeat(0.95, pgs.Count).ToList() : max_momentum.ToList();

                                    for (int i = 0; i < pgs.Count; i++) {
                                        if (_betas[i] != null) {
                                            var (_, beta2) = _betas[i].Betas;
                                            _betas[i].Betas = (_max_momentum[i], beta2);
                                        } else {
                                            if (_momentum[i] != null)
                                                _momentum[i].Momentum = _max_momentum[i];
                                        }
                                    }
                                    _phase_values["max_momentum"] = _max_momentum;
                                    _phase_values["base_momentum"] = _base_momentum;
                                }
                            }
                            step();
                        }

                        /// <summary>
                        /// Compute the current learning rate for the scheduler.
                        /// </summary>
                        protected override IEnumerable<double> get_lr()
                        {
                            var step_num = _last_epoch;
                            if (step_num > _total_steps) {
                                throw new InvalidOperationException($"Tried to step {step_num + 1} times. The specified number of total steps is {_total_steps}");
                            }

                            var pgs = _optimizer.ParamGroups.ToList();

                            return Enumerable.Range(0, pgs.Count).Select(i => {

                                double start_step = 0;
                                double computed_lr = 0;
                                double computed_momentum = 0;

                                for (var j = 0; j < _schedule_phases.Length; j++) {

                                    var phase = _schedule_phases[j];
                                    var end_step = phase.end_step;

                                    if (step_num <= end_step || i == _schedule_phases.Length - 1) {
                                        var pct = (step_num - start_step) / (end_step - start_step);
                                        computed_lr = _annealing_func(_phase_values[phase.start_lr][i], _phase_values[phase.end_lr][i], pct);
                                        if (_cycle_momentum) {
                                            computed_momentum = _annealing_func(_phase_values[phase.start_momentum][i], _phase_values[phase.end_momentum][i], pct);
                                        }
                                        break;
                                    }
                                    start_step = phase.end_step;
                                }

                                if (_cycle_momentum) {
                                    if (_betas[i] != null) {
                                        var (_, beta2) = _betas[i].Betas;
                                        _betas[i].Betas = (computed_momentum, beta2);
                                    } else {
                                        _momentum[i].Momentum = computed_momentum;
                                    }
                                }
                                return computed_lr;
                            });
                        }

                        private Func<double, double, double, double> _annealing_func;

                        private PhaseDescriptor[] _schedule_phases;
                        private Dictionary<string, List<double>> _phase_values = new Dictionary<string, List<double>>();

                        private bool _cycle_momentum;

                        private List<IBetas> _betas;
                        private List<IMomentum> _momentum;

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
                public static LRScheduler StepLR(Optimizer optimizer, int step_size, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
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
                public static LRScheduler LambdaLR(Optimizer optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false)
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
                public static LRScheduler MultiStepLR(Optimizer optimizer, IList<int> milestones, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.MultiStepLR(optimizer, milestones, gamma, last_epoch, verbose);

                }

                /// <summary>
                /// Constructor
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="total_iters">The number of steps that the scheduler decays the learning rate.</param>
                /// <param name="power">The power of the polynomial.</param>
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler PolynomialLR(Optimizer optimizer, int total_iters = 5, int power = 1, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.PolynomialLR(optimizer, total_iters, power, last_epoch, verbose);
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
                public static LRScheduler MultiplicativeLR(Optimizer optimizer, Func<int, double> lr_lambda, int last_epoch = -1, bool verbose = false)
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
                public static LRScheduler ExponentialLR(Optimizer optimizer, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
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
                /// <param name="verbose">If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler ConstantLR(Optimizer optimizer, double factor = 1.0 / 3, int total_iters = 5, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.ConstantLR(optimizer, factor, total_iters, last_epoch, verbose);
                }

                /// <summary>
                /// Uses a list of schedulers, chosen based on the epoch. A sequence of milestones determines when a switch occurs from one scheduler to the next.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer. All sub-schedulers must be bound to the same optimizer.</param>
                /// <param name="schedulers">List of chained schedulers. Should be one more than the number of milestones.</param>
                /// <param name="milestones">List of integers reflecting the milestone points.</param>
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
                /// <returns></returns>
                /// <remarks>The 'verbose' flag should be passed to each individual sub-scheduler.</remarks>
                public static LRScheduler SequentialLR(Optimizer optimizer, IEnumerable<LRScheduler> schedulers, IEnumerable<int> milestones, int last_epoch = -1)
                {
                    return new impl.SequentialLR(optimizer, schedulers, milestones, last_epoch);
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
                /// <param name="verbose">If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler LinearLR(Optimizer optimizer, double start_factor = 1.0 / 3, double end_factor = 5, int total_iters = 5, int last_epoch = -1, bool verbose = false)
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
                public static LRScheduler CosineAnnealingLR(Optimizer optimizer, double T_max, double eta_min = 0, int last_epoch = -1, bool verbose = false)
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
                public static LRScheduler OneCycleLR(Optimizer optimizer,
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
                public static LRScheduler OneCycleLR(Optimizer optimizer,
                    IEnumerable<double> max_lr,
                    int total_steps = -1,
                    int epochs = -1,
                    int steps_per_epoch = -1,
                    double pct_start = 0.3,
                    impl.OneCycleLR.AnnealStrategy anneal_strategy = impl.OneCycleLR.AnnealStrategy.Cos,
                    bool cycle_momentum = true,
                    IEnumerable<double> base_momentum = null,
                    IEnumerable<double> max_momentum = null,
                    double div_factor = 25,
                    double final_div_factor = 1e4,
                    bool three_phase = false,
                    int last_epoch = -1, bool verbose = false)
                {
                    return new impl.OneCycleLR(optimizer, max_lr, total_steps, epochs, steps_per_epoch, pct_start, anneal_strategy, cycle_momentum, base_momentum, max_momentum, div_factor, final_div_factor, three_phase, last_epoch, verbose);
                }

                /// <summary>
                /// Sets the learning rate of each parameter group according to
                /// cyclical learning rate policy(CLR). The policy cycles the learning
                /// rate between two boundaries with a constant frequency, as detailed in
                /// the paper `Cyclical Learning Rates for Training Neural Networks`_.
                /// The distance between the two boundaries can be scaled on a per-iteration
                /// or per-cycle basis.
                ///
                /// Cyclical learning rate policy changes the learning rate after every batch.
                /// `step` should be called after a batch has been used for training.
                ///
                /// This class has three built-in policies, as put forth in the paper:
                ///    * "triangular": A basic triangular cycle without amplitude scaling.
                ///    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
                ///    * "exp_range": A cycle that scales initial amplitude by gamma^(cycle iterations).
                ///    }`
                /// at each cycle iteration.
                /// /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="base_lr">Initial learning rate which is the lower boundary in the cycle</param>
                /// <param name="max_lr">
                /// Upper learning rate boundaries in the cycle for each parameter group.
                /// Functionally, it defines the cycle amplitude(max_lr - base_lr).
                /// The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore 
                /// max_lr may not actually be reached depending on the scaling function.
                /// </param>
                /// <param name="step_size_up">Number of training iterations in the increasing half of a cycle.</param>
                /// <param name="step_size_down">
                /// Number of training iterations in the decreasing half of a cycle.
                /// If step_size_down is -1, it is set to step_size_up.
                /// </param>
                /// <param name="mode">Values correspond to policies detailed above. If scale_fn is non-null, this argument is ignored.</param>
                /// <param name="gamma">Constant in 'exp_range' scaling function</param>
                /// <param name="scale_fn">Custom scaling policy defined by a single argument lambda function. If specified, then 'mode' is ignored.</param>
                /// <param name="scale_mode">Defines whether scale_fn is evaluated on cycle number or cycle iterations(training iterations since start of cycle)</param>
                /// <param name="cycle_momentum">If true, momentum is cycled inversely to learning rate between 'base_momentum' and 'max_momentum'.</param>
                /// <param name="base_momentum">Lower momentum boundaries in the cycle. Note that momentum is cycled inversely to learning rate</param>
                /// <param name="max_momentum">
                /// Upper momentum boundaries in the cycle.
                /// Functionally, it defines the cycle amplitude(max_momentum - base_momentum).
                /// The momentum at any cycle is the difference of max_momentum and some scaling of the amplitude; therefore
                /// base_momentum may not actually be reached depending on the scaling function.
                /// </param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler CyclicLR(Optimizer optimizer,
                            double base_lr,
                            double max_lr,
                            int step_size_up = 2000,
                            int step_size_down = -1,
                            impl.CyclicLR.Mode mode = impl.CyclicLR.Mode.Triangular,
                            double gamma = 1.0,
                            Func<double, double> scale_fn = null,
                            impl.CyclicLR.ScaleMode scale_mode = impl.CyclicLR.ScaleMode.Cycle,
                            bool cycle_momentum = true,
                            double base_momentum = 0.8,
                            double max_momentum = 0.9,
                            int last_epoch = -1,
                            bool verbose = false)
                {
                    return new impl.CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch, verbose);
                }

                /// <summary>
                /// Sets the learning rate of each parameter group according to
                /// cyclical learning rate policy(CLR). The policy cycles the learning
                /// rate between two boundaries with a constant frequency, as detailed in
                /// the paper `Cyclical Learning Rates for Training Neural Networks`_.
                /// The distance between the two boundaries can be scaled on a per-iteration
                /// or per-cycle basis.
                ///
                /// Cyclical learning rate policy changes the learning rate after every batch.
                /// `step` should be called after a batch has been used for training.
                ///
                /// This class has three built-in policies, as put forth in the paper:
                ///    * "triangular": A basic triangular cycle without amplitude scaling.
                ///    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
                ///    * "exp_range": A cycle that scales initial amplitude by gamma^(cycle iterations).
                ///    }`
                /// at each cycle iteration.
                /// /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="base_lr">Initial learning rate which is the lower boundary in the cycle</param>
                /// <param name="max_lr">
                /// Upper learning rate boundaries in the cycle for each parameter group.
                /// Functionally, it defines the cycle amplitude(max_lr - base_lr).
                /// The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore 
                /// max_lr may not actually be reached depending on the scaling function.
                /// </param>
                /// <param name="step_size_up">Number of training iterations in the increasing half of a cycle.</param>
                /// <param name="step_size_down">
                /// Number of training iterations in the decreasing half of a cycle.
                /// If step_size_down is -1, it is set to step_size_up.
                /// </param>
                /// <param name="mode">Values correspond to policies detailed above. If scale_fn is non-null, this argument is ignored.</param>
                /// <param name="gamma">Constant in 'exp_range' scaling function</param>
                /// <param name="scale_fn">Custom scaling policy defined by a single argument lambda function. If specified, then 'mode' is ignored.</param>
                /// <param name="scale_mode">Defines whether scale_fn is evaluated on cycle number or cycle iterations(training iterations since start of cycle)</param>
                /// <param name="cycle_momentum">If true, momentum is cycled inversely to learning rate between 'base_momentum' and 'max_momentum'.</param>
                /// <param name="base_momentum">Lower momentum boundaries in the cycle. Note that momentum is cycled inversely to learning rate</param>
                /// <param name="max_momentum">
                /// Upper momentum boundaries in the cycle.
                /// Functionally, it defines the cycle amplitude(max_momentum - base_momentum).
                /// The momentum at any cycle is the difference of max_momentum and some scaling of the amplitude; therefore
                /// base_momentum may not actually be reached depending on the scaling function.
                /// </param>
                /// <param name="last_epoch">
                /// The index of the last batch. This parameter is used when resuming a training job.Since `step()` should be invoked after each
                /// batch instead of after each epoch, this number represents the total number of *batches* computed, not the total number of epochs computed.
                /// When last_epoch = -1, the schedule is started from the beginning.
                /// </param>
                /// <param name="verbose"> If true, prints a message to stdout for each update. Default: false.</param>
                /// <returns>A scheduler</returns>
                public static LRScheduler CyclicLR(Optimizer optimizer,
                            IEnumerable<double> base_lr,
                            IEnumerable<double> max_lr,
                            int step_size_up = 2000,
                            int step_size_down = -1,
                            impl.CyclicLR.Mode mode = impl.CyclicLR.Mode.Triangular,
                            double gamma = 1.0,
                            Func<double, double> scale_fn = null,
                            impl.CyclicLR.ScaleMode scale_mode = impl.CyclicLR.ScaleMode.Cycle,
                            bool cycle_momentum = true,
                            IEnumerable<double> base_momentum = null,
                            IEnumerable<double> max_momentum = null,
                            int last_epoch = -1,
                            bool verbose = false)
                {
                    return new impl.CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch, verbose);
                }

                /// <summary>
                /// Reduce learning rate when a metric has stopped improving.
                /// Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
                /// This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
                /// </summary>
                /// <param name="optimizer">Wrapped optimizer.</param>
                /// <param name="mode">
                /// One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
                /// in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’
                /// </param>
                /// <param name="factor">Factor by which the learning rate will be reduced. new_lr = lr * factor.</param>
                /// <param name="patience">
                /// Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2,
                /// then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if
                /// the loss still hasn’t improved then. Default: 10.</param>
                /// <param name="threshold">Threshold for measuring the new optimum, to only focus on significant changes. </param>
                /// <param name="threshold_mode">
                /// One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
                /// </param>
                /// <param name="cooldown">Number of epochs to wait before resuming normal operation after lr has been reduced.</param>
                /// <param name="min_lr">A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.</param>
                /// <param name="eps">Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.</param>
                /// <param name="verbose">Indicates whether to print a message to stdout for each update</param>
                public static LRScheduler ReduceLROnPlateau(Optimizer optimizer, string mode = "min", double factor = 0.1, int patience = 10, double threshold = 1e-4, string threshold_mode = "rel", int cooldown = 0, IList<double> min_lr = null, double eps = 1e-8, bool verbose = false)
                {
                    return new impl.ReduceLROnPlateau(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose);
                }
            }
        }
    }
}
