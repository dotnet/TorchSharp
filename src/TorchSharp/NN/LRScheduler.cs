// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
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

                        _optimizer.LearningRate = LearningRate;
                        _last_lr = LearningRate;
                        Print();
                    }

                    public double get_learning_rate() => LearningRate;

                    private void Print()
                    {
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
                /// <param name="last_epoch">The index of last epoch. Default: -1.</param>
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
            }
        }
    }
}
