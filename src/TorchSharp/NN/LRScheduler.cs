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
                    internal LRScheduler(ILearningRateController optimizer, int last_epoch = -1, bool verbose = false)
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
                        public LambdaLR(ILearningRateController optimizer, Func<int,double> lr_lambda, int last_epoch = -1, bool verbose = false) : base(optimizer, last_epoch, verbose)
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
            }
        }
    }
}
