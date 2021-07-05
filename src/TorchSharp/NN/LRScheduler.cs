// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
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
                    public abstract void step();
                    public double LearningRate => _optimizer.LearningRate;

                    protected ILearningRateController _optimizer;
                }

                public static partial class impl
                {
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
                        public StepLR(ILearningRateController optimizer, uint step_size, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _optimizer = optimizer;
                            _initial = optimizer.LearningRate;
                            _step = step_size;
                            _gamma = gamma;
                            _last = last_epoch;
                            _verbose = verbose;
                        }

                        public override void step()
                        {
                            _epoch += 1;

                            if (_last == -1) {
                                _optimizer.LearningRate = _initial;
                                if (_verbose) Console.WriteLine($"Learning rate updated to: {_initial}");
                            } else if (_epoch % _step == 0 && _epoch <= _last) {
                                var lr = _optimizer.LearningRate;
                                lr *= _gamma;
                                _optimizer.LearningRate = lr;
                                if (_verbose) Console.WriteLine($"Learning rate updated to: {lr}");
                            }
                        }

                        private double _initial;
                        private uint _step;
                        private int _epoch;
                        private double _gamma;
                        private int _last;
                        private bool _verbose;
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
                        public ExponentialLR(ILearningRateController optimizer, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                        {
                            if (optimizer == null) throw new ArgumentNullException("optimizer");
                            _optimizer = optimizer;
                            _initial = optimizer.LearningRate;
                            _gamma = gamma;
                            _last = last_epoch;
                            _verbose = verbose;
                        }

                        public override void step()
                        {
                            _epoch += 1;

                            if (_last == -1) {
                                _optimizer.LearningRate = _initial;
                                if (_verbose) Console.WriteLine($"Learning rate updated to: {_initial}");
                            } else if (_epoch <= _last) {
                                var lr = _optimizer.LearningRate;
                                lr *= _gamma;
                                _optimizer.LearningRate = lr;
                                if (_verbose) Console.WriteLine($"Learning rate updated to: {lr}");
                            }
                        }

                        private double _initial;
                        private int _epoch;
                        private double _gamma;
                        private int _last;
                        private bool _verbose;
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
                public static LRScheduler StepLR(ILearningRateController optimizer, uint step_size, double gamma = 0.1, int last_epoch = -1, bool verbose = false)
                {
                    return new impl.StepLR(optimizer, step_size, gamma, last_epoch, verbose);
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
