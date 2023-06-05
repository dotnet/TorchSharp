// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    public static partial class torch
    {
        public static partial class optim
        {
            /// <summary>
            /// Implements the L-BFGS algorithm, heavily inspired by `minFunc`
            ///
            /// https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="max_iter">Maximal number of iterations per optimization step</param>
            /// <param name="max_eval">Maximal number of function evaluations per optimization step</param>
            /// <param name="tolerange_grad">Termination tolerance on first order optimality</param>
            /// <param name="tolerance_change">Termination tolerance on function value/parameter changes</param>
            /// <param name="history_size">Update history size</param>
            public static LBFGS LBFGS(IEnumerable<(string name, Parameter parameter)> parameters, double lr = 0.01, long max_iter = 20, long? max_eval = null, double tolerange_grad = 1e-5, double tolerance_change = 1e-9, long history_size = 100)
            {
                return LBFGS(parameters.Select(np => np.parameter), lr, max_iter, max_eval.Value, tolerange_grad, tolerance_change, history_size);
            }

            /// <summary>
            /// Implements the L-BFGS algorithm, heavily inspired by `minFunc`
            ///
            /// https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="max_iter">Maximal number of iterations per optimization step</param>
            /// <param name="max_eval">Maximal number of function evaluations per optimization step</param>
            /// <param name="tolerange_grad">Termination tolerance on first order optimality</param>
            /// <param name="tolerance_change">Termination tolerance on function value/parameter changes</param>
            /// <param name="history_size">Update history size</param>
            public static LBFGS LBFGS(IEnumerable<Parameter> parameters, double lr = 0.01, long max_iter = 20, long? max_eval = null, double tolerange_grad = 1e-5, double tolerance_change = 1e-9, long history_size = 100)
            {
                if (!max_eval.HasValue) max_eval = 5 * max_iter / 4;

                using var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_LBFGS_ctor(paramsRef, parray.Array.Length, lr, max_iter, max_eval.Value, tolerange_grad, tolerance_change, history_size);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LBFGS(res, lr);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        // LBGFS does not allow for param groups, so there's no need to use anything but the native implementation.
        public class LBFGS : Optimizer, ILearningRateController
        {
            /// <summary>
            /// Implements L-BFGS algorithm, heavily inspired by `minFunc`
            ///
            /// </summary>
            /// <param name="handle"></param>
            /// <param name="lr"></param>
            public LBFGS(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
                InitialLearningRate = lr;
                _paramGroups = new ParamGroup[] { new ParamGroup { Parameters = this.parameters(), Options = new() { LearningRate = lr, InitialLearningRate = lr } } };
            }

            public double LearningRate {
                get { return _rate; }
                set { THSNN_LBFGS_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            public override IEnumerable<ILearningRateController> ParamGroups { get => _paramGroups; }


            /// <summary>
            /// Performs a single optimization step (parameter update).
            /// </summary>
            /// <param name="closure">A closure that reevaluates the model and returns the loss. Optional for most optimizers.</param>
            /// <returns></returns>
            public override Tensor step(Func<Tensor> closure)
            {
                if (closure == null)
                    throw new ArgumentNullException("'closure' must be non-null when using the LBFGS optimizer. See: https://pytorch.org/docs/1.9.0/optim.html");
                return base.step(closure);
            }

            public double _rate;
            private ParamGroup[] _paramGroups;
        }
    }
}
