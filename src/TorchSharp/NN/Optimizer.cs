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
            public partial class Optimizer : IDisposable
            {
                /// <summary>
                ///    Class wrapping PyTorch's optimzer object reference.
                /// </summary>
                internal sealed class HType : SafeHandle
                {
                    public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
                    {
                        SetHandle(preexistingHandle);
                    }

                    public override bool IsInvalid => handle == IntPtr.Zero;

                    // This is just for marshalling
                    internal HType() : base(IntPtr.Zero, true)
                    {
                    }

                    [DllImport("LibTorchSharp")]
                    private static extern void THSNN_Optimizer_dispose(HType handle);

                    protected override bool ReleaseHandle()
                    {
                        THSNN_Optimizer_dispose(this);
                        return true;
                    }

                    protected override void Dispose(bool disposing)
                    {
                        if (disposing) {
                            ReleaseHandle();
                        }
                    }
                }

                internal HType handle;

                protected Optimizer(IntPtr handle)
                {
                    if (handle != IntPtr.Zero) {
                        this.handle = new HType(handle, true);
                    }
                }

                ~Optimizer()
                {
                    Dispose(false);
                }

                /// <summary>
                ///   Releases the storage.
                /// </summary>
                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                /// <summary>
                ///   Implements the .NET Dispose pattern.
                /// </summary>
                protected virtual void Dispose(bool disposing)
                {
                    if (disposing && handle != null && !handle.IsInvalid) {
                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_zero_grad(HType module);

                public virtual void zero_grad()
                {
                    THSNN_Optimizer_zero_grad(handle);
                    torch.CheckForErrors();
                }

                [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
                public delegate IntPtr LossClosure();


                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_Optimizer_step(HType module, LossClosure closure);

                public virtual Tensor step(Func<Tensor> closure = null)
                {
                    IntPtr res = (closure == null) ?
                        THSNN_Optimizer_step(handle, null) :
                        THSNN_Optimizer_step(handle, () => {
                            return closure().DecoupleFromNativeHandle();
                        });

                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();

                    return (res == IntPtr.Zero) ? null : new Tensor(res);
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_getParameters(HType module, AllocatePinnedArray allocator);

                public virtual IEnumerable<Parameter> parameters()
                {
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        THSNN_Optimizer_getParameters(handle, pa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                    }
                    return ptrArray.Select(x => new Parameter(x));
                }

                public virtual IEnumerable<ILearningRateController> ParamGroups {
                    get => _parameter_groups;
                }

                protected IList<ParamGroup> _parameter_groups;
            }

            public interface ILearningRateController
            {
                double LearningRate { set; get; }

                double InitialLearningRate { set; get; }
            }

            public interface IMomentum
            {
                double Momentum { get; set; }
            }

            public interface IBetas
            {
                (double, double) Betas { get; set; }
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LBFGS_ctor(IntPtr parameters, int len, double learningRate, long max_iter, long max_eval, double tolerange_grad, double tolerance_change, long history_size);

            public static LBFGS LBFGS(IEnumerable<Parameter> parameters, double lr = 0.01, long max_iter = 20, long? max_eval = null, double tolerange_grad = 1e-5, double tolerance_change = 1e-9, long history_size = 100)
            {
                if (!max_eval.HasValue) max_eval = 5 * max_iter / 4;

                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_LBFGS_ctor(paramsRef, parray.Array.Length, lr, max_iter, max_eval.Value, tolerange_grad, tolerance_change, history_size);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LBFGS(res, lr);
            }

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSProp RMSProp(IEnumerable<Parameter> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                return new RMSProp(parameters, lr, alpha, eps, weight_decay, momentum, centered);
            }

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">Learning rate (default: 1e-2)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSProp RMSProp(IEnumerable<RMSProp.ParamGroup> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                return new RMSProp(parameters, lr, alpha, eps, weight_decay, momentum, centered);
            }

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.The implementation of the L2 penalty follows changes proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static Adam Adam(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new Adam(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
            }

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.The implementation of the L2 penalty follows changes proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static Adam Adam(IEnumerable<Adam.ParamGroup> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new Adam(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
            }

            /// <summary>
            /// Implements AdamW algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static AdamW AdamW(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new AdamW(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
            }

            /// <summary>
            /// Implements AdamW algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize</param>
            /// <param name="lr">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <param name="maximize"></param>
            /// <returns></returns>
            public static AdamW AdamW(IEnumerable<AdamW.ParamGroup> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                return new AdamW(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize);
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
            public static Adagrad Adagrad(IEnumerable<Parameter> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                return new Adagrad(parameters, lr, lr_decay, weight_decay, initial_accumulator_value, eps);
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
            public static Adagrad Adagrad(IEnumerable<Adagrad.ParamGroup> parameters, double lr = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                return new Adagrad(parameters, lr, lr_decay, weight_decay, initial_accumulator_value, eps);
            }

            /// <summary>
            /// Implements Adadelta algorithm.
            ///
            /// It has been proposed in ADADELTA: An Adaptive Learning Rate Method.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="rho">Coefficient used for computing a running average of squared gradients (default: 0.9)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static Adadelta Adadelta(IEnumerable<Parameter> parameters, double lr = 1.0, double rho = 0.9, double eps = 1e-6, double weight_decay = 0)
            {
                return new Adadelta(parameters, lr, rho, eps, weight_decay);
            }

            /// <summary>
            /// Implements Adadelta algorithm.
            ///
            /// It has been proposed in ADADELTA: An Adaptive Learning Rate Method.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="rho">Coefficient used for computing a running average of squared gradients (default: 0.9)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static Adadelta Adadelta(IEnumerable<Adadelta.ParamGroup> parameters, double lr = 1.0, double rho = 0.9, double eps = 1e-6, double weight_decay = 0)
            {
                return new Adadelta(parameters, lr, rho, eps, weight_decay);
            }

            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="named_parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public static NAdam NAdam(IEnumerable<Parameter> named_parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                return new NAdam(named_parameters, lr, beta1, beta2, eps, weight_decay, momentum_decay);
            }

            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public static NAdam NAdam(IEnumerable<NAdam.ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                return new NAdam(parameters, lr, beta1, beta2, eps, weight_decay, momentum_decay);
            }

            /// <summary>
            /// Implements RAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to 'On the variance of the adaptive learning rate and beyond.'
            /// https://arxiv.org/abs/1908.03265
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            public static RAdam RAdam(IEnumerable<Parameter> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                return new RAdam(parameters, lr, beta1, beta2, eps, weight_decay);
            }

            /// <summary>
            /// Implements RAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to 'On the variance of the adaptive learning rate and beyond.'
            /// https://arxiv.org/abs/1908.03265
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            public static RAdam RAdam(IEnumerable<RAdam.ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                return new RAdam(parameters, lr, beta1, beta2, eps, weight_decay);
            }

            /// <summary>
            /// Implements Adamax algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static Adamax Adamax(IEnumerable<Parameter> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                return new Adamax(parameters, lr, beta1, beta2, eps, weight_decay);
            }

            /// <summary>
            /// Implements Adamax algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static Adamax Adamax(IEnumerable<Adamax.ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                return new Adamax(parameters, lr, beta1, beta2, eps, weight_decay);
            }

            /// <summary>
            /// Implements Averaged Stochastic Gradient Descent.
            ///
            /// It has been proposed in Acceleration of stochastic approximation by averaging.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static ASGD ASGD(IEnumerable<Parameter> parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
            {
                return new Modules.ASGD(parameters, lr, lambd, alpha, t0, weight_decay);
            }

            /// <summary>
            /// Implements Averaged Stochastic Gradient Descent.
            ///
            /// It has been proposed in Acceleration of stochastic approximation by averaging.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public static ASGD ASGD(IEnumerable<ParamGroup<ASGD.Options>> parameters, double lr = 1e-3, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
            {
                return new Modules.ASGD(parameters, lr, lambd, alpha, t0, weight_decay);
            }

            /// <summary>
            /// Implements the resilient backpropagation algorithm.
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
            /// Implements the resilient backpropagation algorithm.
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

            /// <summary>
            /// Implements stochastic gradient descent (optionally with momentum).
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
            /// Implements stochastic gradient descent (optionally with momentum).
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

        // Most optimizers are implemented in native code, but a few of them are directly implemented in
        // managed code.

        /// <summary>
        /// Base class to help with a couple of the things that managed-code implementations need.
        /// </summary>

        public class OptimizerHelper : Optimizer
        {
            public OptimizerHelper() : base(IntPtr.Zero)
            {
            }

            public override void zero_grad()
            {
                foreach (var g in _parameter_groups) {

                    foreach (var p in g.Parameters) {

                        using var grad = p.grad();

                        if (grad is null) continue;

                        grad.zero_().Dispose();
                    }
                }
            }

            protected Tensor _step<T>(Action<T> body, Func<Tensor> loss_closure = null) where T:ParamGroup
            {
                Tensor loss = null;

                if (loss_closure != null) {
                    using (var _ = torch.enable_grad())
                        loss = loss_closure();
                }

                using (var _ = torch.no_grad()) {

                    using (var d = torch.NewDisposeScope()) {

                        foreach (var group in _parameter_groups) {

                            body(group as T);

                        }

                        d.DisposeEverything();
                    }
                }

                return loss;
            }

            public override IEnumerable<Parameter> parameters()
            {
                return _parameter_groups.SelectMany(pg => pg.Parameters);
            }

            public virtual void add_param_group(ParamGroup param_group)
            {
                _parameter_groups.Add(param_group);
            }

            protected OptimizerOptions _defaults;
        }

        public class OptimizerOptions
        {
            public double? LearningRate { get; set; }
            public double InitialLearningRate { get; set; }
        }

        public class ParamGroup : ILearningRateController
        {
            public IEnumerable<Parameter> Parameters { get; set; }

            public OptimizerOptions Options { get; set; }

            public double LearningRate { get => Options.LearningRate.Value; set => Options.LearningRate = value; }
            public double InitialLearningRate { get => Options.InitialLearningRate; set => Options.InitialLearningRate = value; }

            public IEnumerable<ILearningRateController> ParamGroups { get => throw new InvalidOperationException("ParamGroups should not be called on a ParamGroup"); }
        }

        public class ParamGroup<TOptions> : ParamGroup where TOptions : OptimizerOptions
        {
            public ParamGroup()
            {
            }

            public ParamGroup(IEnumerable<Parameter> parameters, TOptions options = null)
            {
                base.Parameters = parameters;
                base.Options = options;
            }

            public new TOptions Options { get => (TOptions)base.Options; set => base.Options = value; }

        }

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
                foreach (var (name, state) in _state) {
                    if (state.momentum_buffer is not null) {
                        state.momentum_buffer.Dispose();
                    }
                }
                _state.Clear();
            }

            private class State
            {
                public Tensor momentum_buffer;
            }

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

        public class Adadelta : OptimizerHelper
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="rho">Coefficient used for computing a running average of squared gradients (default: 0.9)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            public Adadelta(IEnumerable<Parameter> parameters, double lr, double rho = 0.9, double eps = 1e-6, double weight_decay = 0)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, rho, eps, weight_decay)
            {
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="parameters">Parameters to optimize.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="rho">Coefficient used for computing a running average of squared gradients (default: 0.9)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            public Adadelta(IEnumerable<ParamGroup> parameters, double lr = 1.0, double rho = 0.9, double eps = 1e-6, double weight_decay = 0)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (rho < 0.0 || rho > 1.0) throw new ArgumentException($"Invalid rho value: {rho}");
                if (eps < 0.0) throw new ArgumentException($"Invalid eps value: {eps}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    rho = rho,
                    eps = eps,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var rho = options.rho.Value;
                    var eps = options.eps.Value;
                    var weight_decay = options.weight_decay.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (grad.is_sparse) throw new ArgumentException("Adadelta does not support sparse gradients");

                        var state = _state[param.handle];

                        var square_avg = state.square_avg;
                        var acc_delta = state.acc_delta;

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        square_avg.mul_(rho).addcmul_(grad, grad, 1 - rho);

                        var std = square_avg.add(eps).sqrt_();
                        var delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad);

                        param.add_(delta, alpha: -lr);
                        acc_delta.mul_(rho).addcmul_(delta, delta, 1 - rho);
                    }
                },closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.square_avg.Dispose();
                    state.acc_delta.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor square_avg;
                public Tensor acc_delta;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.rho.HasValue) opt.rho = def.rho;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.square_avg = torch.zeros_like(p);
                    state.acc_delta = torch.zeros_like(p);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? rho;
                public double? eps;
                public double? weight_decay;
            }

            public class ParamGroup : ParamGroup<Options>
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1.0, double rho = 0.9, double eps = 1e-6, double weight_decay = 0)
                    : base(parameters, new Adadelta.Options { LearningRate = lr, rho = rho, eps = eps, weight_decay = weight_decay })
                {
                }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class Adamax : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements Adamax algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public Adamax(IEnumerable<Parameter> parameters, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay)
            {
            }

            /// <summary>
            /// Implements Adamax algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public Adamax(IEnumerable<ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (beta1 < 0.0 || beta1 > 1.0) throw new ArgumentException($"Invalid beta1 value: {beta1}");
                if (beta2 < 0.0 || beta2 > 1.0) throw new ArgumentException($"Invalid beta2 value: {beta2}");
                if (eps < 0.0) throw new ArgumentException($"Invalid eps value: {eps}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    eps = eps,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var eps = options.eps.Value;
                    var weight_decay = options.weight_decay.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (grad.is_sparse) throw new ArgumentException("Adamax does not support sparse gradients");

                        var state = _state[param.handle];

                        state.step += 1;

                        var exp_avg = state.exp_avg;
                        var exp_inf = state.exp_inf;

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);

                        var norm_buf = torch.cat(new Tensor[] {
                                        exp_inf.mul_(beta2).unsqueeze(0),
                                        grad.abs().add_(eps).unsqueeze_(0)
                                    }, 0);

                        torch.amax(norm_buf, new long[] { 0 }, false, exp_inf);

                        var clr = lr / (1 - Math.Pow(beta1, state.step));
                        param.addcdiv_(exp_avg, exp_inf, value: -clr);
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_inf.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor exp_avg;
                public Tensor exp_inf;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.beta1.HasValue) opt.beta1 = def.beta1;
                if (!opt.beta2.HasValue) opt.beta2 = def.beta2;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_inf = torch.zeros_like(p);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? eps;
                public double? weight_decay;
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1.0, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
                    : base(parameters, new Adamax.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay })
                {
                }

                public (double, double) Betas {
                    get => (Options.beta1.Value, Options.beta2.Value);
                    set { Options.beta1 = value.Item1; Options.beta2 = value.Item2; }
                }
            }

            public (double, double) Betas {
                get => ((_defaults as Options).beta1.Value, (_defaults as Options).beta2.Value);
                set { (_defaults as Options).beta1 = value.Item1; (_defaults as Options).beta2 = value.Item2; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class NAdam : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            public NAdam(IEnumerable<Parameter> parameters, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay, momentum_decay)
            {
            }

            /// <summary>
            /// Implements NAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to Incorporating Nesterov Momentum into Adam.
            /// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum_decay">Momentum decay</param>
            /// <returns></returns>
            public NAdam(IEnumerable<ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (beta1 < 0.0 || beta1 > 1.0) throw new ArgumentException($"Invalid beta1 value: {beta1}");
                if (beta2 < 0.0 || beta2 > 1.0) throw new ArgumentException($"Invalid beta2 value: {beta2}");
                if (eps < 0.0) throw new ArgumentException($"Invalid eps value: {eps}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");
                if (momentum_decay < 0.0) throw new ArgumentException($"Invalid momentum_decay value: {momentum_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    eps = eps,
                    weight_decay = weight_decay,
                    momentum_decay = momentum_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var eps = options.eps.Value;
                    var weight_decay = options.weight_decay.Value;
                    var momentum_decay = options.momentum_decay.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        var state = _state[param.handle];

                        state.step += 1;

                        var exp_avg = state.exp_avg;
                        var exp_avg_sq = state.exp_avg_sq;

                        var bias_correction2 = 1 - Math.Pow(beta2, state.step);

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        var mu = beta1 * (1.0 - 0.5 * Math.Pow(0.96, state.step * momentum_decay));
                        var mu_next = beta1 * (1.0 - 0.5 * Math.Pow(0.96, (state.step + 1) * momentum_decay));

                        var mu_product = state.mu_product * mu;
                        var mu_product_next = mu_product * mu * mu_next;

                        exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value: 1 - beta2);

                        var denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps);

                        param.addcdiv_(grad, denom, value: -lr * (1 - mu) / (1 - mu_product));
                        param.addcdiv_(exp_avg, denom, value: -lr * mu_next / (1 - mu_product_next));

                        state.mu_product = mu_product;
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_avg_sq.Dispose();
                }
            }

            private class State
            {
                public int step;
                public double mu_product;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.beta1.HasValue) opt.beta1 = def.beta1;
                if (!opt.beta2.HasValue) opt.beta2 = def.beta2;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.momentum_decay.HasValue) opt.momentum_decay = def.momentum_decay;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? eps;
                public double? weight_decay;
                public double? momentum_decay;
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1.0, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, double momentum_decay = 4e-3)
                    : base(parameters, new NAdam.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay, momentum_decay = momentum_decay })
                {
                }

                public (double, double) Betas {
                    get => (Options.beta1.Value, Options.beta2.Value);
                    set { Options.beta1 = value.Item1; Options.beta2 = value.Item2; }
                }
            }

            public (double, double) Betas {
                get => ((_defaults as Options).beta1.Value, (_defaults as Options).beta2.Value);
                set { (_defaults as Options).beta1 = value.Item1; (_defaults as Options).beta2 = value.Item2; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class RAdam : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements RAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to 'On the variance of the adaptive learning rate and beyond.'
            /// https://arxiv.org/abs/1908.03265
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public RAdam(IEnumerable<Parameter> parameters, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay)
            {
            }

            /// <summary>
            /// Implements RAdam algorithm.
            ///
            /// For further details regarding the algorithm we refer to 'On the variance of the adaptive learning rate and beyond.'
            /// https://arxiv.org/abs/1908.03265
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability, i.e. avoid division-by-zero (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public RAdam(IEnumerable<ParamGroup> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (beta1 < 0.0 || beta1 > 1.0) throw new ArgumentException($"Invalid beta1 value: {beta1}");
                if (beta2 < 0.0 || beta2 > 1.0) throw new ArgumentException($"Invalid beta2 value: {beta2}");
                if (eps < 0.0) throw new ArgumentException($"Invalid eps value: {eps}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    eps = eps,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var eps = options.eps.Value;
                    var weight_decay = options.weight_decay.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        var state = _state[param.handle];

                        state.step += 1;

                        var exp_avg = state.exp_avg;
                        var exp_avg_sq = state.exp_avg_sq;

                        var bias_correction1 = 1 - Math.Pow(beta1, state.step);
                        var bias_correction2 = 1 - Math.Pow(beta2, state.step);

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value: 1 - beta2);

                        var bias_corrected_exp_avg = exp_avg / bias_correction1;

                        var rho_inf = 2 / (1 - beta2) - 1;
                        var rho_t = rho_inf - 2 * state.step * Math.Pow(beta2, state.step) / bias_correction2;

                        var t6 = bias_corrected_exp_avg * lr;

                        if (rho_t > 5) {
                            var rect = Math.Sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t));
                            var adaptive_lr = Math.Sqrt(bias_correction2) / exp_avg_sq.sqrt().add_(eps);

                            param.add_(t6 * lr * adaptive_lr * rect, alpha: -1.0);
                        } else {
                            param.add_(t6, alpha: -1.0);
                        }
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_avg_sq.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.beta1.HasValue) opt.beta1 = def.beta1;
                if (!opt.beta2.HasValue) opt.beta2 = def.beta2;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? eps;
                public double? weight_decay;
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1.0, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0)
                    : base(parameters, new RAdam.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay })
                {
                }

                public (double, double) Betas {
                    get => (Options.beta1.Value, Options.beta2.Value);
                    set { Options.beta1 = value.Item1; Options.beta2 = value.Item2; }
                }
            }

            public (double, double) Betas {
                get => ((_defaults as Options).beta1.Value, (_defaults as Options).beta2.Value);
                set { (_defaults as Options).beta1 = value.Item1; (_defaults as Options).beta2 = value.Item2; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class ASGD : OptimizerHelper
        {
            /// <summary>
            /// Implements ASGD algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public ASGD(IEnumerable<Parameter> parameters, double lr = 0.01, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
                : this(new ParamGroup[] { new() { Parameters = parameters } }, lr, lambd, alpha, t0, weight_decay)
            {
            }

            /// <summary>
            /// Implements ASGD algorithm (a variant of Adam based on infinity norm).
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr ">Learning rate</param>
            /// <param name="lambd">Decay term (default: 1e-4)</param>
            /// <param name="alpha">Power for eta update (default: 0.75)</param>
            /// <param name="t0">Point at which to start averaging (default: 1e6)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <returns></returns>
            public ASGD(IEnumerable<ParamGroup<Options>> parameters, double lr = 0.01, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
            {
                if (lr < 0.0) throw new ArgumentException($"Invalid learning rate: {lr}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    lambd = lambd,
                    alpha = alpha,
                    t0 = t0,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var lambd = options.lambd.Value;
                    var alpha = options.alpha.Value;
                    var weight_decay = options.weight_decay.Value;
                    var t0 = options.t0.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var grad = param.grad();

                        if (grad is null) continue;

                        if (grad.is_sparse) throw new ArgumentException("ASGD does not support sparse gradients");

                        var state = _state[param.handle];

                        state.step += 1;

                        grad = (weight_decay != 0)
                            ? grad.add(param, alpha: weight_decay)
                            : grad.alias();

                        param.mul_(1 - lambd * state.eta);
                        param.add_(grad, alpha: -state.eta);

                        if (state.mu != 1) {
                            state.ax.add_(param.sub(state.ax).mul(state.mu));
                        } else {
                            state.ax.copy_(param);
                        }

                        state.eta = lr / Math.Pow((1 + lambd * lr * state.step), alpha);
                        state.mu = 1 / Math.Max(1, state.step - t0);
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.ax.Dispose();
                }
                _state.Clear();
            }

            private class State
            {
                public int step;
                public double eta;
                public double mu;
                public Tensor ax;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.lambd.HasValue) opt.lambd = def.lambd;
                if (!opt.alpha.HasValue) opt.alpha = def.alpha;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.t0.HasValue) opt.t0 = def.t0;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.eta = param_group.LearningRate;
                    state.mu = 1;
                    state.ax = torch.zeros_like(p);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? lambd;
                public double? alpha;
                public double? weight_decay;
                public double? t0;
            }

            public class ParamGroup : ParamGroup<Options>
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 0.01, double lambd = 1e-4, double alpha = 0.75, double t0 = 1e6, double weight_decay = 0)
                    : base(parameters, new ASGD.Options { LearningRate = lr, lambd = lambd, alpha = alpha, t0 = t0, weight_decay = weight_decay })
                {
                }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

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
                    state.prev.Dispose();
                    state.step_size.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor prev;
                public Tensor step_size;
            }

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

                        var state = _state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        var clr = lr / (1 + (state.step - 1) * lr_decay);

                        if (grad.is_sparse) {
                            throw new NotImplementedException("Adagrad optimization over sparse parameters");
                        } else if (torch.is_complex(param)) {
                            throw new NotImplementedException("Adagrad optimization over complex parameters");
                        } else {
                            state.sum.addcmul_(grad, grad, value: 1);
                            var std = state.sum.sqrt().add_(eps);
                            param.addcdiv_(grad, std, value: -clr);
                        }
                    }


                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (name, state) in _state) {
                    state.sum.Dispose();
                }
            }

            private class State
            {
                public int step;
                public Tensor sum;
            }

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
                    state.sum = torch.full_like(p, init_value);
                }
            }

            public class Options : OptimizerOptions
            {
                public double? lr_decay;
                public double? initial_accumulator_value;
                public double? eps;
                public double? weight_decay;
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

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class Adam : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.The implementation of the L2 penalty follows changes proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="beta1">First coefficient used for computing running averages of gradient and its square</param>
            /// <param name="beta2">Second coefficient used for computing running averages of gradient and its square</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing</param>
            /// <returns></returns>
            public Adam(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize)
            {
            }

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.The implementation of the L2 penalty follows changes proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="beta1">First coefficient used for computing running averages of gradient and its square</param>
            /// <param name="beta2">Second coefficient used for computing running averages of gradient and its square</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing</param>
            /// <returns></returns>
            public Adam(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (beta1 < 0 || beta1 >= 1.0) throw new ArgumentException($"Invalid beta1: {beta1}");
                if (beta2 < 0 || beta2 >= 1.0) throw new ArgumentException($"Invalid beta2: {beta2}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    maximize = maximize,
                    eps = eps,
                    amsgrad = amsgrad,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var weight_decay = options.weight_decay.Value;
                    var amsgrad = options.amsgrad.Value;
                    var maximize = options.maximize.Value;
                    var eps = options.eps.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = _state[param.handle];

                        var grad = (maximize) ? -param.grad() : param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        var bias_correction1 = 1 - Math.Pow(beta1, state.step);
                        var bias_correction2 = 1 - Math.Pow(beta2, state.step);

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        state.exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);
                        // When complex types are supported:
                        //state.exp_avg_sq.mul_(_beta2).addcmul_(grad, grad.conj(), value: 1 - _beta2)
                        state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value: 1 - beta2);

                        Tensor denom = null;
                        if (amsgrad) {
                            var t0 = state.max_exp_avg_sq;
                            state.max_exp_avg_sq = torch.maximum(t0, state.exp_avg_sq).DetatchFromDisposeScope();
                            t0.Dispose();
                            denom = (state.max_exp_avg_sq.sqrt() / Math.Sqrt(bias_correction2)).add_(eps);
                        } else {
                            denom = (state.exp_avg_sq.sqrt() / Math.Sqrt(bias_correction2)).add_(eps);
                        }

                        var step_size = lr / bias_correction1;
                        param.addcdiv_(state.exp_avg, denom, value: -step_size);
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (_, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_avg_sq.Dispose();
                    if (state.max_exp_avg_sq is not null) {
                        state.max_exp_avg_sq.Dispose();
                    }
                }
            }

            private class State
            {
                public int step;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
                public Tensor max_exp_avg_sq;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.beta1.HasValue) opt.beta1 = def.beta1;
                if (!opt.beta2.HasValue) opt.beta2 = def.beta2;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.amsgrad.HasValue) opt.amsgrad = def.amsgrad;
                if (!opt.maximize.HasValue) opt.maximize = def.maximize;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                    if (opt.amsgrad.Value) {
                        state.max_exp_avg_sq = torch.zeros_like(p);
                    }
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? weight_decay;
                public double? eps;
                public bool? amsgrad;
                public bool? maximize;
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
                    : base(parameters, new Adam.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad, maximize = maximize })
                {
                }

                public (double, double) Betas {
                    get => (Options.beta1.Value, Options.beta2.Value);
                    set { Options.beta1 = value.Item1; Options.beta2 = value.Item2; }
                }
            }

            public (double, double) Betas {
                get => ((_defaults as Options).beta1.Value, (_defaults as Options).beta2.Value);
                set { (_defaults as Options).beta1 = value.Item1; (_defaults as Options).beta2 = value.Item2; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class AdamW : OptimizerHelper, IBetas
        {
            /// <summary>
            /// Implements AdamW algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="beta1">First coefficient used for computing running averages of gradient and its square</param>
            /// <param name="beta2">Second coefficient used for computing running averages of gradient and its square</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing</param>
            /// <returns></returns>
            public AdamW(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize)
            {
            }

            /// <summary>
            /// Implements AdamW algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the <b>named</b> parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="beta1">First coefficient used for computing running averages of gradient and its square</param>
            /// <param name="beta2">Second coefficient used for computing running averages of gradient and its square</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm</param>
            /// <param name="maximize">Maximize the params based on the objective, instead of minimizing</param>
            /// <returns></returns>
            public AdamW(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (beta1 < 0 || beta1 >= 1.0) throw new ArgumentException($"Invalid beta1: {beta1}");
                if (beta2 < 0 || beta2 >= 1.0) throw new ArgumentException($"Invalid beta2: {beta2}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    beta1 = beta1,
                    beta2 = beta2,
                    maximize = maximize,
                    eps = eps,
                    amsgrad = amsgrad,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var beta1 = options.beta1.Value;
                    var beta2 = options.beta2.Value;
                    var weight_decay = options.weight_decay.Value;
                    var amsgrad = options.amsgrad.Value;
                    var maximize = options.maximize.Value;
                    var eps = options.eps.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = _state[param.handle];

                        var grad = (maximize) ? -param.grad() : param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        param.mul_(1 - lr * weight_decay);

                        var bias_correction1 = 1 - Math.Pow(beta1, state.step);
                        var bias_correction2 = 1 - Math.Pow(beta2, state.step);

                        state.exp_avg.mul_(beta1).add_(grad, alpha: 1 - beta1);
                        state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value: 1 - beta2);

                        Tensor denom = null;
                        if (amsgrad) {
                            var t0 = state.max_exp_avg_sq;
                            state.max_exp_avg_sq = torch.maximum(t0, state.exp_avg_sq).DetatchFromDisposeScope();
                            t0.Dispose();
                            denom = (state.max_exp_avg_sq.sqrt() / Math.Sqrt(bias_correction2)).add_(eps);
                        } else {
                            denom = (state.exp_avg_sq.sqrt() / Math.Sqrt(bias_correction2)).add_(eps);
                        }

                        var step_size = lr / bias_correction1;
                        param.addcdiv_(state.exp_avg, denom, value: -step_size);
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (_, state) in _state) {
                    state.exp_avg.Dispose();
                    state.exp_avg_sq.Dispose();
                    if (state.max_exp_avg_sq is not null) {
                        state.max_exp_avg_sq.Dispose();
                    }
                }
            }

            private class State
            {
                public int step;
                public Tensor exp_avg;
                public Tensor exp_avg_sq;
                public Tensor max_exp_avg_sq;
            }

            public override void add_param_group(Modules.ParamGroup param_group)
            {
                var def = _defaults as Options;
                if (param_group.Options is null) {
                    param_group.Options = new Options();
                }

                var opt = param_group.Options as Options;

                // Make sure all the options are set.
                if (!opt.LearningRate.HasValue) opt.LearningRate = def.LearningRate;
                if (!opt.beta1.HasValue) opt.beta1 = def.beta1;
                if (!opt.beta2.HasValue) opt.beta2 = def.beta2;
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.amsgrad.HasValue) opt.amsgrad = def.amsgrad;
                if (!opt.maximize.HasValue) opt.maximize = def.maximize;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.step = 0;
                    state.exp_avg = torch.zeros_like(p);
                    state.exp_avg_sq = torch.zeros_like(p);
                    if (opt.amsgrad.Value) {
                        state.max_exp_avg_sq = torch.zeros_like(p);
                    }
                }
            }

            public class Options : OptimizerOptions
            {
                public double? beta1;
                public double? beta2;
                public double? weight_decay;
                public double? eps;
                public bool? amsgrad;
                public bool? maximize;
            }

            public class ParamGroup : ParamGroup<Options>, IBetas
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false, bool maximize = false)
                    : base(parameters, new AdamW.Options { LearningRate = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad, maximize = maximize })
                {
                }

                public (double, double) Betas {
                    get => (Options.beta1.Value, Options.beta2.Value);
                    set { Options.beta1 = value.Item1; Options.beta2 = value.Item2; }
                }
            }

            public (double, double) Betas {
                get => ((_defaults as Options).beta1.Value, (_defaults as Options).beta2.Value);
                set { (_defaults as Options).beta1 = value.Item1; (_defaults as Options).beta2 = value.Item2; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();
        }

        public class RMSProp : OptimizerHelper, IMomentum
        {

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="alpha">Smoothing constant.</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="centered">if ``True``, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public RMSProp(IEnumerable<Parameter> parameters, double lr = 1e-3, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0.0, bool centered = false)
                : this(new ParamGroup[] { new ParamGroup { Parameters = parameters } }, lr, alpha, eps, weight_decay, momentum, centered)
            {
            }

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Parameters to optimize. This optimizer requires the parameters collection.</param>
            /// <param name="lr">Learning rate</param>
            /// <param name="alpha">Smoothing constant.</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="centered">if ``True``, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public RMSProp(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0.0, bool centered = false)
            {
                if (lr < 0) throw new ArgumentException($"Invalid learning rate: {lr}");
                if (eps < 0) throw new ArgumentException($"Invalid ε: {eps}");
                if (alpha < 0) throw new ArgumentException($"Invalid alpha: {alpha}");
                if (momentum < 0.0) throw new ArgumentException($"Invalid momentum value: {momentum}");
                if (weight_decay < 0.0) throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");

                var options = new Options {
                    LearningRate = lr,
                    InitialLearningRate = lr,
                    eps = eps,
                    alpha = alpha,
                    momentum = momentum,
                    centered = centered,
                    weight_decay = weight_decay
                };

                _defaults = options;
                _parameter_groups = new List<Modules.ParamGroup>();

                foreach (var g in parameters) {
                    add_param_group(g);
                }
            }

            public override Tensor step(Func<Tensor> closure = null)
            {
                return _step<ParamGroup>(group => {

                    var options = group.Options as Options;
                    var momentum = options.momentum.Value;
                    var alpha = options.alpha.Value;
                    var weight_decay = options.weight_decay.Value;
                    var centered = options.centered.Value;
                    var eps = options.eps.Value;
                    var lr = options.LearningRate.Value;

                    foreach (var param in group.Parameters) {

                        var state = _state[param.handle];

                        var grad = param.grad();

                        if (grad is null) continue;

                        state.step += 1;

                        if (weight_decay != 0) {
                            grad = grad.add(param, alpha: weight_decay);
                        }

                        state.square_avg.mul_(alpha).addcmul_(grad, grad, value: 1 - alpha);

                        Tensor avg = null;

                        if (centered) {
                            var grad_avg = state.grad_avg;
                            grad_avg.mul_(alpha).add_(grad, alpha: 1 - alpha);
                            avg = state.square_avg.addcmul(grad_avg, grad_avg, value: -1).sqrt_().add_(eps);
                        } else {
                            avg = state.square_avg.sqrt().add_(eps);
                        }

                        if (momentum > 0) {
                            var buf = state.momentum_buffer;
                            buf.mul_(momentum).addcdiv_(grad, avg);
                            param.add_(buf, alpha: -lr);
                        } else {
                            param.addcdiv_(grad, avg, -lr);
                        }
                    }
                }, closure);
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                foreach (var (_, state) in _state) {
                    state.momentum_buffer.Dispose();
                    state.square_avg.Dispose();
                    state.grad_avg.Dispose();
                }
                _state.Clear();
            }

            private class State
            {
                public int step;
                public Tensor square_avg;
                public Tensor momentum_buffer;
                public Tensor grad_avg;
            }



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
                if (!opt.eps.HasValue) opt.eps = def.eps;
                if (!opt.alpha.HasValue) opt.alpha = def.alpha;
                if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
                if (!opt.centered.HasValue) opt.centered = def.centered;

                opt.InitialLearningRate = opt.LearningRate.Value;

                _parameter_groups.Add(param_group);

                foreach (var p in param_group.Parameters) {
                    var state = new State();
                    _state[p.Handle] = state;
                    state.square_avg = torch.zeros_like(p);
                    state.grad_avg = torch.zeros_like(p);
                    state.momentum_buffer = torch.zeros_like(p);
                }
            }
            public class Options : Modules.OptimizerOptions
            {
                public double? momentum;
                public double? alpha;
                public double? eps;
                public double? weight_decay;
                public bool? centered;
            }

            public class ParamGroup : ParamGroup<Options>, IMomentum
            {
                public ParamGroup() { }

                public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) { }

                public ParamGroup(IEnumerable<Parameter> parameters, double lr = 1e-3, double eps = 1e-8, double alpha = 0.99, double weight_decay = 0, double momentum = 0.0, bool centered = false)
                    : base(parameters, new RMSProp.Options { LearningRate = lr, eps = eps, alpha = alpha, weight_decay = weight_decay, momentum = momentum, centered = centered })
                {
                }

                public double Momentum { get => Options.momentum.Value; set => Options.momentum = value; }
            }

            private Dictionary<IntPtr, State> _state = new Dictionary<IntPtr, State>();

            public double Momentum { get => (_defaults as Options).momentum.Value; set => (_defaults as Options).momentum = value; }
        }

        // The following optimizers are wrappers for the native code implementations.
        //
        // LBGFS does not allow for param groups, so there's no need to use anything but the native implementation.

        public class LBFGS : Optimizer, ILearningRateController
        {
            public LBFGS(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
                InitialLearningRate = lr;
                _paramGroups = new ParamGroup[] { new ParamGroup { Parameters = this.parameters(), Options = new() { LearningRate = lr, InitialLearningRate = lr } } };
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_LBFGS_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_LBFGS_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public double InitialLearningRate { get; set; }

            public override IEnumerable<ILearningRateController> ParamGroups { get => _paramGroups; }

            
            public override Tensor step(Func<Tensor> closure = null)
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

