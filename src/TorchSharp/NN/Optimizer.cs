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
                    this.handle = new HType(handle, true);
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
                protected void Dispose(bool disposing)
                {
                    if (disposing) {
                        handle.Dispose();
                        handle.SetHandleAsInvalid();
                    }
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_zero_grad(HType module);

                public void zero_grad()
                {
                    THSNN_Optimizer_zero_grad(handle);
                    torch.CheckForErrors();
                }

                [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
                public delegate IntPtr LossClosure();


                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_step(HType module, LossClosure closure);

                public virtual void step(Func<Tensor> closure = null)
                {
                    if (closure == null) {
                        THSNN_Optimizer_step(handle, null);
                    } else {
                        THSNN_Optimizer_step(handle, () => {
                            var res = closure();
                            GC.SuppressFinalize(res);
                            return res.handle;
                        });
                    }
                    torch.CheckForErrors();
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSNN_Optimizer_getParameters(HType module, AllocatePinnedArray allocator);

                public IEnumerable<Tensor> parameters()
                {
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        THSNN_Optimizer_getParameters(handle, pa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                    }
                    return ptrArray.Select(x => new Tensor(x));
                }
            }

            public interface ILearningRateController
            {
                double LearningRate { set; get; }
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LBFGS_ctor(IntPtr parameters, int len, double learningRate, long max_iter, long max_eval, double tolerange_grad, double tolerance_change, long history_size);

            public static LBFGSOptimizer LBFGS(IEnumerable<Tensor> parameters, double learningRate = 0.01, long max_iter = 20, long? max_eval = null, double tolerange_grad = 1e-5, double tolerance_change = 1e-9, long history_size = 100)
            {
                if (!max_eval.HasValue) max_eval = 5 * max_iter / 4;

                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_LBFGS_ctor(paramsRef, parray.Array.Length, learningRate, max_iter, max_eval.Value, tolerange_grad, tolerance_change, history_size);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LBFGSOptimizer(res, learningRate);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_RMSprop_ctor(IntPtr parameters, int len, double learningRate, double alpha, double eps, double weight_decay, double momemtum, bool centered);

            /// <summary>
            /// Implements RMSprop algorithm.
            ///
            /// Proposed by G.Hinton in his course.
            /// </summary>
            /// <param name="parameters">Prameters to optimize</param>
            /// <param name="learningRate">Learning rate (default: 1e-2)</param>
            /// <param name="alpha">Smoothing constant (default: 0.99)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="centered">if true, compute the centered RMSProp, the gradient is normalized by an estimation of its variance</param>
            /// <returns></returns>
            public static RMSPropOptimizer RMSProp(IEnumerable<Tensor> parameters, double learningRate = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0, double momentum = 0, bool centered = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_RMSprop_ctor(paramsRef, parray.Array.Length, learningRate, alpha, eps, weight_decay, momentum, centered);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new RMSPropOptimizer(res, learningRate);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Adam_ctor(IntPtr parameters, int len, double learningRate, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization.The implementation of the L2 penalty follows changes proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Prameters to optimize</param>
            /// <param name="learningRate">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <returns></returns>
            public static AdamOptimizer Adam(IEnumerable<Tensor> parameters, double learningRate = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_Adam_ctor(paramsRef, parray.Array.Length, learningRate, beta1, beta2, eps, weight_decay, amsgrad);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdamOptimizer(res, learningRate);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdamW_ctor(IntPtr parameters, int len, double learningRate, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);

            /// <summary>
            /// Implements Adam algorithm.
            ///
            /// It has been proposed in Adam: A Method for Stochastic Optimization. The AdamW variant was proposed in Decoupled Weight Decay Regularization.
            /// </summary>
            /// <param name="parameters">Prameters to optimize</param>
            /// <param name="learningRate">learning rate (default: 1e-3)</param>
            /// <param name="beta1">Coefficient used for computing running averages of gradient and its square (default: 0.9)</param>
            /// <param name="beta2">Coefficient used for computing running averages of gradient and its square (default: 0.999)</param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-8)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="amsgrad">Whether to use the AMSGrad variant of this algorithm. (default: False)</param>
            /// <returns></returns>
            public static AdamWOptimizer AdamW(IEnumerable<Tensor> parameters, double learningRate = 1e-3, double beta1 = 0.9, double beta2 = 0.99, double eps = 1e-8, double weight_decay = 0, bool amsgrad = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_AdamW_ctor(paramsRef, parray.Array.Length, learningRate, beta1, beta2, eps, weight_decay, amsgrad);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdamWOptimizer(res, learningRate);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Adagrad_ctor(IntPtr parameters, int len, double learningRate, double lr_decay, double weight_decay, double initial_accumulator_value, double eps);

            /// <summary>
            /// Implements Adagrad algorithm.
            ///
            /// It has been proposed in Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
            /// </summary>
            /// <param name="parameters">Prameters to optimize</param>
            /// <param name="learningRate">learning rate (default: 1e-2)</param>
            /// <param name="lr_decay">learning rate decay (default: 0)</param>
            /// <param name="weight_decay">weight decay (L2 penalty) (default: 0)</param>
            /// <param name="initial_accumulator_value"></param>
            /// <param name="eps">Term added to the denominator to improve numerical stability (default: 1e-10)</param>
            /// <returns></returns>
            public static AdagradOptimizer Adagrad(IEnumerable<Tensor> parameters, double learningRate = 1e-2, double lr_decay = 0, double weight_decay = 0, double initial_accumulator_value = 0, double eps = 1e-10)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_Adagrad_ctor(paramsRef, parray.Array.Length, learningRate, lr_decay, weight_decay, initial_accumulator_value, eps);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdagradOptimizer(res, learningRate);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_SGD_ctor(IntPtr parameters, int len, double learningRate, double momentum, double dampening, double weight_decay, bool nesterov);

            /// <summary>
            /// Implements stochastic gradient descent (optionally with momentum).
            /// </summary>
            /// <param name="parameters">Prameters to optimize</param>
            /// <param name="learningRate">Learning rate</param>
            /// <param name="momentum">Momentum factor (default: 0)</param>
            /// <param name="dampening">Dampening for momentum (default: 0)</param>
            /// <param name="weight_decay">Weight decay (L2 penalty) (default: 0)</param>
            /// <param name="nesterov">Enables Nesterov momentum (default: False)</param>
            /// <returns></returns>
            public static SGDOptimizer SGD(IEnumerable<Tensor> parameters, double learningRate, double momentum = 0, double dampening = 0, double weight_decay = 0, bool nesterov = false)
            {
                var parray = new PinnedArray<IntPtr>();
                IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

                var res = THSNN_SGD_ctor(paramsRef, parray.Array.Length, learningRate, momentum, dampening, weight_decay, nesterov);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new SGDOptimizer(res, learningRate);
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        public class AdagradOptimizer : Optimizer, ILearningRateController
        {
            public AdagradOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_Adagrad_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_Adagrad_set_lr(handle, value); torch.CheckForErrors(); }
            }

            private double _rate;
        }

        public class AdamOptimizer : Optimizer, ILearningRateController
        {
            public AdamOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_Adam_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_Adam_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            private double _rate;
        }

        public class AdamWOptimizer : Optimizer, ILearningRateController
        {
            public AdamWOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_AdamW_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_AdamW_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            private double _rate;
        }

        public class RMSPropOptimizer : Optimizer, ILearningRateController
        {
            public RMSPropOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_RMSprop_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_RMSprop_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            private double _rate;
        }

        public class LBFGSOptimizer : Optimizer, ILearningRateController
        {
            public LBFGSOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_LBFGS_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_LBFGS_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            public override void step(Func<Tensor> closure = null)
            {
                if (closure == null)
                    throw new ArgumentNullException("'closure' must be non-null when using the LBFGS optimizer. See: https://pytorch.org/docs/1.9.0/optim.html");
                base.step(closure);
            }

            private double _rate;
        }

        public class SGDOptimizer : Optimizer, ILearningRateController
        {
            public SGDOptimizer(IntPtr handle, double lr) : base(handle)
            {
                _rate = lr;
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSNN_SGD_set_lr(HType optimizer, double lr);

            public double LearningRate {
                get { return _rate; }
                set { THSNN_SGD_set_lr(handle, value); torch.CheckForErrors(); _rate = value; }
            }

            private double _rate;
        }
    }
}

