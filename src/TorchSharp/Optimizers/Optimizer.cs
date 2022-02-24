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
            /// Base class for all optimizers.
            /// </summary>
            public abstract partial class Optimizer : IDisposable
            {
                /// <summary>
                /// Class wrapping PyTorch's optimzer object reference.
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

                /// <summary>
                /// Constructor used for optimizers implemented in native code.
                /// </summary>
                /// <param name="handle"></param>
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

                /// <summary>
                /// Sets the gradients of all parameters to zero.
                /// </summary>
                public virtual void zero_grad()
                {
                    THSNN_Optimizer_zero_grad(handle);
                    torch.CheckForErrors();
                }

                [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
                public delegate IntPtr LossClosure();

                /// <summary>
                /// Add a param group to the Optimizer s param_groups.
                /// </summary>
                /// <param name="param_group"></param>
                /// <remarks>This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the Optimizer as training progresses.</remarks>
                public virtual void add_param_group(ParamGroup param_group)
                {
                    throw new NotImplementedException($"add_param_group");
                }


                [DllImport("LibTorchSharp")]
                private static extern IntPtr THSNN_Optimizer_step(HType module, LossClosure closure);

                /// <summary>
                /// Performs a single optimization step (parameter update).
                /// </summary>
                /// <param name="closure">A closure that reevaluates the model and returns the loss. Optional for most optimizers.</param>
                /// <returns></returns>
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

                /// <summary>
                /// Get the parameters that the optimizer is handling.
                /// </summary>
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

            /// <summary>
            /// This interfce is used by learning rate schedulers to access and control
            /// the rates used by optimizers.
            /// </summary>
            public interface ILearningRateController
            {
                /// <summary>
                /// The current LR
                /// </summary>
                double LearningRate { set; get; }

                /// <summary>
                /// The initial LR
                /// </summary>
                double InitialLearningRate { set; get; }
            }

            /// <summary>
            /// Indicates optimizers with support for momentum, which some LR schedulers require.
            /// </summary>
            public interface IMomentum
            {
                double Momentum { get; set; }
            }

            /// <summary>
            /// Indicates optimizers with support for betas instead of momentum.
            /// </summary>
            public interface IBetas
            {
                (double, double) Betas { get; set; }
            }
        }
    }

    namespace Modules
    {
        using static torch.optim;

        /// <summary>
        /// Base class to help with a couple of the things that managed-code implementations need.
        /// </summary>
        public class OptimizerHelper : Optimizer
        {
            public OptimizerHelper() : base(IntPtr.Zero)
            {
            }

            /// <summary>
            /// Sets the gradients of all parameters to zero.
            /// </summary>
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

            /// <summary>
            /// Support routine for implementation of step() in all optimizers that support parameter groups.
            /// </summary>
            /// <typeparam name="T">The ParamGroup type in use</typeparam>
            /// <param name="body">The body of the step update.</param>
            /// <param name="loss_closure">The closure, if any, for computing the loss.</param>
            /// <returns></returns>
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
                    }
                }

                return loss;
            }

            /// <summary>
            /// Get the parameters that the optimizer is handling.
            /// </summary>
            public override IEnumerable<Parameter> parameters()
            {
                return _parameter_groups.SelectMany(pg => pg.Parameters);
            }

            /// <summary>
            /// Add a param group to the Optimizer s param_groups.
            /// </summary>
            /// <param name="param_group"></param>
            /// <remarks>This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the Optimizer as training progresses.</remarks>
            public override void add_param_group(ParamGroup param_group)
            {
                _parameter_groups.Add(param_group);
            }

            protected OptimizerOptions _defaults;
        }

        /// <summary>
        /// Base class for optimizer options.
        /// </summary>
        public class OptimizerOptions
        {
            public double? LearningRate { get; set; }
            public double InitialLearningRate { get; set; }
        }

        /// <summary>
        /// Base class for parameter groups
        /// </summary>
        public class ParamGroup : ILearningRateController
        {
            public IEnumerable<Parameter> Parameters { get; set; }

            public OptimizerOptions Options { get; set; }

            public double LearningRate { get => Options.LearningRate.Value; set => Options.LearningRate = value; }
            public double InitialLearningRate { get => Options.InitialLearningRate; set => Options.InitialLearningRate = value; }
        }

        /// <summary>
        /// Generic-typed version of ParamGroup
        /// </summary>
        /// <typeparam name="TOptions">The type of options used for the parameter group.</typeparam>
        public class ParamGroup<TOptions> : ParamGroup where TOptions : OptimizerOptions
        {
            /// <summary>
            /// Constructor
            /// </summary>
            public ParamGroup()
            {
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="parameters">The parameters of the parameter group</param>
            /// <param name="options">The options of the parameter group</param>
            public ParamGroup(IEnumerable<Parameter> parameters, TOptions options = null)
            {
                base.Parameters = parameters;
                base.Options = options;
            }

            /// <summary>
            /// Parameter group options / hyperparameters, used to control the optimizer algorithm.
            /// </summary>
            public new TOptions Options { get => (TOptions)base.Options; set => base.Options = value; }
        }



        

        

        

        

        

        

        

        

       
    }
}

