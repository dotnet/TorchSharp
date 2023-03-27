// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using System.IO;
    using Modules;
    using TorchSharp.Utils;

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

                public class StateDictionary
                {
                    internal StateDictionary() { Options = new List<OptimizerOptions>(); State = new List<OptimizerState>(); }

                    public List<OptimizerOptions> Options { get; private set; }

                    public List<OptimizerState> State { get; private set; }
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
        public abstract class OptimizerHelper : Optimizer
        {
            public OptimizerHelper() : base(IntPtr.Zero)
            {
            }


            protected enum TypeCode
            {
                Double = 0,
                Long = 1,
                Tensor = 2,
                Options = 3,
                State = 4,
            }

            /// <summary>
            /// Saves the optimizer state.
            /// </summary>
            /// <param name="location">The name of a file where optimizer state data will be stored.</param>
            public void save_state_dict(string location)
            {
                using var stream = System.IO.File.Create(location);
                using (var writer = new System.IO.BinaryWriter(stream)) {

                    save_state_dict(writer);
                    writer.Flush();
                    stream.Flush();
                }
            }

            /// <summary>
            /// Loads the optimizer state.
            /// </summary>
            /// <param name="location">The name of a file where optimizer state data is stored.</param>
            /// <remarks>
            /// Optimizer state saved from PyTorch cannot be restored -- the file format is unique to TorchSharp.
            /// </remarks>
            public void load_state_dict(string location)
            {
                using var stream = System.IO.File.OpenRead(location);
                using (var reader = new System.IO.BinaryReader(stream)) {

                    load_state_dict(reader);
                }
            }

            /// <summary>
            /// Saves the optimizer state.
            /// </summary>
            /// <param name="writer">A binary writer connected to a stream where .</param>
            public void save_state_dict(System.IO.BinaryWriter writer)
            {
                // Save the name of the optimizer, so that we can avoid problems.
                writer.Write(this.GetType().Name);

                var sd = state_dict();

                writer.Encode(sd.Options.Count); // 4 bytes
                writer.Encode(sd.State.Count);   // 4 bytes

                foreach (var opts in sd.Options) {

                    opts.SaveStateDict(writer);
                }

                foreach (var state in sd.State) {
                    state.SaveStateDict(writer);
                }
            }

            /// <summary>
            /// Loads the optimizer state.
            /// </summary>
            /// <param name="reader">A binary reader connected to a stream containing saved optimizer state.</param>
            /// <remarks>
            /// Optimizer state saved from PyTorch cannot be restored -- the file format is unique to TorchSharp.
            /// </remarks>
            public void load_state_dict(System.IO.BinaryReader reader)
            {
                var optName = reader.ReadString();
                if (optName != this.GetType().Name) {
                    throw new InvalidDataException($"Mismatched optimizer type: expected '{this.GetType().Name}', but found '{optName}' in the loaded stream.");
                }

                // First, figure out how many entries.
                var options = reader.Decode();
                var states = reader.Decode();

                var sd = state_dict();

                if (options != sd.Options.Count) throw new ArgumentException("Invalid optimizer state -- different number of parameter groups.");
                if (states != sd.State.Count) throw new ArgumentException("Invalid optimizer state -- different number of states.");

                for (var i = 0; i < options; i++) {
                    var opts = sd.Options[i] as OptimizerOptions;
                    opts.LoadStateDict(reader);
                }

                for (var i = 0; i < states; i++) {

                    var state = sd.State[i] as OptimizerState;
                    state.LoadStateDict(reader);
                }
            }

            /// <summary>
            /// Loads the optimizer state.
            /// </summary>
            /// <param name="state_dict">Optimizer state. Should be an object returned from a call to state_dict().</param>
            /// <remarks>
            /// The format of the optimizer state dict is different from PyTorch's. Instead of a dictionary with two entries,
            /// the state is represented as record with two entries, both containing lists with state.
            /// </remarks>
            public virtual void load_state_dict(StateDictionary state_dict)
            {
                var sd = this.state_dict();

                if (state_dict.Options.Count != sd.Options.Count) throw new ArgumentException("Invalid optimizer state -- different number of parameter groups.");
                if (state_dict.State.Count != sd.State.Count) throw new ArgumentException("Invalid optimizer state -- different number of states.");

                for (var i = 0; i < state_dict.Options.Count; i++) {

                    var st_opts = sd.Options[i] as OptimizerOptions;
                    if (st_opts != state_dict.Options[i]) {
                        st_opts.LoadStateDict(state_dict.Options[i]);
                    }
                }

                for (var i = 0; i < state_dict.State.Count; i++) {

                    var st_state = sd.State[i] as OptimizerState;
                    if (st_state != state_dict.State[i]) {
                        st_state.LoadStateDict(state_dict.State[i]);
                    }
                }
            }

            /// <summary>
            /// Returns the state of the optimizer as a dict.
            /// </summary>
            /// <returns></returns>
            /// <remarks>
            /// The format of the optimizer state dict is different from PyTorch's. Instead of a dictionary with two entries,
            /// the state is represented as record with two entries, both containing lists with state.
            /// </remarks>
            public virtual StateDictionary state_dict()
            {
                var dict = new StateDictionary();

                int pgidx = -1;
                int pridx = 0;
                foreach (var pg in _parameter_groups) {

                    dict.Options.Add(pg.Options);

                    foreach (var p in pg.Parameters) {

                        dict.State.Add(_state[p.Handle]);
                        pridx++;
                    }
                    pgidx--;
                }
                return dict;
            }

            /// <summary>
            /// Move all the state to the indicated device.
            /// </summary>
            /// <param name="device">The device to move all state to.</param>
            public void to(Device device)
            {
                foreach (var pg in _parameter_groups) {
                    foreach (var p in pg.Parameters) {
                        var state = _state[p.Handle];
                        state.to(device);
                    }
                }
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
            protected Tensor _step<T>(Action<T> body, Func<Tensor> loss_closure = null) where T : ParamGroup
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

            protected Utils.OrderedDict<IntPtr, OptimizerState> _state = new Utils.OrderedDict<IntPtr, OptimizerState>();
        }

        /// <summary>
        /// Base class for optimizer options.
        /// </summary>
        public class OptimizerOptions
        {
            public double? LearningRate { get; set; }
            public double InitialLearningRate { get; set; }

            /// <summary>
            /// Save the optimizer options (param-group hyperparameters) to a stream.
            /// </summary>
            /// <param name="writer">A binary writer connected to a stream open for writing.</param>
            public virtual void SaveStateDict(BinaryWriter writer)
            {
                writer.Write(InitialLearningRate);
                writer.Write(LearningRate.Value);
            }

            /// <summary>
            /// Load the optimizer options (param-group hyperparameters) from a stream.
            /// </summary>
            /// <param name="reader">A binary reader connected to a stream open for reading.</param>
            public virtual void LoadStateDict(BinaryReader reader)
            {
                InitialLearningRate = reader.ReadDouble();
                LearningRate = reader.ReadDouble();
            }

            /// <summary>
            /// Load optimizer options (param-group hyperparameters) from another optimizer.
            /// </summary>
            /// <param name="source">An optimizer options record.</param>
            public virtual void LoadStateDict(OptimizerOptions source)
            {
                InitialLearningRate = source.InitialLearningRate;
                LearningRate = source.LearningRate;
            }
        }

        /// <summary>
        /// Base class for optimizer options.
        /// </summary>
        public abstract class OptimizerState
        {
            /// <summary>
            /// Save the optimizer parameter state to a stream.
            /// </summary>
            /// <param name="writer">A binary writer connected to a stream open for writing.</param>
            public abstract void SaveStateDict(BinaryWriter writer);

            /// <summary>
            /// Load the optimizer parameter state from a stream.
            /// </summary>
            /// <param name="reader">A binary reader connected to a stream open for reading.</param>
            public abstract void LoadStateDict(BinaryReader reader);

            /// <summary>
            /// Load optimizer parameter state from another optimizer.
            /// </summary>
            /// <param name="source">An optimizer state record.</param>
            public abstract void LoadStateDict(OptimizerState source);

            /// <summary>
            /// Useful for tests, allows comparison of one state with another.
            /// </summary>
            /// <param name="other">The other optimizer state</param>
            /// <returns></returns>
            public virtual bool ApproximatelyEquals(OptimizerState other)
            {
                return false;
            }

            /// <summary>
            /// Move all the state to the indicated device.
            /// </summary>
            /// <param name="device">The device to move all state to.</param>
            public virtual void to(Device device) { }

            protected static void LoadConditionalStateTensor(BinaryReader reader, ref Tensor result)
            {
                var hasTensor = reader.ReadBoolean();

                if (hasTensor) {
                    TensorExtensionMethods.Load(ref result, reader);
                } else {
                    if (result is not null)
                        result.Dispose();
                    result = null;
                }
            }

            protected static void SaveConditionalStateTensor(BinaryWriter writer, Tensor tensor)
            {
                if (tensor is not null) {
                    writer.Write(true);
                    tensor.Save(writer);
                } else {
                    writer.Write(false);
                }
            }
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

