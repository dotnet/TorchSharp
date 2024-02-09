// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;
using System.Net;
using TorchSharp.PInvoke;
using SkiaSharp;
using System.Security.Authentication.ExtendedProtection;
using System.Runtime.InteropServices;
using System.Threading;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class autograd
        {
            internal class SavedVariable : IDisposable
            {
                private IntPtr _handle;
                private bool _emptyConstructor;

                internal SavedVariable(Tensor tensor, NodeUnmanagedPtr nodeHandle, bool is_inplace_on_view = false)
                {
                    _handle = THSAutograd_SavedVariable_ctor(tensor.Handle, nodeHandle, is_inplace_on_view);
                    CheckForErrors();
                }

                internal SavedVariable()
                {
                    _handle = IntPtr.Zero;
                    _emptyConstructor = true;
                }

                public Tensor unpack(Node saved_for = null)
                {
                    if (_emptyConstructor)
                        return null;

                    IntPtr result = THSAutograd_SavedVariable_unpack(this._handle, saved_for?.handle ?? new());
                    CheckForErrors();
                    return new Tensor(result);
                }

                public void reset_data()
                {
                    if (!_emptyConstructor) {
                        THSAutograd_SavedVariable_reset_data(this._handle);
                        CheckForErrors();
                    }
                }

                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                public void Dispose(bool disposing)
                {
                    if (_handle != IntPtr.Zero) {
                        THSAutograd_SavedVariable_dispose(_handle);
                        CheckForErrors();
                        _handle = IntPtr.Zero;
                    }
                }

                ~SavedVariable()
                {
                    Dispose(false);
                }
            }

            // An simplified version of node which doesn't have a type template, for use in other functions which just
            // need access to the handle.
            internal abstract class Node
            {
                public NodeUnmanagedPtr handle;

                private static List<Node> AliveNodes = new();
                protected static void AddNode(Node node)
                {
                    lock (AliveNodes) {
                        AliveNodes.Add(node);
                    }
                }

                protected static void DeleteNode(Node node)
                {
                    lock (AliveNodes) {
                        AliveNodes.Remove(node);
                    }
                }
            }

            internal class Node<T> : Node, IDisposable where T : Function<T>, new()
            {
                // Store delegate refs to the functions, so that the garbage collector doesn't clear them up
                private ApplyFunc _applyFuncRef;
                private ManagedDeleteNode _managedDeleteNode;

                // Store the state of the node
                private AutogradContext _context;
                private List<bool> _isVariableInput;
                private List<Tensor> _outputCache;
                private object _mutex;
                private bool _disposedSharedPtr;

                // Store a copy of a PinnedArray object which is used to return a list of tensors to the C++ code
                // since this function is passed as a delegate, we don't know when to dispose the array created as the
                // return value from the function. Therefore, store this object here and dispose it with the object. 
                private PinnedArray<IntPtr> _applyFuncReturnArray;

                internal Node()
                {
                    _applyFuncRef = ApplyFunc;
                    _managedDeleteNode = DeleteNode;
                    handle = THSAutograd_CSharpNode_ctor(_applyFuncRef, _managedDeleteNode);
                    CheckForErrors();
                    AddNode(this);

                    _applyFuncReturnArray = new();
                    _isVariableInput = new List<bool>();
                    _context = new AutogradContext(this);
                    _mutex = new();
                }

                /// <summary>
                /// Given a list of arguments passed to the forward function, check which of the inputs are Tensors and which
                /// are objects of non-tensor types.
                /// </summary>
                /// <param name="args">The arguments passed to the Function.forward implementtation</param>
                /// <returns>A list of only the tensor objects from the args</returns>
                internal List<Tensor> ComputeVariableInput(object[] args)
                {
                    _isVariableInput.Clear();
                    var ret = new List<Tensor>();
                    foreach (object arg in args) {
                        if (arg is torch.Tensor tensor && !tensor.IsInvalid) {
                            ret.Add(tensor);
                            _isVariableInput.Add(true);
                        } else _isVariableInput.Add(false);
                    }
                    return ret;
                }

                internal void SetNextEdges(List<Tensor> inputVars, bool isExecutable)
                {
                    using var l = new PinnedArray<IntPtr>();
                    THSAutograd_CSharpNode_setNextEdges(handle, l.CreateArrayWithSize(inputVars.Select(v => v.Handle).ToArray()), isExecutable);
                    CheckForErrors();
                }

                internal void ClearInputMetadata()
                {
                    THSAutograd_CSharpNode_clearInputMetadata(handle);
                    CheckForErrors();
                }

                internal List<Tensor> WrapOutputs(List<Tensor> inputVars, List<Tensor> outputs, bool isExecutable)
                {
                    using var varsArr = new PinnedArray<IntPtr>();
                    using var diffArr = new PinnedArray<IntPtr>();
                    using var dirtyArr = new PinnedArray<IntPtr>();
                    using var outputArr = new PinnedArray<IntPtr>();
                    using var resultsArr = new PinnedArray<IntPtr>();

                    var varsPtr = varsArr.CreateArrayWithSize(inputVars.Select(v => v.Handle).ToArray());
                    var diffsPtr = diffArr.CreateArrayWithSize(_context.NonDifferentiableTensors.Select(v => v.Handle).ToArray());
                    var dirtyPtr = diffArr.CreateArrayWithSize(_context.DirtyTensors.Select(v => v.Handle).ToArray());
                    var outputPtr = outputArr.CreateArrayWithSize(outputs.Select(v => v.Handle).ToArray());

                    THSAutograd_Function_wrapOutputs(varsPtr, diffsPtr, dirtyPtr, outputPtr, isExecutable ? handle : new(), resultsArr.CreateArray);
                    CheckForErrors();

                    var ret = resultsArr.Array.Select(x => new Tensor(x)).ToList();
                    SaveOutputsInfo(ret, isExecutable);

                    return ret;
                }

                internal void SaveOutputsInfo(List<Tensor> outputs, bool isExecutable)
                {
                    _outputCache = new();
                    foreach (Tensor t in outputs) {
                        if (isExecutable && !t.IsInvalid)
                            _outputCache.Add(torch.empty_like(t));
                        else if (isExecutable)
                            _outputCache.Add(null);
                    }
                }

                internal void save_variables_to_ctx()
                {
                    _context.save_variables();
                }

                internal AutogradContext Context => _context;

                /// <summary>
                /// This is the function which gets passed back to the unmanaged code, which is used in the autograd graph
                /// to call the custom backward function and compute the gradients. This function wraps around the Function.backward
                /// implementation converting between unmanaged and managed objects. 
                /// </summary>
                /// <returns>A pointer to an array of tensors with the size</returns>
                private ArrayWithSize ApplyFunc(IntPtr[] tensors, int size)
                {
                    lock (_mutex) {
                        // Go through the input tensors and build the input to the backward.
                        // If the tensor is not defined but we are materializing grads, and we have info on the output
                        // in the cache, then initialize a zeros vector
                        var backwardInputs = new List<Tensor>();
                        for (int i = 0; i < tensors.Length; i++) {
                            if (!_context.MaterializeGrads || tensors[i] != IntPtr.Zero || _outputCache[i] is null)
                                backwardInputs.Add(new Tensor(tensors[i]));
                            else backwardInputs.Add(torch.zeros_like(_outputCache[i]));
                        }

                        var output = Function<T>.Instance.backward_internal(_context, backwardInputs);

                        // Returning too many results is ok, but only as long as they're all
                        // undefined. Truncate the result vector in that case.
                        if (output.Count > _isVariableInput.Count) {
                            if (output.Skip(_isVariableInput.Count).All(t => t is null || t.IsInvalid))
                                output = output.GetRange(0, _isVariableInput.Count);
                        }

                        // Safety checks:
                        // 1] The number of returned output items must be the same as the input
                        if (output.Count != _isVariableInput.Count)
                            throw new NotImplementedException($"Function {Function<T>.Instance.Name} returned an incorrect number of gradients (expected {_isVariableInput.Count}), got {output.Count}");

                        // 2] The valid tensors positions should match the variable inputs
                        var returnValue = new List<Tensor>();
                        for (int i = 0; i < output.Count; i++) {
                            if (!_isVariableInput[i]) {
                                if (output[i] is not null && !output[i].IsInvalid)
                                    throw new NotImplementedException($"Function {Function<T>.Instance.Name} returned a gradient different that is defined at position {i + 1}, but the corresponding forward input was not a Tensor");
                                continue;
                            }
                            returnValue.Add(output[i]);
                        }
                        
                        // Convert back to C++ array
                        return _applyFuncReturnArray.CreateArrayWithSize(returnValue.Select(p => p?.handle ?? IntPtr.Zero).ToArray());
                    }
                }

                /// <summary>
                /// This function is also passed as a parameter to the unmanaged object, so that when the unmanaged object
                /// is destroyed, then we will remove the hard reference we have of this node so that the GC can collect it. 
                /// </summary>
                private void DeleteNode()
                {
                    DeleteNode(this);
                    Dispose();
                }

                /// <summary>
                /// When we create the unmanaged object we need to hold a strong reference to it while we apply the whole process
                /// of populating it with information, but then once we've attached it to the graph we want to still maintain a reference
                /// to it so that we can call into it, but not a strong reference so that it will be destroyed. Therefore, we store
                /// both a shared and a weak pointer to the object, and so once we attach it to a graph, we will dispose the shared ptr.
                /// </summary>
                public void DisposeSharedPtr()
                {
                    lock (_mutex) {
                        if (!_disposedSharedPtr) {
                            // The lock only holds while we are in managed code, but as soon as we call into the native
                            // function the lock releases, and so we have a situation where we might call dispose twice
                            // on the same pointer. Therefore, it's important to set the dispose flag *before* the call to
                            // the native code. Therefore, we will catch and throw the error, flipping the flag back if it
                            // crashed, so that the future dispose can try again.
                            _disposedSharedPtr = true;
                            THSAutograd_CSharpNode_disposeSharedPtr(handle);
                            try {
                                CheckForErrors();
                            } catch {
                                _disposedSharedPtr = false;
                                throw;
                            }
                            handle.sharedPtr = IntPtr.Zero;
                        }
                    }
                }

                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                public void Dispose(bool disposing)
                {
                    if (handle.sharedPtr != IntPtr.Zero) 
                        DisposeSharedPtr();
                    if (handle.weakPtr != IntPtr.Zero) {
                        THSAutograd_CSharpNode_disposeWeakPtr(handle);
                        CheckForErrors();
                        handle.weakPtr = IntPtr.Zero;
                    }

                    if (disposing) {
                        _context?.Dispose();
                        _context = null;
                        _applyFuncReturnArray?.Dispose();
                        _applyFuncReturnArray = null;
                        _outputCache?.ForEach(t => t?.Dispose());
                        _outputCache = null;
                    }
                }

                ~Node()
                {
                    Dispose(false);
                }
            }


            /// <summary>
            /// Context to save information during `forward` that can be accessed in
            /// `backward` in custom autograd operations (see `torch::autograd::Function` for details).
            /// </summary>
            public class AutogradContext : IDisposable
            {
                private Node _node;
                private bool _materializeGrads = true;
                private List<Tensor> _tensorsToSave = new();
                private List<Tensor> _dirtyTensors = new();
                private List<Tensor> _nonDifferentiableTensors = new();
                private List<SavedVariable> _savedVariables = new();
                private Dictionary<string, object> _savedData = new();
                
                internal AutogradContext(Node node)
                {
                    _node = node;
                }

                /// <summary>
                /// Saves the list of variables for a future call to `backward`. This
                /// should be called at most once from inside of `forward`.
                /// </summary>
                public void save_for_backward(List<Tensor> tensors) => _tensorsToSave = tensors;

                /// <summary>
                /// Saves a non tensor object in the data field for a future call to `backward`.
                /// </summary>
                public void save_data(string key, object value)
                {
                    if (value is Tensor)
                        Console.WriteLine("Attempted to save a tensor object in the data object. Please store tensors by calling `save_for_backward()`");
                    _savedData[key] = value;
                }

                /// <summary>
                /// Marks variables in the list as modified in an in-place operation. This
                /// should be called at most once from inside of `forward` and all arguments
                /// should be inputs.
                /// </summary>
                public void mark_dirty(List<Tensor> tensors) => _dirtyTensors = tensors.ToList();

                /// <summary>
                /// Marks outputs in the list as not requiring gradients. This should be
                /// called at most once from inside of `forward` and all arguments should be
                /// outputs.
                /// </summary>
                public void mark_non_differentiable(List<Tensor> tensors) => _nonDifferentiableTensors = tensors.ToList();

                /// <summary>
                /// Sets whether undefined output grad tensors should be expanded to tensors
                /// full of zeros before calling backward function. Default value is true.
                /// </summary>
                public void set_materialize_grads(bool value) => _materializeGrads = value;

                public bool MaterializeGrads => _materializeGrads;

                /// <summary>
                /// Get the list of variables that were saved in `forward` using
                /// `save_for_backward()`. Before returning them to the user, a check is made
                /// to ensure that they were not modified by any in-place operations.
                /// </summary>
                public List<Tensor> get_saved_variables()
                {
                    lock (_node) {
                        return _savedVariables.Select(v => v.unpack(_node)).ToList();
                    }
                }

                /// <summary>
                /// Retrieve a non-tensor value stored in the context in the `forward` function.
                /// </summary>
                public object get_data(string key) => _savedData.TryGetValue(key, out var value) ? value : default;

                /// <summary>
                /// Internal function called during the construction of the node on the graph to commit the tensors
                /// to memory as SavedVariable unmanaged objects. 
                /// </summary>
                internal void save_variables()
                {
                    _savedVariables.Clear();
                    lock (_node) {
                        foreach (var tensor in _tensorsToSave)
                            _savedVariables.Add(tensor is null || tensor.IsInvalid ? new() : new(tensor, _node.handle));
                    }
                }

                internal List<Tensor> DirtyTensors => _dirtyTensors;
                internal List<Tensor> NonDifferentiableTensors => _nonDifferentiableTensors;

                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                protected virtual void Dispose(bool disposing)
                {
                    if (disposing) {
                        _savedVariables?.ForEach(v => v.Dispose());
                        _savedVariables = null;
                    }
                }
            }

            /// <summary>
            /// To use custom autograd operations, implement a Function subclass implementing tthe
            /// forward and backward functions:
            /// `forward` can take as many arguments as you want, passed an object[], and will return either
            /// a list of tensors or a single tensor (depending which subclass you inherit).
            /// Use of any direct Variable arguments will be registered in the graph but no vectors/sets or any
            /// other data structures will be traversed. You can pass null tensors as one of the arguments
            /// and it will be registered as a variable in the graph if the argument has a
            /// value. It should take a pointer to `torch::autograd::AutogradContext` as the
            /// first argument. Tensors can be saved in the `ctx` using `ctx->save_for_backward`
            /// (see `torch::autograd::AutogradContext::save_for_backward`) and other data
            /// can be saved using the `ctx->save_data` function (see `torch::autograd::AutogradContext::save_data`)
            /// in the form of `(string, object)` pairs.
            /// Variables saved in `forward` can be accessed with `ctx->get_saved_variables` (see
            /// `torch::autograd::AutogradContext::get_saved_variables`) and other saved
            /// data can be accessed using `ctx->get_data()`.
            /// </summary>
            public abstract class Function<T> where T : Function<T>, new()
            {
                internal static T Instance { get; set; } = new();

                /// <summary>
                /// When calling the function, the user should call Function.apply and not the forward function, so that
                /// the computation graph is built correctly.
                /// </summary>
                public static List<Tensor> apply_internal(params object[] vars)
                {
                    var node = new Node<T>();

                    // Compute the variable input - check which of the arguments are valid tensors
                    var inputVars = node.ComputeVariableInput(vars);

                    // Set up the node with the relevant information
                    bool isExecutable = AutoGradMode.IsEnabled && inputVars.Any(v => v.requires_grad);
                    node.SetNextEdges(inputVars, isExecutable);
                    node.ClearInputMetadata();

                    // Call the forward function and then wrap the outputs
                    List<Tensor> outputs;
                    using (var d = new AutoGradMode(false))
                        outputs = Instance.forward_internal(node.Context, vars);
                    outputs = node.WrapOutputs(inputVars, outputs, isExecutable);
                    
                    if (isExecutable)
                        node.save_variables_to_ctx();

                    node.DisposeSharedPtr();
                    
                    return outputs;

                }

                internal abstract List<Tensor> forward_internal(AutogradContext ctx, params object[] vars);
                internal abstract List<Tensor> backward_internal(AutogradContext ctx, List<Tensor> grad_outputs);
                public abstract string Name { get; }
            }

            public abstract class SingleTensorFunction<T> : Function<T> where T : SingleTensorFunction<T>, new()
            {
                public static Tensor apply(params object[] vars)
                {
                    return apply_internal(vars)[0];
                }

                internal sealed override List<Tensor> forward_internal(AutogradContext ctx, params object[] vars)
                {
                    return new List<Tensor>() { forward(ctx, vars) };
                }

                internal sealed override List<Tensor> backward_internal(AutogradContext ctx, List<Tensor> grad_outputs)
                {
                    return backward(ctx, grad_outputs[0]);
                }

                public abstract Tensor forward(AutogradContext ctx, params object[] vars);
                public abstract List<Tensor> backward(AutogradContext ctx, Tensor grad_output);
            }

            public abstract class MultiTensorFunction<T> : Function<T> where T : MultiTensorFunction<T>, new()
            {
                public static List<Tensor> apply(params object[] vars)
                {
                    return apply_internal(vars);
                }

                internal sealed override List<Tensor> forward_internal(AutogradContext ctx, params object[] vars)
                {
                    return forward(ctx, vars);
                }

                internal sealed override List<Tensor> backward_internal(AutogradContext ctx, List<Tensor> grad_outputs)
                {
                    return backward(ctx, grad_outputs);
                }

                public abstract List<Tensor> forward(AutogradContext ctx, params object[] vars);

                public abstract List<Tensor> backward(AutogradContext ctx, List<Tensor> grad_outputs);
            }
        }
    }

    static class Extensions
    {
        public static ArrayWithSize CreateArrayWithSize(this PinnedArray<IntPtr> arr, IntPtr[] array)
        {
            return new ArrayWithSize() {
                Array = arr.CreateArray(array),
                Size = array.LongLength
            };
        }
    }
}
