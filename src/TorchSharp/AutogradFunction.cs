// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;
using System.Net;
using TorchSharp.PInvoke;
using SkiaSharp;

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
                    Dispose(true);
                }
            }
            internal abstract class Node
            {
                protected static List<Node> AliveNodes = new();

                public NodeUnmanagedPtr handle;
            }

            internal class Node<T> : Node, IDisposable where T : Function<T>, new()
            {
                private AutogradContext _context;
                private PinnedArray<IntPtr> _ptrArray;
                private List<bool> _isVariableInput;
                private List<Tensor> _outputCache;
                private ApplyFunc _applyFuncRef;
                private ManagedDeleteNode _managedDeleteNode;
                private object _mutex;

                internal Node()
                {
                    _applyFuncRef = ApplyFunc;
                    _managedDeleteNode = DeleteNode;
                    handle = THSAutograd_CSharpNode_ctor(_applyFuncRef, _managedDeleteNode);
                    CheckForErrors();
                    AliveNodes.Add(this);

                    _ptrArray = new();
                    _isVariableInput = new List<bool>();
                    _context = new AutogradContext(this);
                    _mutex = new();
                }

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
                    THSAutograd_CSharpNode_setNextEdges(handle, l.CreateArrayWithSize(inputVars.Select(v => v?.handle ?? IntPtr.Zero).ToArray()), isExecutable);
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

                    var varsPtr = varsArr.CreateArrayWithSize(inputVars.Select(v => v?.handle ?? IntPtr.Zero).ToArray());
                    var diffsPtr = diffArr.CreateArrayWithSize(_context.NonDifferentiableTensors.ToArray());
                    var dirtyPtr = diffArr.CreateArrayWithSize(_context.DirtyTensors.ToArray());
                    var outputPtr = outputArr.CreateArrayWithSize(outputs.Select(v => v?.handle ?? IntPtr.Zero).ToArray());

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

                private ArrayWithSize ApplyFunc(IntPtr[] tensors)
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

                        var output = Function<T>.Instance.backward(_context, backwardInputs);

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
                        for (int i = 0; i < output.Count; i++) {
                            if (!_isVariableInput[i] && output[i] is not null && !output[i].IsInvalid)
                                throw new NotImplementedException($"Function {Function<T>.Instance.Name} returned a gradient different that is defined at position {i + 1}, but the corresponding forward input was not a Tensor");
                        }

                        // Convert back to C++ array
                        return _ptrArray.CreateArrayWithSize(output.Select(p => p?.handle ?? IntPtr.Zero).ToArray());
                    }
                }

                private void DeleteNode()
                {
                    AliveNodes.Remove(this);
                    Dispose();
                }

                public void DisposeSharedPtr()
                {
                    THSAutograd_CSharpNode_disposeSharedPtr(handle);
                    CheckForErrors();
                    handle.sharedPtr = IntPtr.Zero;
                }

                public void Dispose()
                {
                    Dispose(true);
                    GC.SuppressFinalize(this);
                }

                public void Dispose(bool disposing)
                {
                    if (disposing) {
                        if (handle.sharedPtr != IntPtr.Zero) 
                            DisposeSharedPtr();
                        if (handle.weakPtr != IntPtr.Zero) {
                            THSAutograd_CSharpNode_disposeWeakPtr(handle);
                            CheckForErrors();
                            handle.weakPtr = IntPtr.Zero;
                        }

                        _context?.Dispose();
                        _context = null;
                        _ptrArray?.Dispose();
                        _ptrArray = null;
                        _outputCache?.ForEach(t => t.Dispose());
                        _outputCache = null;
                    }
                }

                ~Node()
                {
                    Dispose(true);
                }
            }


            public class AutogradContext : IDisposable
            {
                private Node _node;
                private List<Tensor> _toSave = new();
                private HashSet<IntPtr> _dirtyTensors = new();
                private HashSet<IntPtr> _nonDifferentiableTensors = new();
                private List<SavedVariable> _savedVariables = new();
                private bool _materializeGrads = true;

                internal AutogradContext(Node node)
                {
                    _node = node;
                }

                public void save_for_backward(List<Tensor> tensors) => _toSave = tensors;
                public void mark_dirty(List<Tensor> tensors) => _dirtyTensors = tensors.Select(t => t?.handle ?? IntPtr.Zero).ToHashSet();
                public void mark_non_differentiable(List<Tensor> tensors) => _nonDifferentiableTensors = tensors.Select(t => t?.handle ?? IntPtr.Zero).ToHashSet();
                public void set_materialize_grads(bool value) => _materializeGrads = value;

                public List<Tensor> get_saved_variables()
                {
                    lock (_node) {
                        return _savedVariables.Select(v => v.unpack(_node)).ToList();
                    }
                }

                internal void save_variables()
                {
                    _savedVariables.Clear();
                    lock (_node) {
                        foreach (var tensor in _toSave)
                            _savedVariables.Add(tensor is null || tensor.IsInvalid ? new() : new(tensor, _node.handle));
                    }
                }

                internal bool MaterializeGrads => _materializeGrads;
                internal HashSet<IntPtr> DirtyTensors => _dirtyTensors;
                internal HashSet<IntPtr> NonDifferentiableTensors => _nonDifferentiableTensors;

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
            public abstract class Function<T> where T : Function<T>, new()
            {
                internal static T Instance { get; set; } = new();

                public static List<Tensor> apply(params object[] vars)
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
                        outputs = Instance.forward(node.Context, vars);
                    outputs = node.WrapOutputs(inputVars, outputs, isExecutable);
                    
                    if (isExecutable)
                        node.save_variables_to_ctx();

                    node.DisposeSharedPtr();
                    
                    return outputs;

                }

                public abstract List<Tensor> forward(AutogradContext ctx, params object[] vars);

                public abstract List<Tensor> backward(AutogradContext ctx, List<Tensor> grad_outputs);

                public abstract string Name { get; }
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
