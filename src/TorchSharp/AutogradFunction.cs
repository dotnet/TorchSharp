// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class autograd
        {
            public class Node
            {
                private AutogradContext _context;
                private IntPtr _handle;
                
                public Node(Action releaseFunc, Func<List<Tensor>, List<Tensor>> applyFunc)
                {
                    _handle = THSAutograd_CSharpNode_ctor(releaseFunc.Invoke, WrapApplyFunc(applyFunc));
                }

                private static ApplyFunc WrapApplyFunc(Func<List<Tensor>, List<Tensor>> applyFunc)
                {
                    return tensors => {
                        var output = applyFunc(tensors.Select(t => new Tensor(t)).ToList());
                        using var ts = new PinnedArray<IntPtr>();
                        return ts.CreateArray(output.Select(p => p.Handle).ToArray());
                    };
                }
            }


            public class AutogradContext
            {
                private List<Tensor> _toSave;
                private List<Tensor> _savedForForward;
                private HashSet<IntPtr> _dirtyTensors;
                private HashSet<IntPtr> _nonDifferentiableTensors;
                private bool _materializeGrads = true;

                public void save_for_backward(List<Tensor> tensors) => _toSave = tensors;
                public void save_for_forward(List<Tensor> tensors) => _savedForForward = tensors;
                public void mark_dirty(List<Tensor> tensors) => _dirtyTensors = tensors.Select(t => t.handle).ToHashSet();
                public void mark_non_differentiable(List<Tensor> tensors) => _nonDifferentiableTensors = tensors.Select(t => t.handle).ToHashSet();
                public void set_materialize_grads(bool value) => _materializeGrads = value;
            }
            public abstract class Function<T> where T : Function<T>, new()
            {
                protected Function()
                {
                }

                private static T Instance { get; set; } = new();

                public static List<Tensor> apply(params object[] vars)
                {
                    return Instance.forward(null, vars);
                }

                public abstract List<Tensor> forward(AutogradContext ctx, params object[] vars);

                public abstract List<Tensor> backward(AutogradContext ctx, List<Tensor> grad_output);
            }

        }
    }
}
