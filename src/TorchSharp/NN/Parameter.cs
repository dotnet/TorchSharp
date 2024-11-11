// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A kind of Tensor that is to be considered a module parameter.
        ///
        /// Parameters are Tensor subclasses, that have a very special property when used with Modules -
        /// when they’re assigned as Module attributes they are automatically added to the list of its parameters,
        /// and will appear e.g. in parameters() iterator.Assigning a Tensor doesn’t have such effect.This is because
        /// one might want to cache some temporary state, like last hidden state of the RNN, in the model.
        /// If there was no such class as Parameter, these temporaries would get registered too.
        /// </summary>
        public class Parameter : Tensor
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="data">A tensor, which will become empty.</param>
            /// <param name="requires_grad"></param>
            public Parameter(Tensor data, bool requires_grad = true) :
                base(data.with_requires_grad(requires_grad).MoveHandle(), false)
            {
                var scope = data.OwningDisposeScope;
                if (scope is not null) {
                    DisposeScope.ReplaceWith(data, this);
                }
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="handle">A tensor handle.</param>
            internal Parameter(System.IntPtr handle) : base(handle)
            {
            }
        };
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public static Parameter Parameter(Tensor data, bool requires_grad = true) =>
                new Parameter(data, requires_grad);

        }
    }
}
