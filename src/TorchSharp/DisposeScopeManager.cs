// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace TorchSharp
{
    /// <summary>
    /// Manages dispose scopes, that can make automatic tensor disposal easier. Note that the DisposeScopeManager is
    /// thread static, so each thread has its own DisposeScopeManager. The DisposeScopeManager can also manage other
    /// disposables, such as training batches, and will dispose them when it's disposed itself. Tensors are
    /// automatically added and removed from the current DisposeScope, if there is one.
    /// </summary>
    public class DisposeScopeManager
    {
        [ThreadStatic] private static DisposeScopeManager _threadSingleton;

        internal static DisposeScopeManager ThreadSingleton => (_threadSingleton ??= new DisposeScopeManager());
        internal Stack<DisposeScope> DisposeScopeStack { get; } = new();
        private long _createdOutsideScopeCount;
        private long _createdInsideScopeCount;
        private long _disposedInsideScopeCount;

        /// <summary>
        /// The number of disposables that are currently live on the current thread. This number is thread safe, which
        /// Tensor.TotalCount isn't, but it's aproximate because some tensors may refer to the same backing storage,
        /// see Tensor.TotalCount.
        /// </summary>
        public static long ThreadTotalLiveCount => ThreadSingleton._createdOutsideScopeCount - ThreadSingleton._disposedInsideScopeCount;

        /// <summary>
        /// Keeps track of how often a tensor has been created on this thread, but there was no DisposeScope to
        /// receive it. For those tensors, we don't track when/if they're disposed.
        /// </summary>
        public static long CreatedOutsideScopeCount => ThreadSingleton._createdOutsideScopeCount;

        /// <summary>
        /// The number of tensors that have been created in a scope on this thread, any scope, not just the current. This
        /// number is thread safe, which Tensor.TotalCount isn't.
        /// </summary>
        public static long CreatedInsideScopeCount => ThreadSingleton._createdInsideScopeCount;

        /// <summary>
        /// The number of tensors that have been disposed in the scope on this theread.
        /// </summary>
        public static long DisposedInsideScopeCount => ThreadSingleton._disposedInsideScopeCount;

        /// <summary>
        /// If you have additional disposables that you would like tracked by the dispose scope, batch data and things
        /// like that, you can make your class disposable and call RegisterOnCurrentDisposeScope in it's constructor.
        /// That way, your object will be disposed when the scope is disposed.
        /// </summary>
        public DisposeScope RegisterOnCurrentDisposeScope(IDisposable disposable)
        {
            if (DisposeScopeStack.Count == 0) {
                _createdOutsideScopeCount++;
                return null;
            }

            _createdInsideScopeCount++;
            var current = DisposeScopeStack.Peek();
            current.Include(disposable);
            return current;
        }

        internal static DisposeScope NewDisposeScope()
        {
            return ThreadSingleton.InnerNewDisposeScope();
        }

        internal void RemoveDisposeScope(DisposeScope disposeScope)
        {
            Debug.Assert(DisposeScopeStack.Count > 0);
            Debug.Assert(DisposeScopeStack.Peek() == disposeScope);
            DisposeScopeStack.Pop();
        }

        internal void WasDisposed(IDisposable disposable)
        {
            _disposedInsideScopeCount++;
        }

        private DisposeScope InnerNewDisposeScope()
        {
            var disposeScope = new DisposeScope(this);
            DisposeScopeStack.Push(disposeScope);
            return disposeScope;
        }
    }
}