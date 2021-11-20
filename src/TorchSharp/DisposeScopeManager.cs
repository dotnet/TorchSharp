// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace TorchSharp
{
    /// <summary>
    /// Manages dispose scopes, that can make automatic tensor disposal easier. Note that the
    /// DisposeManager is thread local. The DisposeScopeManager can also manage other disposables, such as training
    /// batches and the like.
    /// </summary>
    public class DisposeScopeManager
    {
        [ThreadStatic] private static DisposeScopeManager _threadSingleton;

        internal static DisposeScopeManager ThreadSingleton => (_threadSingleton ??= new DisposeScopeManager());
        internal Stack<DisposeScope> DisposeScopeStack { get; } = new();
        public long CreatedOutsideScopeCount { get; private set; }
        public long CreatedInScopeCount { get; private set; }
        public long DisposedInScopeCount { get; internal set; }

        /// <summary>
        /// The number of disposables that are currently live on the current thread. It's aproximate, see
        /// Tensor.TotalCount.
        /// </summary>
        public static long ThreadTotalLiveCount => ThreadSingleton.CreatedInScopeCount - ThreadSingleton.DisposedInScopeCount;

        internal DisposeScope RegisterOnCurrentDisposeScope(IDisposable disposable)
        {
            if (DisposeScopeStack.Count == 0) {
                CreatedOutsideScopeCount++;
                return null;
            }

            CreatedInScopeCount++;
            var current = DisposeScopeStack.Peek();
            current.Include(disposable);
            return current;
        }

        internal static DisposeScope NewDisposeScope()
        {
            return ThreadSingleton.InnerNewDisposeScope();
        }

        private DisposeScope InnerNewDisposeScope()
        {
            var disposeScope = new DisposeScope(this);
            DisposeScopeStack.Push(disposeScope);
            return disposeScope;
        }

        internal void RemoveDisposeScope(DisposeScope disposeScope)
        {
            Debug.Assert(DisposeScopeStack.Count > 0);
            Debug.Assert(DisposeScopeStack.Peek() == disposeScope);
            DisposeScopeStack.Pop();
        }
    }
}