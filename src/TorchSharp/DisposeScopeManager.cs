// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;

#nullable enable
namespace TorchSharp
{
    /// <summary>
    /// Manages dispose scopes, that can make automatic tensor disposal easier. Note that the
    /// DisposeManager is thread local. The DisposeScopeManager can also manage other disposables, such as training
    /// batches and the like.
    /// </summary>
    public class DisposeScopeManager
    {
        [ThreadStatic] private static DisposeScopeManager? _threadSingleton;
        internal ThreadDisposeScopeStatistics StatisticsInstance { get; } = new ThreadDisposeScopeStatistics();

        internal static DisposeScopeManager ThreadSingleton => (_threadSingleton ??= new DisposeScopeManager());
        internal List<DisposeScope> DisposeScopeStack { get; } = new();

        public static ThreadDisposeScopeStatistics Statistics => ThreadSingleton.StatisticsInstance;

        internal DisposeScope? RegisterOnCurrentDisposeScope(IDisposable disposable)
        {
            if (DisposeScopeStack.Count == 0) {
                StatisticsInstance.CreatedOutsideScopeCount++;
                return null;
            }

            StatisticsInstance.CreatedInScopeCount++;
            var current = DisposeScopeStack[DisposeScopeStack.Count - 1];
            current.Include(disposable);
            return current;
        }

        internal static DisposeScope NewDisposeScope()
        {
            return ThreadSingleton.InnerNewDisposeScope();
        }

        internal void RemoveDisposeScope(DisposeScope disposeScope)
        {
            var index = DisposeScopeStack.LastIndexOf(disposeScope);
            if (index is not -1)
                DisposeScopeStack.RemoveAt(index);
        }

        private DisposeScope InnerNewDisposeScope()
        {
            var disposeScope = new DisposeScope(this);
            DisposeScopeStack.Add(disposeScope);
            return disposeScope;
        }
    }
}