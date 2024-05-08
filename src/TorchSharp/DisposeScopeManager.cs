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
        internal DisposeScope? CurrentDisposeScope { get; set; } = null;

        public static ThreadDisposeScopeStatistics Statistics => ThreadSingleton.StatisticsInstance;

        internal DisposeScope? RegisterOnCurrentDisposeScope(IDisposable disposable)
        {
            if (this.CurrentDisposeScope is null) {
                StatisticsInstance.CreatedOutsideScopeCount++;
                return null;
            }

            StatisticsInstance.CreatedInScopeCount++;
            this.CurrentDisposeScope.Include(disposable);
            return CurrentDisposeScope;
        }

        internal static DisposeScope NewDisposeScope()
        {
            return ThreadSingleton.InnerNewDisposeScope();
        }

        internal void RemoveDisposeScope(DisposeScope disposeScope)
        {
            var scope = this.CurrentDisposeScope;
            if (object.ReferenceEquals(scope, disposeScope)) {
                this.CurrentDisposeScope = scope.OuterScope;
                return;
            }
            if (scope is null) {
                return;
            }

            for (; ; ) {
                var outerScope = scope.OuterScope;
                if (object.ReferenceEquals(outerScope, disposeScope)) {
                    scope.OuterScope = outerScope.OuterScope;
                    return;
                }

                if (outerScope is null) {
                    return;
                }
                scope = outerScope;
            }
        }

        private DisposeScope InnerNewDisposeScope()
        {
            this.CurrentDisposeScope = new DisposeScope(this);
            return this.CurrentDisposeScope;
        }
    }
}