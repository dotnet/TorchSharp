// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;

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
        internal static DisposeScopeManager ThreadSingleton => (_threadSingleton ??= new DisposeScopeManager());

        internal ThreadDisposeScopeStatistics StatisticsInstance { get; } = new ThreadDisposeScopeStatistics();
        internal DisposeScope? CurrentDisposeScope { get; private set; } = null;

        internal DisposeScope? RegisterOnCurrentDisposeScope(IDisposable item)
        {
            if (this.CurrentDisposeScope is null) {
                if (item is torch.Tensor t) {
                    StatisticsInstance.TensorStatistics.CreatedOutsideScopeCount++;
                } else if (item is torch.nn.utils.rnn.PackedSequence p) {
                    StatisticsInstance.PackedSequenceStatistics.CreatedOutsideScopeCount++;
                }
                return null;
            } else {
                if (item is torch.Tensor t) {
                    StatisticsInstance.TensorStatistics.CreatedInScopeCount++;
                } else if (item is torch.nn.utils.rnn.PackedSequence p) {
                    StatisticsInstance.PackedSequenceStatistics.CreatedInScopeCount++;
                }
                this.CurrentDisposeScope.Disposables.Add(item);
                return CurrentDisposeScope;
            }
        }

        internal void DisposingOnCurrentScope(torch.Tensor item)
        {
            if (item.OwningDisposeScope == null) {
                StatisticsInstance.TensorStatistics.DisposedOutsideScopeCount++;
            } else {
                StatisticsInstance.TensorStatistics.DisposedInScopeCount++;
            }
        }
        internal void DisposingOnCurrentScope(torch.nn.utils.rnn.PackedSequence item)
        {
            if (item.OwningDisposeScope == null) {
                StatisticsInstance.PackedSequenceStatistics.DisposedOutsideScopeCount++;
            } else {
                StatisticsInstance.PackedSequenceStatistics.DisposedInScopeCount++;
            }
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

        internal DisposeScope NewDisposeScope()
        {
            this.CurrentDisposeScope = new DisposeScope(this);
            return this.CurrentDisposeScope;
        }

        public static ThreadDisposeScopeStatistics Statistics => ThreadSingleton.StatisticsInstance;
    }
}