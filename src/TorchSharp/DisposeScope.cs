// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Utils;

#nullable enable
namespace TorchSharp
{
    /// <summary>
    /// Keeps track of all disposables that are in the current scope - the dispose scopes can be nested and the
    /// nesting functionality is mainly managed by DisposeScopeManager.
    /// </summary>
    public sealed class DisposeScope : IDisposable
    {
        private DisposeScopeManager? _disposeScopeManager;

        internal DisposeScope(DisposeScopeManager disposeScopeManager)
        {
            _disposeScopeManager = disposeScopeManager;
            this.OuterScope = disposeScopeManager.CurrentDisposeScope;
        }

        /// <summary>
        /// The outer scope with relation to this scope.
        /// </summary>
        internal DisposeScope? OuterScope { get; set; }

        /// <summary>
        /// The disposables that are scheduled for disposing.
        /// </summary>
        internal HashSet<IDisposeScopeClient> Disposables { get; private set; } =
            new HashSet<IDisposeScopeClient>(ReferenceEqualityComparer<IDisposeScopeClient>.Default);

        /// <summary>
        /// A view of the disposables in the scope - this list will not be kept in synch with the disposables
        /// in the scope.
        /// </summary>
        public IReadOnlyList<IDisposeScopeClient> DisposablesView {
            get {
                if (this._disposeScopeManager is null)
                    throw new ObjectDisposedException(this.GetType().FullName);
                return Disposables.ToArray();
            }
        }

        /// <summary>
        /// The number of disposables currently held in the scope
        /// </summary>
        public int DisposablesCount {
            get {
                if (this._disposeScopeManager is null)
                    throw new ObjectDisposedException(this.GetType().FullName);
                return Disposables.Count;
            }
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public T MoveToOuter<T>(T disposable) where T : IDisposeScopeClient
        {
            MoveToOuter(new IDisposeScopeClient[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public (T1 first, T2 second) MoveToOuter<T1, T2>(T1 first, T2 second)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient
        {
            MoveToOuter(new IDisposeScopeClient[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public (T1 first, T2 second, T3 third) MoveToOuter<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient where T3 : IDisposeScopeClient
        {
            MoveToOuter(new IDisposeScopeClient[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public void MoveToOuter(params IDisposeScopeClient[] disposables) =>
            MoveToOuter((IEnumerable<IDisposeScopeClient>)disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public void MoveToOuter(IEnumerable<IDisposeScopeClient> disposables) =>
            MoveToOther(OuterScope, disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all scopes, use Detach.
        /// </summary>
        public T MoveToOther<T>(DisposeScope? scope, T disposable) where T : IDisposeScopeClient
        {
            MoveToOther(scope, new IDisposeScopeClient[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all scopes, use Detach.
        /// </summary>
        public (T1 first, T2 second) MoveToOther<T1, T2>(DisposeScope? scope, T1 first, T2 second)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient
        {
            MoveToOther(scope, new IDisposeScopeClient[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all scopes, use Detach.
        /// </summary>
        public (T1 first, T2 second, T3 third) MoveToOther<T1, T2, T3>(DisposeScope? scope, T1 first, T2 second, T3 third)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient where T3 : IDisposeScopeClient
        {
            MoveToOther(scope, new IDisposeScopeClient[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all scopes, use Detach.
        /// </summary>
        public void MoveToOther(DisposeScope? scope, params IDisposeScopeClient[] disposables) =>
            MoveToOther(scope, (IEnumerable<IDisposeScopeClient>)disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all scopes, use Detach.
        /// </summary>
        public void MoveToOther(DisposeScope? scope, IEnumerable<IDisposeScopeClient> disposables)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            foreach (var disposable in disposables) {
                if (Disposables.Remove(disposable)) {
                    AddToOther(scope, disposable);
                }
            }
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public T Detach<T>(T disposable) where T : IDisposeScopeClient
        {
            Detach(new IDisposeScopeClient[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public (T1 first, T2 second) Detach<T1, T2>(T1 first, T2 second)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient
        {
            Detach(new IDisposeScopeClient[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public (T1 first, T2 second, T3 third) Detach<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient where T3 : IDisposeScopeClient
        {
            Detach(new IDisposeScopeClient[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public void Detach(params IDisposeScopeClient[] disposables) => Detach((IEnumerable<IDisposeScopeClient>)disposables);

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public void Detach(IEnumerable<IDisposeScopeClient> disposables)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            foreach (var disposable in disposables) {
                if (Disposables.Remove(disposable)) {
                    _disposeScopeManager.StatisticsInstance.DetachedFromScopeCount++;
                    disposable.OwningDisposeScope = null;
                }
            }
        }

        public void Attach(IDisposeScopeClient disposable)
        {
            _ = Attach((IEnumerable<IDisposeScopeClient>)new[] { disposable });
        }

        public void Attach(params IDisposeScopeClient[] disposables)
        {
            _ = Attach((IEnumerable<IDisposeScopeClient>)disposables);
        }

        public IReadOnlyList<IDisposeScopeClient> Attach(IEnumerable<IDisposeScopeClient> disposables)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);

            var result = new List<IDisposeScopeClient>();
            foreach (var disposable in disposables) {
                if (disposable.OwningDisposeScope == null && !disposable.IsInvalid) {
                    _disposeScopeManager.StatisticsInstance.DetachedFromScopeCount--;
                }

                AddToOther(this, disposable);
                result.Add(disposable);
            }

            return result;
        }

        /// <summary>
        /// Disposes everything currently in the dispose scope.
        /// </summary>
        public void DisposeEverything() => DisposeEverythingBut(Enumerable.Empty<IDisposeScopeClient>());

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public void DisposeEverythingBut(IEnumerable<IDisposeScopeClient> inKeep)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            // Avoiding multiple enumerations
            var oldList = Disposables;
            Disposables = inKeep.ToHashSet(ReferenceEqualityComparer<IDisposeScopeClient>.Default);
            foreach (var disposable in oldList) {
                if (Disposables.Contains(disposable)) {
                    continue;
                }
                // No need to have the disposable call back to the scope
                disposable.OwningDisposeScope = null;
                if (!disposable.IsInvalid) {
                    _disposeScopeManager.StatisticsInstance.DisposedInScopeCount++;
                }
                disposable.Dispose();
            }
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public void DisposeEverythingBut(params IDisposeScopeClient[] keep) =>
            DisposeEverythingBut((IEnumerable<IDisposeScopeClient>)keep);

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public T DisposeEverythingBut<T>(T keep) where T : IDisposeScopeClient
        {
            DisposeEverythingBut(new IDisposeScopeClient[] { keep });
            return keep;
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public (T1 first, T2 second) DisposeEverythingBut<T1, T2>(T1 first, T2 second)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient
        {
            DisposeEverythingBut(new IDisposeScopeClient[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public (T1 first, T2 second, T3 third) DisposeEverythingBut<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposeScopeClient where T2 : IDisposeScopeClient where T3 : IDisposeScopeClient
        {
            DisposeEverythingBut(new IDisposeScopeClient[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Disposes of the DisposeScope and all the disposables in its list. You would typically not call this method,
        /// instead you should use a usings clause around the scope.
        /// </summary>
        public void Dispose()
        {
            if (this._disposeScopeManager is null)
                return;
            DisposeEverything();
            _disposeScopeManager.RemoveDisposeScope(this);
            this._disposeScopeManager = null;
        }

        /// <summary>
        /// A method that notifies the DisposeScope that a disposable was disposed, so that it can be removed from the
        /// tracked list. This will be called if a tensor is manually disposed, but you can also add your own
        /// disposables to the dispose scope. If you do, and dispose them manually, you should make sure to call this
        /// method.
        /// </summary>
        /// <param name="disposable">The disposable that was disposed</param>
        public void MarkAsDisposed(IDisposeScopeClient disposable)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            _disposeScopeManager.StatisticsInstance.DisposedInScopeCount++;
            Disposables.Remove(disposable);
            disposable.OwningDisposeScope = null;
        }

        /// <summary>
        /// Checks if the DisposeScope contains the disposable
        /// </summary>
        /// <param name="disposable">The disposable that's searched for</param>
        /// <returns></returns>
        public bool Contains(IDisposeScopeClient disposable) => Disposables.Contains(disposable);

        private void AddToOther(DisposeScope? scope, IDisposeScopeClient disposable)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            if (scope != null) {
                scope.Disposables.Add(disposable);
            } else {
                _disposeScopeManager.StatisticsInstance.DetachedFromScopeCount++;
            }

            disposable.OwningDisposeScope = scope;
        }

        internal HashSet<IDisposeScopeClient> DetachAllAndDispose()
        {
            var disposables = this.Disposables;
            foreach (var disposable in this.Disposables) {
                this._disposeScopeManager!.StatisticsInstance.DetachedFromScopeCount++;
                disposable.OwningDisposeScope = null;
            }

            this.Disposables = new();
            this.Dispose();

            return disposables;
        }
    }
}