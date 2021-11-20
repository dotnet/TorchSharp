using System;
using System.Collections.Generic;
using System.Linq;

namespace TorchSharp
{
    /// <summary>
    /// Keeps track of all disposables that are in the current scope - the dispose scopes can be nested and the
    /// nesting functionality is mainly managed by DisposeScopeManager.
    /// </summary>
    public class DisposeScope : IDisposable
    {
        private readonly DisposeScopeManager _disposeScopeManager;

        public DisposeScope(DisposeScopeManager disposeScopeManager)
        {
            _disposeScopeManager = disposeScopeManager;
            if (disposeScopeManager.DisposeScopeStack.Count > 0) {
                OuterScope = disposeScopeManager.DisposeScopeStack.Peek();
            }
        }

        /// <summary>
        /// The outer scope, disposables that are moved up from "this" scope will be moved to OuterScope. If OuterScope
        /// is empty then that means this is the root scope, and moving anything up will detach them from the entire
        /// DisposeScope system.
        /// </summary>
        internal DisposeScope OuterScope { get; set; }

        /// <summary>
        /// The disposables that are scheduled for disposing.
        /// </summary>
        /// TODO: There is a ReferenceEqualityComparer coming in .NET 6, use that when available.
        internal HashSet<IDisposable> Disposables { get; private set; } =
            new HashSet<IDisposable>(ReferenceEqualityComparer<IDisposable>.Default);

        /// <summary>
        /// The number of disposables currently held in the scope
        /// </summary>
        public int DisposablesCount => Disposables.Count;

        /// <summary>
        /// A view of the disposables in the scope - this list will not be kept in synch with the disposables
        /// in the scope.
        /// </summary>
        public IReadOnlyList<IDisposable> DisposablesView => Disposables.ToList();

        /// <summary>
        /// Checks if a specific disposable is contained in the scope.
        /// </summary>
        public bool Contains(IDisposable disposable) => Disposables.Contains(disposable);

        /// <summary>
        /// Includes a disposable in the scope - for tensors this is done automatically once the scope has been
        /// created. Use this method to add additional disposables that should be disposed with the scope.
        /// </summary>
        /// <param name="disposable">The disposable to keep in the scope</param>
        /// <returns></returns>
        public T Include<T>(T disposable) where T : IDisposable
        {
            Disposables.Add(disposable);
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public T MoveToOuter<T>(T disposable) where T : IDisposable
        {
            MoveToOuter(new IDisposable[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public (T1 first, T2 second) MoveToOuter<T1, T2>(T1 first, T2 second)
            where T1 : IDisposable where T2 : IDisposable
        {
            MoveToOuter(new IDisposable[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public (T1 first, T2 second, T3 third) DetachFromDisposeScope<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            MoveToOuter(new IDisposable[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public void MoveToOuter(params IDisposable[] disposables) => MoveToOuter((IEnumerable<IDisposable>)disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all scopes,
        /// use Detach.
        /// </summary>
        public void MoveToOuter(IEnumerable<IDisposable> disposables)
        {
            foreach (var disposable in disposables) {
                if (Disposables.Remove(disposable)) {
                    AddToParent(disposable);
                }
            }
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public T Detach<T>(T disposable) where T : IDisposable
        {
            Detach(new IDisposable[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public (T1 first, T2 second) DetachFromDisposeScope<T1, T2>(T1 first, T2 second)
            where T1 : IDisposable where T2 : IDisposable
        {
            Detach(new IDisposable[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public (T1 first, T2 second, T3 third) ExcludeGlobally<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            Detach(new IDisposable[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public void Detach(IDisposable[] disposables) => Detach((IEnumerable<IDisposable>)disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public void Detach(IEnumerable<IDisposable> disposables)
        {
            foreach (var disposable in disposables) {
                if (Disposables.Remove(disposable)) {
                    if (disposable is torch.Tensor tensor) {
                        tensor.OwningDisposeScope = null;
                    }
                }
            }
        }

        /// <summary>
        /// Disposes everything currenly in the dispose scope.
        /// </summary>
        public void DisposeEverything() => DisposeEverythingBut(Enumerable.Empty<IDisposable>());

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope.
        /// </summary>
        public void DisposeEverythingBut(IEnumerable<IDisposable> keep)
        {
            var oldList = Disposables;
            Disposables = keep.ToHashSet(ReferenceEqualityComparer<IDisposable>.Default);
            foreach (var disposable in oldList) {
                if (!Disposables.Contains(disposable)) {
                    disposable.Dispose();
                }
            }
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope.
        /// </summary>
        public T DisposeEverythingBut<T>(T keep) where T : IDisposable
        {
            DisposeEverythingBut(new IDisposable[] { keep });
            return keep;
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope.
        /// </summary>
        public (T1 first, T2 second) DisposeEverythingBut<T1, T2>(T1 first, T2 second)
            where T1 : IDisposable where T2 : IDisposable
        {
            DisposeEverythingBut(new IDisposable[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope.
        /// </summary>
        public (T1 first, T2 second, T3 third) DisposeEverythingBut<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            DisposeEverythingBut(new IDisposable[] { first, second, third });
            return (first, second, third);
        }

        public void Dispose()
        {
            foreach (var disposable in Disposables.ToList()) {
                disposable.Dispose();
            }

            Disposables.Clear();
            _disposeScopeManager.RemoveDisposeScope(this);
        }

        internal void WasDisposed(IDisposable disposable)
        {
            _disposeScopeManager.WasDisposed(disposable);
            Disposables.Remove(disposable);
            if (disposable is torch.Tensor tensor) {
                tensor.OwningDisposeScope = null;
            }
        }

        private void AddToParent(IDisposable disposable)
        {
            if (OuterScope != null) {
                OuterScope.Disposables.Add(disposable);
            }

            if (disposable is torch.Tensor tensor) {
                tensor.OwningDisposeScope = OuterScope;
            }
        }
    }
}