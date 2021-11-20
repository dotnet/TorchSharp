// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace TorchSharp
{
    /// <summary>
    /// Manages dispose scopes, that can make automatic tensor disposal easier. Note that the
    /// DisposeManager is thread local. The DisposeScopeManager can also manage other disposables, such as training
    /// batches and the like.
    /// </summary>
    public class DisposeScopeManager
    {
        private long _createdCount;
        private long _disposedCount;

        [ThreadStatic] private static DisposeScopeManager _threadSingleton;

        internal static DisposeScopeManager ThreadSingleton => (_threadSingleton ??= new DisposeScopeManager());

        internal Stack<DisposeScope> DisposeScopeStack { get; } = new();
        internal IDisposable CurrentlyDisposing { get; set; }

        /// <summary>
        /// The number of disposables that are currently live on the current thread. It's aproximate, see
        /// Tensor.TotalCount.
        /// </summary>
        public static long ThreadTotalLiveCount => ThreadSingleton._createdCount - ThreadSingleton._disposedCount;

        internal DisposeScope RegisterOnCurrentDisposeScope(IDisposable disposable)
        {
            _createdCount++;
            if (DisposeScopeStack.Count == 0) {
                return null;
            }

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

        public class DisposeScope : IDisposable
        {
            private readonly DisposeScopeManager _disposeScopeManager;

            public DisposeScope(DisposeScopeManager disposeScopeManager)
            {
                _disposeScopeManager = disposeScopeManager;
                if (disposeScopeManager.DisposeScopeStack.Count > 0) {
                    ParentDisposeScope = disposeScopeManager.DisposeScopeStack.Peek();
                }
            }

            /// <summary>
            /// The parente scope of this disposable.
            /// </summary>
            internal DisposeScope ParentDisposeScope { get; set; }

            /// <summary>
            /// The disposables that are scheduled for disposing.
            /// </summary>
            internal HashSet<IDisposable> Disposables { get; private set; } = new();

            /// <summary>
            /// A view of the disposables in the scope - this list will not be kept in synch with the disposables
            /// in the scope.
            /// </summary>
            public IReadOnlyList<IDisposable> DisposablesView => Disposables.ToList();

            /// <summary>
            /// The number of disposables currently held in the scope
            /// </summary>
            public int DisposablesCount => Disposables.Count;

            /// <summary>
            /// Includes a disposable in the scope - for tensors this is done automatically once the scope has been
            /// created. Use this method to add additional disposables that should be disposed, but you typically
            /// don't need to call this method.
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
            /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes,
            /// use Detach.
            /// </summary>
            public T MoveToOuter<T>(T disposable) where T : IDisposable
            {
                MoveToOuter(new IDisposable[] { disposable });
                return disposable;
            }

            public (T1 first, T2 second) MoveToOuter<T1, T2>(T1 first, T2 second)
                where T1 : IDisposable where T2 : IDisposable
            {
                MoveToOuter(new IDisposable[] { first, second });
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) DetachFromDisposeScope<T1, T2, T3>(T1 first, T2 second, T3 third)
                where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
            {
                MoveToOuter(new IDisposable[] { first, second, third });
                return (first, second, third);
            }

            public void MoveToOuter(params IDisposable[] disposables) =>
                MoveToOuter((IEnumerable<IDisposable>)disposables);

            public void MoveToOuter(IEnumerable<IDisposable> disposables)
            {
                foreach (var disposable in disposables) {
                    if (Disposables.Contains(disposable)) {
                        Disposables.Remove(disposable);
                        AddToParent(disposable);
                    }
                }
            }

            /// <summary>
            /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See Exclude
            /// if you wish to move it to the outer dispose scope.
            /// </summary>
            public T Detach<T>(T disposable) where T : IDisposable
            {
                Detach(new IDisposable[] { disposable });
                return disposable;
            }

            public (T1 first, T2 second) DetachFromDisposeScope<T1, T2>(T1 first, T2 second)
                where T1 : IDisposable where T2 : IDisposable
            {
                Detach(new IDisposable[] { first, second });
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) ExcludeGlobally<T1, T2, T3>(T1 first, T2 second, T3 third)
                where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
            {
                Detach(new IDisposable[] { first, second, third });
                return (first, second, third);
            }

            public void Detach(IDisposable[] disposables) => Detach((IEnumerable<IDisposable>)disposables);

            public void Detach(IEnumerable<IDisposable> disposables)
            {
                foreach (var disposable in disposables) {
                    if (Disposables.Contains(disposable)) {
                        Disposables.Remove(disposable);
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
            /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
            /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
            /// here.
            /// </summary>
            public void DisposeEverythingBut(IEnumerable<IDisposable> keep)
            {
                var oldList = Disposables;
                Disposables = keep.ToHashSet();
                foreach (var disposable in oldList) {
                    if (!Disposables.Contains(disposable)) {
                        disposable.Dispose();
                    }
                }
            }

            public T DisposeEverythingBut<T>(T keep) where T : IDisposable
            {
                DisposeEverythingBut(new IDisposable[] { keep });
                return keep;
            }

            public (T1 first, T2 second) DisposeEverythingBut<T1, T2>(T1 first, T2 second)
                where T1 : IDisposable where T2 : IDisposable
            {
                DisposeEverythingBut(new IDisposable[] { first, second });
                return (first, second);
            }

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

            public void WasDisposed(IDisposable disposable)
            {
                _disposeScopeManager._disposedCount++;
                Disposables.Remove(disposable);
            }

            public bool Contains(IDisposable disposable)
            {
                if (disposable is torch.Tensor tensor && tensor.IsInvalid) {
                    return false;
                }

                return Disposables.Contains(disposable);
            }

            private void AddToParent(IDisposable disposable)
            {
                if (ParentDisposeScope != null) {
                    ParentDisposeScope.Disposables.Add(disposable);
                }

                if (disposable is torch.Tensor tensor) {
                    tensor.OwningDisposeScope = ParentDisposeScope;
                }
            }
        }
    }
}