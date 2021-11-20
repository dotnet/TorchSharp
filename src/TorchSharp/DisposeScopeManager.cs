// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;

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

        static DisposeScopeManager()
        {
            torch.Tensor.OnTensorCreated += TryRegisterInDisposeScope;
            torch.Tensor.BeforeTensorDisposed += TryUnregisterInDisposeScope;
        }

        [ThreadStatic] private static DisposeScopeManager _threadSingleton;

        private static DisposeScopeManager ThreadSingleton => (_threadSingleton ??= new DisposeScopeManager());

        internal Stack<DisposeScope> DisposeScopeStack { get; } = new();
        internal IDisposable CurrentlyDisposing { get; set; }

        /// <summary>
        /// The number of disposables that are currently live on the current thread. It's aproximate, see
        /// Tensor.TotalCount.
        /// </summary>
        public static long ThreadTotalLiveCount => ThreadSingleton._createdCount - ThreadSingleton._disposedCount;

        internal static void TryUnregisterInDisposeScope(IDisposable disposable) =>
            ThreadSingleton.InnerTryUnregisterInDisposeScope(disposable);

        internal static void TryRegisterInDisposeScope(IDisposable disposable) =>
            ThreadSingleton.InnerTryRegisterInDisposeScope(disposable);

        private void InnerTryRegisterInDisposeScope(IDisposable disposable)
        {
            _createdCount++;
            if (DisposeScopeStack.Count > 0) {
                DisposeScopeStack.Peek().Include(disposable);
            }
        }

        private void InnerTryUnregisterInDisposeScope(IDisposable disposable)
        {
            // It's us! We're disposing of it.
            if (disposable == CurrentlyDisposing) {
                return;
            }

            _disposedCount++;
            foreach (var disposeScope in DisposeScopeStack.Reverse()) {
                if (disposeScope.Disposables.Remove(disposable)) {
                    break;
                }
            }
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

        internal void AddToOuterScope(DisposeScope disposeScope, IDisposable disposable)
        {
            var array = DisposeScopeStack.ToArray();
            for (var i = array.Length - 1; i >= 1; i--) {
                if (array[i] == disposeScope) {
                    array[i - 1].Include(disposable);
                    return;
                }
            }
        }

        public class DisposeScope : IDisposable
        {
            private readonly DisposeScopeManager _disposeScopeManager;

            public DisposeScope(DisposeScopeManager disposeScopeManager)
            {
                _disposeScopeManager = disposeScopeManager;
            }

            /// <summary>
            /// The disposables that are scheduled for disposing.
            /// </summary>
            public HashSet<IDisposable> Disposables { get; private set; } = new();

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
            /// use ExcludeGlobally.
            /// </summary>
            public T Exclude<T>(T exclude) where T : IDisposable
            {
                Exclude(new IDisposable[] { exclude }, false);
                return exclude;
            }

            public (T1 first, T2 second) Exclude<T1, T2>(T1 first, T2 second)
                where T1 : IDisposable where T2 : IDisposable
            {
                Exclude(new IDisposable[] { first, second }, false);
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) Exclude<T1, T2, T3>(T1 first, T2 second, T3 third)
                where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
            {
                Exclude(new IDisposable[] { first, second, third }, false);
                return (first, second, third);
            }

            public void Exclude(IEnumerable<IDisposable> exclude, bool excludeGlobally)
            {
                foreach (var disposable in exclude) {
                    if (Disposables.Contains(disposable)) {
                        Disposables.Remove(disposable);
                        if (!excludeGlobally) {
                            ThreadSingleton.AddToOuterScope(this, disposable);
                        }
                    }
                }
            }

            /// <summary>
            /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See Exclude
            /// if you wish to move it to the outer dispose scope.
            /// </summary>
            public T ExcludeGlobally<T>(T exclude) where T : IDisposable
            {
                Exclude(new IDisposable[] { exclude }, true);
                return exclude;
            }

            public (T1 first, T2 second) ExcludeGlobally<T1, T2>(T1 first, T2 second)
                where T1 : IDisposable where T2 : IDisposable
            {
                Exclude(new IDisposable[] { first, second }, true);
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) ExcludeGlobally<T1, T2, T3>(T1 first, T2 second, T3 third)
                where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
            {
                Exclude(new IDisposable[] { first, second, third }, true);
                return (first, second, third);
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
                        DoDispose(disposable, false);
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
                foreach (var disposable in Disposables) {
                    DoDispose(disposable, false);
                }

                Disposables.Clear();
                _disposeScopeManager.RemoveDisposeScope(this);
            }

            private void DoDispose(IDisposable disposable, bool removeFromDisposables = true)
            {
                _disposeScopeManager.CurrentlyDisposing = disposable;
                try {
                    disposable.Dispose();
                    _disposeScopeManager._disposedCount--;
                    if (removeFromDisposables) {
                        Disposables.Remove(disposable);
                    }
                } finally {
                    _disposeScopeManager.CurrentlyDisposing = null;
                }
            }
        }
    }
}