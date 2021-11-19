using System;
using System.Collections.Generic;
using System.Drawing;
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
        internal static int _disposedTensorCount;
        internal static int _disposedCount;

        static DisposeScopeManager()
        {
            torch.Tensor.OnTensorCreateCallback.Add(TryRegisterInDisposeScope);
            torch.Tensor.OnTensorDisposeCallback.Add(TryUnregisterInDisposeScope);
        }

        [ThreadStatic] private static DisposeScopeManager _singleton;

        public static DisposeScopeManager Singleton = (_singleton ??= new DisposeScopeManager());

        internal Stack<DisposeScope> DisposeScopeStack { get; } = new();
        internal IDisposable CurrentlyDisposing { get; set; }

        public static int DisposedTensorCount => _disposedTensorCount;
        public static int DisposedCount => _disposedCount;

        public static long TotalAllocatedSize => Singleton.InnerTotalAllocatedSize;

        internal static void TryUnregisterInDisposeScope(IDisposable disposable)=>
            Singleton.InnerTryUnregisterInDisposeScope(disposable);

        internal static void TryRegisterInDisposeScope(IDisposable disposable) =>
            Singleton.InnerTryRegisterInDisposeScope(disposable);

        private long InnerTotalAllocatedSize {
            get {
                long size = 0;
                foreach (var disposeScope in DisposeScopeStack) {
                    size += disposeScope.TotalAllocatedSize;
                }
                return size;
            }
        }

        private void InnerTryRegisterInDisposeScope(IDisposable disposable)
        {
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
            foreach (var disposeScope in DisposeScopeStack) {
                disposeScope.Disposables.Remove(disposable);
            }
        }

        internal static DisposeScope NewDisposeScope()
        {
            lock (Singleton ??= new DisposeScopeManager()) {
                return Singleton.InnerNewDisposeScope();
            }
        }

        private DisposeScope InnerNewDisposeScope()
        {
            var disposeScope = new DisposeScope(this);
            DisposeScopeStack.Push(disposeScope);
            return disposeScope;
        }

        internal void RemoveDisposeScope(DisposeScope disposeScope)
        {
            lock (this) {
                if (DisposeScopeStack.Count == 0) {
                    throw new InvalidOperationException($"There are no DisposeScopes! ");
                }

                if (DisposeScopeStack.Peek() != disposeScope) {
                    throw new InvalidOperationException(
                        $"DisposeScope for {disposeScope.Disposables} disposed out of order!");
                }

                DisposeScopeStack.Pop();
            }
        }

        internal void MoveToOuterScope(DisposeScope disposeScope, IDisposable disposable)
        {
            lock (this) {
                var array = DisposeScopeStack.ToArray();
                for (var i = array.Length - 1; i >= 1; i--) {
                    if (array[i] == disposeScope) {
                        array[i - 1].Include(disposable);
                        return;
                    }
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
            public HashSet<IDisposable> Disposables { get; } = new();

            public long TotalAllocatedSize =>
                Disposables.OfType<torch.Tensor>().Sum(x => x.NumberOfElements * x.ElementSize);

            /// <summary>
            /// Includes a disposable in the scope - for tensors this is done automatically once the scope has been
            /// created. Use this method to add additional disposables that should be disposed, but you typically
            /// don't need to call this method.
            /// </summary>
            /// <param name="disposable">The disposable to keep in the scope</param>
            /// <returns></returns>
            public T Include<T>(T disposable) where T:IDisposable
            {
                lock (this) {
                    Disposables.Add(disposable);
                }

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

            public (T1 first, T2 second) Exclude<T1,T2>(T1 first, T2 second) where T1:IDisposable where T2:IDisposable
            {
                Exclude(new IDisposable[] { first, second }, false);
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) Exclude<T1,T2,T3>(T1 first, T2 second, T3 third)
                where T1:IDisposable where T2:IDisposable where T3:IDisposable
            {
                Exclude(new IDisposable[] { first, second, third }, false);
                return (first, second, third);
            }

            /// <summary>
            /// Excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See Exclude
            /// if you wish to move it to the outer dispose scope.
            /// </summary>
            public T ExcludeGlobally<T>(T exclude)  where T:IDisposable
            {
                Exclude(new IDisposable[] { exclude }, true);
                return exclude;
            }

            public (T1 first, T2 second) ExcludeGlobally<T1,T2>(T1 first, T2 second)
                where T1:IDisposable where T2:IDisposable
            {
                Exclude(new IDisposable[] { first, second }, true);
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) ExcludeGlobally<T1,T2,T3>(T1 first, T2 second, T3 third)
                where T1:IDisposable where T2:IDisposable where T3:IDisposable
            {
                Exclude(new IDisposable[] { first, second, third }, true);
                return (first, second, third);
            }

            /// <summary>
            /// Disposes everything currenly in the dispose scope.
            /// </summary>
            public void DisposeEverything() => DisposeEverythingBut();

            /// <summary>
            /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
            /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
            /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
            /// here.
            /// </summary>
            public IDisposable[] DisposeEverythingBut(params IDisposable[] keep)
            {
                lock (this) {
                    foreach (var disposable in keep) {
                        if (!Disposables.Contains(disposable)) {
                            throw new InvalidOperationException("The disposable does not belong to this scope!");
                        }
                    }

                    var toDispose = Disposables.Where(disposable => !keep.Contains(disposable)).ToList();
                    foreach (var disposable in toDispose) {
                        DoDispose(disposable);
                    }
                }

                return keep;
            }

            public T DisposeEverythingBut<T>(T keep) where T:IDisposable
            {
                DisposeEverythingBut(new IDisposable[] { keep });
                return keep;
            }

            public (T1 first, T2 second) DisposeEverythingBut<T1,T2>(T1 first, T2 second)
                where T1: IDisposable where T2:IDisposable
            {
                DisposeEverythingBut(new IDisposable[] { first, second });
                return (first, second);
            }

            public (T1 first, T2 second, T3 third) DisposeEverythingBut<T1,T2,T3>(T1 first, T2 second, T3 third)
                where T1:IDisposable where T2:IDisposable where T3:IDisposable
            {
                DisposeEverythingBut(new IDisposable[] { first, second, third });
                return (first, second, third);
            }

            public void Dispose()
            {
                ReleaseUnmanagedResources();
                GC.SuppressFinalize(this);
            }

            private void ReleaseUnmanagedResources()
            {
                lock (Singleton) {
                    foreach (var disposable in Disposables) {
                        DoDispose(disposable);
                    }

                    Singleton.RemoveDisposeScope(this);
                }
            }

            private IDisposable _currentlyDisposing=null;
            private void DoDispose(IDisposable disposable)
            {
                _currentlyDisposing = disposable;
                if (disposable is torch.Tensor tensor)
                {
                    if (!tensor.IsDisposed)
                    {
                        Interlocked.Increment(ref _disposedCount);
                        Interlocked.Increment(ref _disposedTensorCount);
                        tensor.Dispose();
                    }
                }
                else
                {
                    Interlocked.Increment(ref _disposedCount);
                    disposable.Dispose();
                }

                _currentlyDisposing = null;
                Disposables.Remove(disposable);
            }

            private IDisposable[] Exclude(IDisposable[] exclude, bool excludeGlobally)
            {
                lock (this) {
                    foreach (var disposable in exclude) {
                        if (!Disposables.Contains(disposable)) {
                            throw new InvalidOperationException("The disposable does not belong to this scope!");
                        }

                        Disposables.Remove(disposable);
                        if (!excludeGlobally) {
                            Singleton.MoveToOuterScope(this, disposable);
                        }
                    }
                }

                return exclude;
            }

            ~DisposeScope()
            {
                ReleaseUnmanagedResources();
            }
        }
    }
}