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
        internal HashSet<IDisposable> Disposables { get; private set; } =
            new HashSet<IDisposable>(ReferenceEqualityComparer<IDisposable>.Default);

        /// <summary>
        /// A view of the disposables in the scope - this list will not be kept in synch with the disposables
        /// in the scope.
        /// </summary>
        public IReadOnlyList<IDisposable> DisposablesView {
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
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes,
        /// use Detach.
        /// </summary>
        public T MoveToOuter<T>(T disposable) where T : IDisposable
        {
            MoveToOuter(new IDisposable[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes,
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
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes,
        /// use Detach.
        /// </summary>
        public (T1 first, T2 second, T3 third) MoveToOuter<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            MoveToOuter(new IDisposable[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes,
        /// use Detach.
        /// </summary>
        public void MoveToOuter(params IDisposable[] disposables) =>
            MoveToOuter((IEnumerable<IDisposable>)disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it up to the outer
        /// dispose scope, if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes,
        /// use Detach.
        /// </summary>
        public void MoveToOuter(IEnumerable<IDisposable> disposables) =>
            MoveToOther(OuterScope, disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all sccopes, use Detach.
        /// </summary>
        public T MoveToOther<T>(DisposeScope? scope, T disposable) where T : IDisposable
        {
            MoveToOther(scope, new IDisposable[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all sccopes, use Detach.
        /// </summary>
        public (T1 first, T2 second) MoveToOther<T1, T2>(DisposeScope? scope, T1 first, T2 second)
            where T1 : IDisposable where T2 : IDisposable
        {
            MoveToOther(scope, new IDisposable[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all sccopes, use Detach.
        /// </summary>
        public (T1 first, T2 second, T3 third) MoveToOther<T1, T2, T3>(DisposeScope? scope, T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            MoveToOther(scope, new IDisposable[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all sccopes, use Detach.
        /// </summary>
        public void MoveToOther(DisposeScope? scope, params IDisposable[] disposables) =>
            MoveToOther(scope, (IEnumerable<IDisposable>)disposables);

        /// <summary>
        /// Excludes a set of tensors/disposables from the current dispose scope, and moves it to another
        /// dispose scope. See overloaded methods. If you wish to exclude a tensor from all sccopes, use Detach.
        /// </summary>
        public void MoveToOther(DisposeScope? scope, IEnumerable<IDisposable> disposables)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            if (scope == null) {
                Detach(disposables);
            } else {
                foreach (var disposable in disposables) {
                    if (Disposables.Remove(disposable)) {
                        AddToOther(scope, disposable);
                    }
                }
            }
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public T Detach<T>(T disposable) where T : IDisposable
        {
            Detach(new IDisposable[] { disposable });
            return disposable;
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public (T1 first, T2 second) Detach<T1, T2>(T1 first, T2 second)
            where T1 : IDisposable where T2 : IDisposable
        {
            Detach(new IDisposable[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public (T1 first, T2 second, T3 third) Detach<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            Detach(new IDisposable[] { first, second, third });
            return (first, second, third);
        }

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public void Detach(params IDisposable[] disposables) => Detach((IEnumerable<IDisposable>)disposables);

        /// <summary>
        /// Detaches/excludes a set of tensors/disposables from the all dispose scopes, see overloaded methods. See MoveToOuter
        /// if you wish to move it to the outer dispose scope.
        /// </summary>
        public void Detach(IEnumerable<IDisposable> disposables)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            foreach (var disposable in disposables) {
                if (Disposables.Remove(disposable)) {
                    if (disposable is torch.Tensor tensor) {
                        _disposeScopeManager.StatisticsInstance.TensorStatistics.DetachedFromScopeCount++;
                        tensor.OwningDisposeScope = null;
                    } else if (disposable is torch.nn.utils.rnn.PackedSequence sequence) {
                        _disposeScopeManager.StatisticsInstance.PackedSequenceStatistics.DetachedFromScopeCount++;
                        sequence.OwningDisposeScope = null;
                    }
                }
            }
        }

        /// <summary>
        /// Replaces registration of one tensor with another.
        /// </summary>
        /// <param name="original">The original tensor, possibly registered under a dispose scope.</param>
        /// <param name="replacement">The replacement tensor.</param>
        internal static void ReplaceWith(torch.Tensor original, torch.Tensor replacement)
        {
            DisposeScope? scope = original.OwningDisposeScope;

            if (scope != null && scope.Disposables.Remove(original)) {
                original.OwningDisposeScope = null;
                AddToOther(scope, replacement);
            }
        }

        public void Attach(IDisposable disposable)
        {
            _ = Attach((IEnumerable<IDisposable>)new[] { disposable });
        }

        public void Attach(params IDisposable[] disposables)
        {
            _ = Attach((IEnumerable<IDisposable>)disposables);
        }

        public IReadOnlyList<IDisposable> Attach(IEnumerable<IDisposable> disposables)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);

            var result = new List<IDisposable>();
            foreach (var disposable in disposables) {
                if (AddToOther(this, disposable)) {
                    if (disposable is torch.Tensor tensor) {
                        _disposeScopeManager.StatisticsInstance.TensorStatistics.AttachedToScopeCount++;
                    } else if (disposable is torch.nn.utils.rnn.PackedSequence sequence) {
                        _disposeScopeManager.StatisticsInstance.PackedSequenceStatistics.AttachedToScopeCount++;
                    }
                }
                result.Add(disposable);
            }

            return result;
        }

        /// <summary>
        /// Disposes everything currently in the dispose scope.
        /// </summary>
        public void DisposeEverything() => DisposeEverythingBut(Enumerable.Empty<IDisposable>());

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public void DisposeEverythingBut(IEnumerable<IDisposable> inKeep)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);
            // Avoiding multiple enumerations
            var oldList = Disposables;
            Disposables = inKeep.ToHashSet(ReferenceEqualityComparer<IDisposable>.Default);
            foreach (var disposable in oldList) {
                if (Disposables.Contains(disposable)) {
                    continue;
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
        public void DisposeEverythingBut(params IDisposable[] keep) =>
            DisposeEverythingBut((IEnumerable<IDisposable>)keep);

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public T DisposeEverythingBut<T>(T keep) where T : IDisposable
        {
            DisposeEverythingBut(new IDisposable[] { keep });
            return keep;
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public (T1 first, T2 second) DisposeEverythingBut<T1, T2>(T1 first, T2 second)
            where T1 : IDisposable where T2 : IDisposable
        {
            DisposeEverythingBut(new IDisposable[] { first, second });
            return (first, second);
        }

        /// <summary>
        /// As an intermediate step, you can dispose all the tensors/disposables currently scheduled for dispose, to
        /// clear up some memory without creating a new scope. Note that this doesn't permanently exclude the
        /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
        /// here.
        /// </summary>
        public (T1 first, T2 second, T3 third) DisposeEverythingBut<T1, T2, T3>(T1 first, T2 second, T3 third)
            where T1 : IDisposable where T2 : IDisposable where T3 : IDisposable
        {
            DisposeEverythingBut(new IDisposable[] { first, second, third });
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
        public void MarkAsDisposed(IDisposable disposable)
        {
            if (this._disposeScopeManager is null)
                throw new ObjectDisposedException(this.GetType().FullName);

            Disposables.Remove(disposable);
            if (disposable is torch.Tensor tensor) {
                tensor.OwningDisposeScope = null;
            }
            else if (disposable is torch.nn.utils.rnn.PackedSequence sequence) {
                sequence.OwningDisposeScope = null;
            }
        }

        /// <summary>
        /// Checks if the DisposeScope contains the disposable
        /// </summary>
        /// <param name="disposable">The disposable that's searched for</param>
        /// <returns></returns>
        public bool Contains(IDisposable disposable) => Disposables.Contains(disposable);

        private static bool AddToOther(DisposeScope scope, IDisposable disposable)
        {
            // if (this._disposeScopeManager is null)
            //     throw new ObjectDisposedException(this.GetType().FullName);

            DisposeScope? oldScope;
            if (disposable is torch.Tensor t) {
                oldScope = t.OwningDisposeScope;
            } else if (disposable is torch.nn.utils.rnn.PackedSequence sequence) {
                oldScope = sequence.OwningDisposeScope;
            } else {
                throw new InvalidOperationException("DisposeScope can only manage Tensor or PackedSequence");
            }

            if (scope == oldScope) return false;

            scope.Disposables.Add(disposable);
            if (oldScope != null) {
                oldScope.Disposables.Remove(disposable);
            }

            if (disposable is torch.Tensor tensor) {
                tensor.OwningDisposeScope = scope;
            } else if (disposable is torch.nn.utils.rnn.PackedSequence sequence) {
                sequence.OwningDisposeScope = scope;
            }

            return true;
        }

        internal HashSet<IDisposable> DetachAllAndDispose()
        {
            var disposables = this.Disposables;
            foreach (var disposable in this.Disposables) {
                if (disposable is torch.Tensor tensor) {
                    this._disposeScopeManager!.StatisticsInstance.TensorStatistics.DetachedFromScopeCount++;
                    tensor.OwningDisposeScope = null;
                } else if (disposable is torch.nn.utils.rnn.PackedSequence sequence) {
                    this._disposeScopeManager!.StatisticsInstance.PackedSequenceStatistics.DetachedFromScopeCount++;
                    sequence.OwningDisposeScope = null;
                }
            }

            this.Disposables = new();
            this.Dispose();

            return disposables;
        }
    }
}