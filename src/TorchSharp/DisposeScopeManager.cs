using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TorchSharp
{
    /// <summary>
    /// Manages dispose scopes, that can make automatic tensor disposal easier. Note that the
    /// DisposeManager isn't "thread safe", since all CPU Devices are the same, the Dispose manager
    /// would pick up tensors from other threads. Therefore, each thread gets it's own DisposeScopeManager.
    /// </summary>
    public class DisposeScopeManager
    {
        [ThreadStatic] public static DisposeScopeManager Singleton;
        internal static int _disposedTensorCount;

        internal Dictionary<string, Stack<DisposeScope>> DisposeScopeStacks { get; } = new();
        internal DisposeScope NewDisposeScope(torch.Tensor tensor) => NewDisposeScope(tensor.device);

        public static int DisposedTensorCount {
            get => _disposedTensorCount;
            set => _disposedTensorCount = value;
        }

        internal static void TryRegisterInDisposeScope(torch.Tensor tensor)
        {
            lock (Singleton ??= new DisposeScopeManager()) {
                var stack = Singleton.GetStackForDevice(tensor);
                if (stack?.Count > 0) {
                    stack.Peek().Include(tensor);
                }
            }
        }

        internal static DisposeScope NewDisposeScope(torch.Device device)
        {
            lock (Singleton ??= new DisposeScopeManager()) {
                var stack = Singleton.GetStackForDevice(device);
                var disposeScope = new DisposeScope(device);
                stack.Push(disposeScope);
                return disposeScope;
            }
        }

        internal void RemoveDisposeScope(DisposeScope disposeScope)
        {
            lock (this) {
                var stack = GetStackForDevice(disposeScope.Device);
                if (stack.Count == 0) {
                    throw new InvalidOperationException($"Device {disposeScope.Device} has no DisposeScopes! ");
                }

                if (stack.Peek() != disposeScope) {
                    throw new InvalidOperationException(
                        $"DisposeScope for {disposeScope.Tensors} disposed out of order!");
                }

                stack.Pop();
            }
        }

        internal void MoveToOuterScope(DisposeScope disposeScope, torch.Tensor tensor)
        {
            lock (this) {
                var stack = GetStackForDevice(disposeScope.Device);
                var array = stack.ToArray();
                for (var i = array.Length - 1; i >= 1; i--) {
                    if (array[i] == disposeScope) {
                        array[i - 1].Include(tensor);
                        return;
                    }
                }
            }
        }

        private Stack<DisposeScope> GetStackForDevice(torch.Device device) =>
            GetStackForDevice(device.type, device.index);

        private Stack<DisposeScope> GetStackForDevice(torch.Tensor tensor) =>
            GetStackForDevice(tensor.device_type, tensor.device_index);

        private Stack<DisposeScope> GetStackForDevice(DeviceType deviceType, int deviceIndex)
        {
            var key = deviceType == DeviceType.CPU ? "CPU" : deviceType + "/" + deviceIndex;

            if (!DisposeScopeStacks.TryGetValue(key, out var stack)) {
                stack = new Stack<DisposeScope>();
                DisposeScopeStacks.Add(key, stack);
            }

            return stack;
        }

        public class DisposeScope : IDisposable
        {
            public DisposeScope(torch.Device device)
            {
                Device = device;
            }

            /// <summary>
            /// The device that the dispose scope refers to.
            /// </summary>
            public torch.Device Device { get; }

            /// <summary>
            /// The tensors that are scheduled for disposing.
            /// </summary>
            public List<torch.Tensor> Tensors { get; } = new();

            /// <summary>
            /// Includes a tensor in the scope - this is done automatically once the scope has been created.
            /// Use this method to add additional tensors that should be disposed.
            /// </summary>
            /// <param name="tensor">The tensor to be disposed</param>
            /// <returns></returns>
            public torch.Tensor Include(torch.Tensor tensor)
            {
                lock (this) {
                    Tensors.Add(tensor);
                }

                return tensor;
            }

            /// <summary>
            /// Excludes a set of tensors from the current dispose scope, and moves it up to the outer dispose scope,
            /// if one exists. See overloaded methods. If you wish to exclude a tensor from all sccopes, use
            /// ExcludeGlobally.
            /// </summary>
            public torch.Tensor Exclude(torch.Tensor tensorToExclude) =>
                Exclude(new[] { tensorToExclude }, false)[0];

            public (torch.Tensor first, torch.Tensor second) Exclude(torch.Tensor first, torch.Tensor second)
            {
                Exclude(new[] { first, second }, false);
                return (first, second);
            }

            public (torch.Tensor first, torch.Tensor second, torch.Tensor third) Exclude(torch.Tensor first,
                torch.Tensor second, torch.Tensor third)
            {
                Exclude(new[] { first, second, third }, false);
                return (first, second, third);
            }

            /// <summary>
            /// Excludes a set of tensors from the all dispose scopes, see overloaded methods. See Exclude if you
            /// wish to move it to the outer dispose scope.
            /// </summary>
            public torch.Tensor ExcludeGlobally(torch.Tensor tensorToExclude) =>
                Exclude(new[] { tensorToExclude }, true)[0];

            public (torch.Tensor first, torch.Tensor second) ExcludeGlobally(torch.Tensor first, torch.Tensor second)
            {
                Exclude(new[] { first, second }, true);
                return (first, second);
            }

            public (torch.Tensor first, torch.Tensor second, torch.Tensor third) ExcludeGlobally(torch.Tensor first,
                torch.Tensor second, torch.Tensor third)
            {
                Exclude(new[] { first, second, third }, true);
                return (first, second, third);
            }


            /// <summary>
            /// As an intermediate step, you can dispose all the tensors currently scheduled for dispose, to clear
            /// up some memory without creating a new scope. Note that this doesn't permanently exclude the
            /// tensors from disposing, use Exclude for that. Also, excluded tensors don't need to be included
            /// here.
            /// </summary>
            public torch.Tensor[] DisposeEverythingBut(params torch.Tensor[] tensorsToKeep)
            {
                lock (this) {
                    // This code looks odd, but sometimes a tensor has been disposed somewhere else, and once it's
                    // disposed, we can't use the Equals method on it anymore. Therefore we must resort to
                    // ReferenceEquals instead.
                    foreach (torch.Tensor tensor in tensorsToKeep) {
                        if (!Tensors.Any(x => ReferenceEquals(tensor, x))) {
                            throw new InvalidOperationException("The tensor does not belong to this scope!");
                        }
                    }

                    for (var i = Tensors.Count - 1; i >= 0; i--) {
                        var tensor = Tensors[i];
                        if (!tensorsToKeep.Any(x => ReferenceEquals(tensor, x))) {
                            if (tensor.handle != IntPtr.Zero) {
                                Interlocked.Increment(ref _disposedTensorCount);
                                tensor.Dispose();
                            }

                            Tensors.RemoveAt(i);
                        }
                    }
                }

                return tensorsToKeep;
            }

            public torch.Tensor DisposeEverythingBut(torch.Tensor tensorToKeep) =>
                DisposeEverythingBut(new[] { tensorToKeep })[0];

            public (torch.Tensor first, torch.Tensor second) DisposeEverythingBut(torch.Tensor first,
                torch.Tensor second)
            {
                DisposeEverythingBut(new[] { first, second });
                return (first, second);
            }

            public (torch.Tensor first, torch.Tensor second, torch.Tensor third) DisposeEverythingBut(
                torch.Tensor first, torch.Tensor second, torch.Tensor third)
            {
                DisposeEverythingBut(new[] { first, second, third });
                return (first, second, third);
            }

            public void Dispose()
            {
                ReleaseUnmanagedResources();
                GC.SuppressFinalize(this);
            }

            private void ReleaseUnmanagedResources()
            {
                lock (DisposeScopeManager.Singleton) {
                    foreach (var tensor in Tensors) {
                        if (tensor.handle != IntPtr.Zero) {
                            Interlocked.Increment(ref _disposedTensorCount);
                            tensor.Dispose();
                        }
                    }

                    Tensors.Clear();
                    DisposeScopeManager.Singleton.RemoveDisposeScope(this);
                }
            }

            private torch.Tensor[] Exclude(torch.Tensor[] tensorsToExclude, bool excludeGlobally)
            {
                lock (this) {
                    foreach (torch.Tensor tensor in tensorsToExclude) {
                        if (!Tensors.Any(existing => ReferenceEquals(existing, tensor))) {
                            throw new InvalidOperationException("The tensor does not belong to this scope!");
                        }
                    }

                    for (var i = Tensors.Count - 1; i >= 0; i--) {
                        var tensor = Tensors[i];
                        if (tensorsToExclude.Any(x => ReferenceEquals(tensor, x))) {
                            Tensors.RemoveAt(i);

                            if (!excludeGlobally) {
                                Singleton.MoveToOuterScope(this, tensor);
                            }
                        }
                    }
                }

                return tensorsToExclude;
            }

            ~DisposeScope()
            {
                ReleaseUnmanagedResources();
            }
        }
    }
}