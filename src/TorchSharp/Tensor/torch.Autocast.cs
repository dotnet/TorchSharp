// Copyright (c) .NET Foundation and Contributors. All Rights Reserved. See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Runs eligible operations in a lower precision while leaving model parameters unchanged.
        /// Use around the forward pass and loss calculation, then call backward outside the scope.
        /// </summary>
        /// <remarks>
        /// Autocast state is thread-local. Dispose the returned scope on the thread that created it.
        /// This API does not provide gradient scaling for float16 training.
        /// </remarks>
        public static IDisposable autocast(DeviceType deviceType, ScalarType dtype)
        {
            if (deviceType != DeviceType.CPU && deviceType != DeviceType.CUDA)
                throw new ArgumentOutOfRangeException(nameof(deviceType), "Only CPU and CUDA autocast are supported.");
            if (dtype != ScalarType.BFloat16 && dtype != ScalarType.Float16)
                throw new ArgumentOutOfRangeException(nameof(dtype), "Autocast requires bfloat16 or float16.");
            if (deviceType == DeviceType.CUDA && !cuda.is_available())
                throw new InvalidOperationException("CUDA is not available for autocast.");

            return new AutocastScope(deviceType, dtype);
        }

        /// <summary>Whether autocast is enabled on this thread for the requested device.</summary>
        public static bool is_autocast_enabled(DeviceType deviceType)
        {
            if (deviceType != DeviceType.CPU && deviceType != DeviceType.CUDA)
                throw new ArgumentOutOfRangeException(nameof(deviceType));
            var enabled = THSTorch_autocast_is_enabled((sbyte)deviceType);
            CheckForErrors();
            return enabled;
        }

        // A finalizer cannot restore thread-local autocast state on its own GC thread.
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Usage", "CA2216:Disposable types should declare finalizer",
            Justification = "Autocast must be disposed on the creating thread; a finalizer cannot do that.")]
        private sealed class AutocastScope : IDisposable
        {
            private IntPtr _guard;
            private readonly int _threadId;

            internal AutocastScope(DeviceType deviceType, ScalarType dtype)
            {
                _threadId = Environment.CurrentManagedThreadId;
                _guard = THSTorch_autocast_begin((sbyte)deviceType, (sbyte)dtype);
                if (_guard == IntPtr.Zero)
                {
                    CheckForErrors();
                    throw new InvalidOperationException("Could not enter autocast.");
                }
            }

            public void Dispose()
            {
                if (_guard == IntPtr.Zero) return;
                if (Environment.CurrentManagedThreadId != _threadId)
                    throw new InvalidOperationException("Autocast must be disposed on the thread that entered it.");

                THSTorch_autocast_end(_guard);
                _guard = IntPtr.Zero;
                CheckForErrors();
            }
        }
    }
}
