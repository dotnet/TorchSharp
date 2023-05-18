// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable

namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Random Number Generator
        /// </summary>
        public class Generator : IDisposable
        {
            public IntPtr Handle { get; private set; }

            /// <summary>
            /// Gets the current device of the generator.
            /// </summary>
            public torch.Device device { get; private set; }

            /// <summary>
            /// Returns the Generator state as a torch.ByteTensor.
            /// </summary>
            /// <returns></returns>
            public Tensor get_state()
            {
                var res = THSGenerator_get_rng_state(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Sets the Generator state.
            /// </summary>
            /// <param name="value">The desired state.</param>
            public void set_state(Tensor value)
            {
                THSGenerator_set_rng_state(Handle, value is null ? IntPtr.Zero : value.Handle);
                torch.CheckForErrors();
            }

            public Generator manual_seed(long seed)
            {
                torch.TryInitializeDeviceType(DeviceType.CUDA);
                THSGenerator_gen_manual_seed(Handle, seed);
                return this;
            }

            /// <summary>
            /// Gets a non-deterministic random number from std::random_device or the current time and uses it to seed a Generator.
            /// </summary>
            /// <returns></returns>
            public long seed()
            {
                long seed = DateTime.UtcNow.Ticks;
                torch.TryInitializeDeviceType(DeviceType.CUDA);
                THSGenerator_gen_manual_seed(Handle, seed);
                return seed;
            }

            internal Generator(IntPtr nativeHandle)
            {
                Handle = nativeHandle;
                device = torch.CPU;
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="seed">An initial seed to use with the generator.</param>
            /// <param name="device">The desired device.</param>
            public Generator(ulong seed = 0, torch.Device? device = null) :
                this(THSGenerator_new(seed, (long)(device?.type ?? DeviceType.CPU), device?.index ?? -1))
            {
                this.device = device ?? torch.CPU;
            }

            public static Generator Default {
                get {
                    return new Generator(THSGenerator_default_generator());
                }
            }

            /// <summary>
            /// Returns the initial seed for generating random numbers.
            /// </summary>
            /// <returns></returns>
            public long initial_seed()
            {
                return THSGenerator_initial_seed(Handle);
            }

            #region Dispose() support
            protected virtual void Dispose(bool disposing)
            {
                if (Handle != IntPtr.Zero) {
                    var h = Handle;
                    Handle = IntPtr.Zero;
                    THSGenerator_dispose(h);
                }
            }

            ~Generator()
            {
                // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
                Dispose(disposing: false);
            }

            public void Dispose()
            {
                // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
                Dispose(disposing: true);
                GC.SuppressFinalize(this);
            }
            #endregion
        }
    }
}
