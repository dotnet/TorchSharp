// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
    public static partial class torch
    {
        public class Generator : IDisposable
        {
            public IntPtr Handle { get; private set; }

            public torch.Device device { get; private set; }

            public Tensor get_state()
            {
                var res = THSGenerator_get_rng_state(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public void set_state(Tensor value)
            {
                THSGenerator_set_rng_state(Handle, value.Handle);
                torch.CheckForErrors();
            }

            public Generator manual_seed(long seed)
            {
                torch.TryInitializeDeviceType(DeviceType.CUDA);
                THSGenerator_gen_manual_seed(Handle, seed);
                return this;
            }

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

            [DllImport("LibTorchSharp")]
            extern static long THSGenerator_initial_seed(IntPtr handle);

            public long initial_seed()
            {
                return THSGenerator_initial_seed(Handle);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSGenerator_get_rng_state(IntPtr handle);

            [DllImport("LibTorchSharp")]
            extern static void THSGenerator_set_rng_state(IntPtr handle, IntPtr tensor);

            [DllImport("LibTorchSharp")]
            extern static void THSGenerator_gen_manual_seed(IntPtr handle, long seed);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSGenerator_new(ulong seed, long device_type, long device_index);

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSGenerator_default_generator();

            [DllImport("LibTorchSharp")]
            extern static void THSGenerator_dispose(IntPtr handle);

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
