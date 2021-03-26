// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable

namespace TorchSharp
{
    public class TorchGenerator : IDisposable
    {
        public IntPtr Handle { get; private set; }

        internal TorchGenerator(IntPtr nativeHandle)
        {
            Handle = nativeHandle;
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSGenerator_default_generator();

        public static TorchGenerator Default {
            get {
                return new TorchGenerator(THSGenerator_default_generator());
            }
        }

        [DllImport("LibTorchSharp")]
        extern static long THSGenerator_initial_seed(IntPtr handle);

        public long InitialSeed {
            get {
                return THSGenerator_initial_seed(Handle);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THSGenerator_dispose(IntPtr handle);

        protected virtual void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero) {
                THSGenerator_dispose(Handle);
                Handle = IntPtr.Zero;
            }
        }

        ~TorchGenerator()
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
    }
}
