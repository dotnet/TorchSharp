using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    using Debug = System.Diagnostics.Debug;

    public static class Torch
    {
        [DllImport("LibTorchSharp")]
        private static extern void THSTorch_seed(long seed);

        public static void SetSeed(long seed)
        {
            THSTorch_seed(seed);
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTorch_isCudaAvailable();

        public static bool IsCudaAvailable()
        {
            return THSTorch_isCudaAvailable();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTorch_get_and_reset_last_err();

        //[Conditional("DEBUG")]
        internal static void CheckForErrors()
        {
            var error = THSTorch_get_and_reset_last_err();

            if (error != IntPtr.Zero)
            {
                throw new ExternalException(Marshal.PtrToStringAnsi(error));
            }
        }
    }
}
