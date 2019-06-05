using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public static class Torch
    {
        [DllImport("libTorchSharp")]
        extern static void THSTorch_seed(long seed);

        public static void SetSeed(long seed)
        {
            THSTorch_seed(seed);
        }

        [DllImport("libTorchSharp")]
        extern static bool THSTorch_isCudaAvailable();

        public static bool IsCudaAvailable()
        {
            return THSTorch_isCudaAvailable();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTorch_get_and_reset_last_err();

        [Conditional("DEBUG")]
        internal static void AssertNoErrors()
        {
            var error = THSTorch_get_and_reset_last_err();

            if (error != IntPtr.Zero)
            {
                throw new ExternalException(Marshal.PtrToStringAnsi(error));
            }
        }
    }
}
