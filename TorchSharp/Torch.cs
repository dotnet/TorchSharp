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
    }
}
