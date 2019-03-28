using System.Runtime.InteropServices;

namespace TorchSharp
{
    public static class Torch
    {
        [DllImport("libTorchSharp")]
        extern static long NN_Seed(long seed);

        public static void SetSeed(long seed)
        {
            NN_Seed(seed);
        }
    }
}
