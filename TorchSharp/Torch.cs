using System;
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

    public class AutoGradMode : IDisposable
    {
        [DllImport("LibTorchSharp")]
        extern static bool THS_gradmode_is_enabled();

        [DllImport("LibTorchSharp")]
        extern static void THS_gradmode_set_enabled(bool enabled);

        public AutoGradMode(bool enabled)
        {
            prev_mode = THS_gradmode_is_enabled();
            THS_gradmode_set_enabled(enabled);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            if (disposing)
            {
                THS_gradmode_set_enabled(prev_mode);
            }
        }

        bool prev_mode;
    }
}
