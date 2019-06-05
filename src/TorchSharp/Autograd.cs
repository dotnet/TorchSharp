using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public sealed class AutoGradMode : IDisposable
    {
        private bool _isPrevGrad;

        [DllImport("LibTorchSharp")]
        extern static bool THSAutograd_isGradEnabled();

        [DllImport("LibTorchSharp")]
        extern static void THSAutograd_setGrad(bool enabled);

        public AutoGradMode(bool enabled)
        {
            _isPrevGrad = THSAutograd_isGradEnabled();
            THSAutograd_setGrad(enabled);
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
                THSAutograd_setGrad(_isPrevGrad);
            }
        }

        public static bool IsAutogradEnabled()
        {
            return THSAutograd_isGradEnabled();
        }
    }
}
