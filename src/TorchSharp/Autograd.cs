// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public sealed class AutoGradMode : IDisposable
    {
        private bool _isPrevGrad;

        [DllImport("LibTorchSharp")]
        private static extern bool THSAutograd_isGradEnabled();

        [DllImport("LibTorchSharp")]
        private static extern void THSAutograd_setGrad(bool enabled);

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
