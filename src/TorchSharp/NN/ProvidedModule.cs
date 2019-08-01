using System;
using System.Runtime.InteropServices;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a module provided by Torch (e.g., Linear).
    /// </summary>
    public abstract class ProvidedModule : Module
    {
        internal ProvidedModule() : base(IntPtr.Zero)
        {
        }

        internal ProvidedModule(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static bool THSNN_save_module(HType handle, string location);

        public override void Save(String location)
        {
            if (!THSNN_save_module(handle, location))
            {
                throw new Exception("Error while saving the module.");
            }
        }
    }
}
