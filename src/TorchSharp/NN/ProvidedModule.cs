using System;
using System.IO;
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
        extern static void THSNN_save_module(string location, HType handle);

        public override void Save(String modelPath)
        {
            if (File.Exists(modelPath))
            {
                throw new Exception(string.Format("{0} already existing.", modelPath));
            }

            THSNN_save_module(modelPath, handle);
        }
    }
}
