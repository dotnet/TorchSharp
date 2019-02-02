using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public class Module : IDisposable
    {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
        /// </summary>
        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            protected override bool ReleaseHandle()
            {
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal Module(IntPtr handle)
        {
            this.handle = new HType(handle, true);
        }

        ~Module()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr Module_load(string filename);

        static public Module LoadModule(string filename)
        {
            var handle = Module_load(filename);
            return new Module(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType Forward(Module.HType module, FloatTensor.HType tensor);

        public FloatTensor Score(FloatTensor tensor)
        {
            return new FloatTensor(Forward(handle, tensor.handle));
        }

        [DllImport("LibTorchSharp")]
        extern static int Get_number_of_modules(Module.HType module);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string Module_get(Module.HType module, int index);

        public string[] GetModules()
        {
            var numModules = Get_number_of_modules(handle);
            string[] result = new string[numModules];

            for (int i = 0; i < numModules; i++)
            {
                result[i] = Module_get(handle, i);
            }

            return result;
        }
    }
}
