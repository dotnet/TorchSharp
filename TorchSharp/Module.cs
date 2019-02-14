using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace TorchSharp.NN
{
    public partial class Module
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

        protected struct TensorPointerWrapper
        {
            public IntPtr ptr;
        }
    }

    public partial class Module : IDisposable
    {
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

        static public Sequential Sequential(params Module[] modules)
        {
            return new Sequential(modules);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr NN_linearModule(int input, int output, bool hasBias);

        static public Module Linear(int input, int output, bool hasBias = false)
        {
            return new Linear(NN_linearModule(input, output, hasBias));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr NN_reluModule();

        static public Module Relu()
        {
            return new Module(NN_reluModule());
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_functionalModule_Forward(Module.HType module, FloatTensor.HType tensor);

        public virtual FloatTensor Forward(FloatTensor tensor)
        {
            return new FloatTensor(NN_functionalModule_Forward(handle, tensor.handle));
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_functionalModule_ZeroGrad(Module.HType module);

        public virtual void ZeroGrad()
        {
            NN_functionalModule_ZeroGrad(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_functionalModule_GetParameters(Module.HType module, AllocatePinnedArray allocator);

        public virtual IEnumerable<FloatTensor> Parameters()
        {
            TensorPointerWrapper[] ros;

            using (var pa = new PinnedArray<TensorPointerWrapper>())
            {
                NN_functionalModule_GetParameters(handle, pa.CreateArray);
                ros = pa.Array;
            }
            return ros.Select(x => new FloatTensor(new FloatTensor.HType(x.ptr, true)));
        }

        [DllImport("LibTorchSharp")]
        extern static long NN_GetNumberOfChildren(Module.HType module);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string NN_GetModule(Module.HType module, int index);

        public virtual string[] GetModules()
        {
            var numModules = NN_GetNumberOfChildren(handle);
            string[] result = new string[numModules];

            for (int i = 0; i < numModules; i++)
            {
                result[i] = NN_GetModule(handle, i);
            }

            return result;
        }

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string NN_GetModuleName(Module.HType module);

        public string GetName()
        {
            return NN_GetModuleName(handle);
        }
    }
}
